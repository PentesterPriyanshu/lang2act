"""The lang2act agent: planner → executor → verifier.

Three roles with isolated contexts, one locally-served VLM behind them:

  planner   — sees the task + first camera frame, emits a short step plan.
  executor  — drives the robot through grammar-constrained tool calls.
  verifier  — fresh context; judges the *final camera frame* against the
              task, independent of what the executor believes it did.

Context engineering for CPU-friendly prompts: only the most recent camera
frame is kept in the executor's context; older frames collapse to a text
placeholder. History is rebuilt each turn from a compact action log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .llm import LLMClient, image_content, text_content
from .robot import Robot
from .tools import ACTION_SCHEMA, TOOL_DOCS, ToolResult, dispatch
from .trace import Trace

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "array", "items": {"type": "string", "maxLength": 120},
                 "minItems": 2, "maxItems": 8},
    },
    "required": ["plan"],
    "additionalProperties": False,
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "assessment": {"type": "string", "maxLength": 250},
    },
    "required": ["success", "assessment"],
    "additionalProperties": False,
}

EXECUTOR_SYSTEM = """\
You control a Fetch robot arm on a tabletop. You act by replying with exactly
one JSON action per turn — no prose outside the JSON.

{tool_docs}
Example action (one JSON object, nothing else):
{{"thought": "descend to the block", "tool": "go_to", "x": 1.34, "y": 0.75, "z": 0.42}}

Ground rules:
- Positions are metres in the world frame; the numeric state is authoritative.
- go_to always needs numeric x, y, z copied from the state readout.
- To grasp: open_gripper, go_to the block's exact (x, y, z), close_gripper.
- To place: go_to the target (x, y, z) while holding, then open_gripper.
- After open/close, the result message tells you whether the grasp held.
- Call done(reason) as soon as the block is at the target.

Your plan:
{plan}
"""


@dataclass
class EpisodeResult:
    task: str
    env_success: bool
    verifier_success: bool
    verifier_assessment: str
    executor_steps: int
    sim_steps: int
    llm_calls: int
    llm_time_s: float
    wall_time_s: float
    prompt_tokens: int
    completion_tokens: int
    plan: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, llm: LLMClient, trace: Trace, max_steps: int = 12):
        self.llm = llm
        self.trace = trace
        self.max_steps = max_steps
        self._calls = 0
        self._llm_time = 0.0
        self._ptok = 0
        self._ctok = 0

    # ------------------------------------------------------------------ llm

    def _chat(self, messages: list[dict], schema: dict) -> dict:
        result = self.llm.chat(messages, json_schema=schema)
        self._calls += 1
        self._llm_time += result.latency_s
        self._ptok += result.prompt_tokens
        self._ctok += result.completion_tokens
        return result.json()

    # ---------------------------------------------------------------- roles

    def plan(self, task: str, robot: Robot) -> list[str]:
        reply = self._chat(
            [
                {"role": "system", "content":
                    "You plan for a tabletop robot arm. Reply with a short plan "
                    "of concrete steps. The numeric state is authoritative."},
                {"role": "user", "content": [
                    image_content(robot.camera()),
                    text_content(
                        f"Task: {task}\n\nScene state:\n{robot.state_text()}\n\n"
                        f"The robot has these skills:\n{TOOL_DOCS}"
                    ),
                ]},
            ],
            PLAN_SCHEMA,
        )
        self.trace.log("plan", plan=reply["plan"])
        return reply["plan"]

    def execute(self, task: str, plan: list[str], robot: Robot) -> tuple[int, bool]:
        """Run the tool loop. Returns (executor_steps, model_called_done)."""
        system = EXECUTOR_SYSTEM.format(
            tool_docs=TOOL_DOCS, plan="\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        )
        history: list[tuple[str, str, bool]] = []  # (action_json, result_text, had_image)
        latest_frame = robot.camera()

        for step in range(1, self.max_steps + 1):
            messages: list[dict] = [{"role": "system", "content": system}]
            intro = f"Task: {task}\n\nCurrent scene state:\n{robot.state_text()}"
            messages.append({"role": "user", "content": [
                image_content(latest_frame), text_content(intro)
            ]})
            for act_json, result_text, had_image in history:
                messages.append({"role": "assistant", "content": act_json})
                note = " [camera frame was shown here]" if had_image else ""
                messages.append({"role": "user", "content": f"result: {result_text}{note}"})

            action = self._chat(messages, ACTION_SCHEMA)
            self.trace.log("action", step=step, **action)

            result: ToolResult = dispatch(robot, action)
            self.trace.log("result", step=step, text=result.text)
            if result.is_done:
                return step, True
            if result.image is not None:
                latest_frame = result.image
            import json as _json
            history.append((_json.dumps(action), result.text, result.image is not None))
            # keep the rolling window small — old exchanges beyond 6 drop off
            history = history[-6:]
        return self.max_steps, False

    def verify(self, task: str, robot: Robot) -> dict:
        reply = self._chat(
            [
                {"role": "system", "content":
                    "You are a strict quality inspector for a robot arm. Judge only "
                    "from the image and numeric state whether the task was completed."},
                {"role": "user", "content": [
                    image_content(robot.camera()),
                    text_content(f"Task was: {task}\n\nFinal state:\n{robot.state_text()}\n"
                                 "Was the task completed successfully?"),
                ]},
            ],
            VERDICT_SCHEMA,
        )
        self.trace.log("verdict", **reply)
        return reply

    # -------------------------------------------------------------- episode

    def run_episode(self, task: str, robot: Robot) -> EpisodeResult:
        t0 = time.monotonic()
        self._calls = 0
        self._llm_time = 0.0
        self._ptok = self._ctok = 0

        plan = self.plan(task, robot)
        steps, _done = self.execute(task, plan, robot)
        verdict = self.verify(task, robot)

        result = EpisodeResult(
            task=task,
            env_success=robot.is_success(),
            verifier_success=bool(verdict["success"]),
            verifier_assessment=verdict["assessment"],
            executor_steps=steps,
            sim_steps=robot.steps_taken,
            llm_calls=self._calls,
            llm_time_s=round(self._llm_time, 2),
            wall_time_s=round(time.monotonic() - t0, 2),
            prompt_tokens=self._ptok,
            completion_tokens=self._ctok,
            plan=plan,
        )
        self.trace.log("episode_end", **result.__dict__)
        return result
