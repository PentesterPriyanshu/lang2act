"""Record real agent episodes into replayable bundles for the demo Space.

For each (task, seed): runs the actual planner/executor/verifier agent
against the local model server, capturing every camera frame during motion
plus the agent's thought/action/result at each step.

Output per episode: space/assets/ep_<task>_<seed>/
    episode.json   plan, steps (with frame ranges), verdict, metrics
    f<NNNN>.jpg    all motion frames, step-aligned

Run (llama-server must be up):
    .venv/bin/python -m space.record_episode --task stack --seeds 0 1 2
    .venv/bin/python -m space.record_episode --task pickplace --seeds 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lang2act.agent import Agent
from lang2act.blocks_robot import TASKS, BlocksRobot
from lang2act.llm import LLMClient
from lang2act.robot import Robot
from lang2act.trace import Trace

PICKPLACE_TASK = "pick up the block and place it on the target marker"
CAPTURE_EVERY = 2
SIZE = 384


def _recording(base):
    """Robot subclass that captures frames during motion."""

    class FrameRobot(base):
        frames: list

        def __post_init__(self):
            self.frames = []
            super().__post_init__()
            self.frames.append(self.camera())

        def _step(self, action: np.ndarray) -> None:
            super()._step(action)
            if self.steps_taken % CAPTURE_EVERY == 0:
                self.frames.append(self.camera())

    return FrameRobot


def record(task: str, seed: int, out_root: Path) -> dict:
    out = out_root / f"ep_{task}_{seed}" if task != "pickplace" else out_root / f"ep_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    assert llm.health(), "llama-server is not up — run scripts/serve.sh"

    if task == "pickplace":
        robot = _recording(Robot)(seed=seed)
        task_text, max_steps = PICKPLACE_TASK, 12
    else:
        robot = _recording(BlocksRobot)(seed=seed, task=task)
        task_text, max_steps = robot.task_text, robot.max_steps

    steps: list[dict] = []
    last_idx = 0

    def on_step(step: int, action: dict, result) -> None:
        nonlocal last_idx
        robot.frames.append(robot.camera())
        steps.append({
            "step": step,
            "thought": action.get("thought", ""),
            "tool": action["tool"],
            "args": {k: v for k, v in action.items() if k in ("x", "y", "z", "reason")},
            "result": result.text,
            "frame_start": last_idx,
            "frame_end": len(robot.frames) - 1,
        })
        last_idx = len(robot.frames) - 1

    agent = Agent(llm, Trace(out / "trace.jsonl"), max_steps=max_steps,
                  step_callback=on_step)
    r = agent.run_episode(task_text, robot)

    for i, f in enumerate(robot.frames):
        img = Image.fromarray(f)
        img.thumbnail((SIZE, SIZE))
        img.save(out / f"f{i:04d}.jpg", quality=82)

    episode = {
        "seed": seed,
        "task_id": task,
        "task": task_text,
        "plan": r.plan,
        "steps": steps,
        "n_frames": len(robot.frames),
        "env_success": r.env_success,
        "verifier_success": r.verifier_success,
        "verifier_assessment": r.verifier_assessment,
        "llm_calls": r.llm_calls,
        "wall_time_s": r.wall_time_s,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
    }
    (out / "episode.json").write_text(json.dumps(episode, indent=2))
    robot.close()
    print(f"{task} seed {seed}: env={'OK' if r.env_success else 'FAIL'} "
          f"steps={len(steps)} frames={len(robot.frames)} wall={r.wall_time_s}s")
    return episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stack",
                        choices=["pickplace", *TASKS.keys()])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    out_root = Path(__file__).parent / "assets"
    for seed in args.seeds:
        record(args.task, seed, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
