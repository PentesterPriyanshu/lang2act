"""CLI entry point.

    python -m lang2act.main "pick up the block and move it to the target"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent
from .llm import LLMClient
from .robot import Robot
from .trace import Trace

DEFAULT_TASK = "pick up the block and place it on the target marker"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lang2act")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--trace", default="traces/run.jsonl")
    args = parser.parse_args(argv)

    llm = LLMClient()
    if not llm.health():
        print(
            "error: no model server at "
            f"{llm.base_url} — start it with scripts/serve.sh",
            file=sys.stderr,
        )
        return 1

    trace = Trace(Path(args.trace))
    robot = Robot(seed=args.seed)
    agent = Agent(llm, trace, max_steps=args.max_steps)

    print(f"task: {args.task}")
    result = agent.run_episode(args.task, robot)
    robot.close()

    print(f"\nplan: {result.plan}")
    print(f"env success:      {result.env_success}")
    print(f"verifier success: {result.verifier_success} — {result.verifier_assessment}")
    print(
        f"steps: {result.executor_steps} agent / {result.sim_steps} sim | "
        f"llm: {result.llm_calls} calls, {result.llm_time_s}s, "
        f"{result.prompt_tokens}+{result.completion_tokens} tokens | "
        f"wall: {result.wall_time_s}s"
    )
    print(f"trace: {args.trace}")
    return 0 if result.env_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
