"""Evaluation harness: N episodes -> success rate, latency, token counts.

Ground truth comes from the environment (block within 5 cm of target), and
the verifier's judgment is scored against it — so the harness measures both
task success AND how well the VLM verifier agrees with reality.

Run:  .venv/bin/python -m eval.run_eval --episodes 10
Writes eval/results.json and prints a markdown summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lang2act.agent import Agent
from lang2act.llm import LLMClient
from lang2act.robot import Robot
from lang2act.trace import Trace

TASK = "pick up the block and place it on the target marker"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    llm = LLMClient()
    assert llm.health(), "llama-server is not up — run scripts/serve.sh"

    results = []
    for ep in range(args.episodes):
        trace = Trace(Path(f"traces/eval_ep{ep:03d}.jsonl"))
        robot = Robot(seed=ep)
        agent = Agent(llm, trace, max_steps=args.max_steps)
        r = agent.run_episode(TASK, robot)
        robot.close()
        results.append(r.__dict__)
        print(f"ep {ep:02d}: env={'OK' if r.env_success else 'fail'} "
              f"verifier={'OK' if r.verifier_success else 'fail'} "
              f"steps={r.executor_steps} wall={r.wall_time_s}s")

    n = len(results)
    env_rate = sum(r["env_success"] for r in results) / n
    ver_agree = sum(r["env_success"] == r["verifier_success"] for r in results) / n
    summary = {
        "episodes": n,
        "task": TASK,
        "success_rate": env_rate,
        "verifier_agreement": ver_agree,
        "mean_wall_s": round(statistics.mean(r["wall_time_s"] for r in results), 1),
        "mean_llm_calls": round(statistics.mean(r["llm_calls"] for r in results), 1),
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in results)),
        "mean_completion_tokens": round(
            statistics.mean(r["completion_tokens"] for r in results)),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n| metric | value |")
    print("|---|---|")
    print(f"| episodes | {n} |")
    print(f"| task success rate | {env_rate:.0%} |")
    print(f"| verifier agreement with ground truth | {ver_agree:.0%} |")
    print(f"| mean wall time / episode | {summary['mean_wall_s']}s |")
    print(f"| mean LLM calls / episode | {summary['mean_llm_calls']} |")
    print(f"| mean tokens / episode | {summary['mean_prompt_tokens']}"
          f"+{summary['mean_completion_tokens']} |")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
