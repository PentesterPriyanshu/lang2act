"""Scripted-oracle sanity check for the puzzle tasks.

Solves stack / tower / swap with hand-coded primitive sequences to prove the
environment, physics, and success checks are sound before any LLM touches it.

Run: .venv/bin/python -m scripts.sanity_blocks
"""

from __future__ import annotations

import numpy as np

from lang2act.blocks_robot import BLOCK_H, TABLE_Z, BlocksRobot


def move_block(r: BlocksRobot, color: str, dest: np.ndarray) -> None:
    src = r.block(color)
    r.open_gripper()
    r.go_to(src[0], src[1], src[2])
    r.close_gripper()
    r.go_to(dest[0], dest[1], dest[2] + 0.003)
    r.open_gripper()
    r.go_to(dest[0], dest[1], dest[2] + 0.10)  # retreat


def solve_stack(r: BlocksRobot) -> None:
    red = r.block("red")
    move_block(r, "green", red + [0, 0, BLOCK_H])


def solve_tower(r: BlocksRobot) -> None:
    red = r.block("red")
    move_block(r, "green", red + [0, 0, BLOCK_H])
    move_block(r, "blue", red + [0, 0, 2 * BLOCK_H])


def solve_swap(r: BlocksRobot) -> None:
    red0, blue0 = r.initial["red"], r.initial["blue"]
    # temp spot: a free corner of the workspace away from all blocks
    temp = np.array([1.20, 0.62, TABLE_Z])
    while any(np.linalg.norm(temp[:2] - r.block(c)[:2]) < 0.09
              for c in ("red", "green", "blue")):
        temp[1] += 0.08
    move_block(r, "red", temp)
    move_block(r, "blue", red0)
    move_block(r, "red", blue0)


def main() -> int:
    solvers = {"stack": solve_stack, "tower": solve_tower, "swap": solve_swap}
    failures = 0
    for task, solver in solvers.items():
        for seed in (0, 1, 2):
            r = BlocksRobot(seed=seed, task=task)
            solver(r)
            ok = r.is_success()
            failures += (not ok)
            print(f"{task:6s} seed {seed}: {'OK ' if ok else 'FAIL'} "
                  f"(sim steps {r.steps_taken})")
            r.close()
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
