"""Sim-only test: scripted pick-and-place using the motion primitives.

No LLM involved — this validates that the primitive layer (go_to /
open_gripper / close_gripper) is sufficient to solve the task when driven
by oracle positions. If this fails, no agent on top of it can succeed.

Run: .venv/bin/python -m tests.test_robot
"""

from lang2act.robot import Robot


def scripted_episode(seed: int) -> bool:
    robot = Robot(seed=seed)
    try:
        o = robot.object_pos
        t = robot.goal_pos
        robot.open_gripper()
        robot.go_to(o[0], o[1], o[2] + 0.08)   # hover above block
        robot.go_to(o[0], o[1], o[2])          # descend to block
        robot.close_gripper()                  # grasp
        robot.go_to(t[0], t[1], max(t[2], o[2] + 0.05))  # carry to target
        robot.go_to(t[0], t[1], t[2])          # settle at target height
        if t[2] < 0.47:                        # table-level goal: set it down
            robot.open_gripper()
            robot.go_to(t[0], t[1], t[2] + 0.10)
        # air goal: keep holding the block at the target
        return robot.is_success()
    finally:
        robot.close()


def main() -> int:
    results = {}
    for seed in (0, 1, 2, 3, 4):
        ok = scripted_episode(seed)
        results[seed] = ok
        print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL'}")
    rate = sum(results.values()) / len(results)
    print(f"\nscripted success rate: {rate:.0%}")
    assert rate >= 0.8, f"primitive layer too weak: {results}"
    print("PRIMITIVES OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
