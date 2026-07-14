"""Record a demo GIF of the pick-and-place task.

Runs the scripted primitive sequence (same skills the agent composes) while
capturing sim frames, and writes docs/demo.gif for the README.

Run: .venv/bin/python -m scripts.record_demo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lang2act.robot import Robot

OUT = Path("docs/demo.gif")
CAPTURE_EVERY = 2   # sim steps between frames
SIZE = 320          # GIF edge length (px)


class RecordingRobot(Robot):
    frames: list

    def __post_init__(self):
        self.frames = []
        super().__post_init__()

    def _step(self, action: np.ndarray) -> None:
        super()._step(action)
        if self.steps_taken % CAPTURE_EVERY == 0:
            self.frames.append(self.camera())


def main() -> int:
    robot = RecordingRobot(seed=3)
    o, t = robot.object_pos, robot.goal_pos
    robot.open_gripper()
    robot.go_to(o[0], o[1], o[2] + 0.08)
    robot.go_to(o[0], o[1], o[2])
    robot.close_gripper()
    robot.go_to(t[0], t[1], max(t[2], o[2] + 0.05))
    robot.go_to(t[0], t[1], t[2])
    if t[2] < 0.47:
        robot.open_gripper()
        robot.go_to(t[0], t[1], t[2] + 0.10)
    success = robot.is_success()
    # hold the final frame for a beat
    robot.frames.extend([robot.camera()] * 8)
    robot.close()

    imgs = []
    for f in robot.frames:
        img = Image.fromarray(f)
        img.thumbnail((SIZE, SIZE))
        imgs.append(img)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    print(f"success={success}  frames={len(imgs)}  wrote {OUT} "
          f"({OUT.stat().st_size / 1048576:.1f} MB)")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
