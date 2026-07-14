"""Task layer for the multi-block puzzle environment.

Three task tiers of increasing reasoning depth:

  stack — "put green on red": spatial precision + placement height.
  tower — "red, green, blue tower": ordering constraints; a wrong order
          physically forces an undo.
  swap  — "swap red and blue": the puzzle. One gripper, both destination
          spots occupied — the agent must invent a temporary location.

Success is judged geometrically from ground-truth block poses; the agent
never sees these checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocks_env import BLOCK_COLORS, BlocksFetchEnv
from .robot import Robot

BLOCK_H = 0.05           # cube edge (m)
TABLE_Z = 0.425          # resting height of a cube on the table
ON_XY_TOL = 0.03         # "stacked on" lateral tolerance
SWAP_XY_TOL = 0.055      # "took the other's spot" tolerance

TASKS = {
    "stack": {
        "text": "put the green block on top of the red block",
        "max_steps": 14,
    },
    "tower": {
        "text": ("stack the blocks into a tower on the table: red at the "
                 "bottom, green in the middle, blue on top"),
        "max_steps": 22,
    },
    "swap": {
        "text": ("swap the positions of the red and blue blocks: red must end "
                 "up where blue started and blue where red started, both on "
                 "the table. Do not move the green block."),
        "max_steps": 24,
    },
}


@dataclass
class BlocksRobot(Robot):
    """Fetch arm over three colored blocks, with puzzle-task success checks."""

    task: str = "stack"

    def __post_init__(self):
        self.env = BlocksFetchEnv(
            render_mode="rgb_array",
            width=self.render_width,
            height=self.render_height,
        )
        self.initial: dict[str, np.ndarray] = {}
        self.reset(self.seed)

    # ---------------------------------------------------------------- state

    def reset(self, seed: int | None = None) -> None:
        self._obs, self._last_info = self.env.reset(seed=seed)
        self._grip_cmd = 1.0
        self.steps_taken = 0
        self.initial = {c: self.block(c) for c in BLOCK_COLORS}

    def block(self, color: str) -> np.ndarray:
        return self.env.block_pos(BLOCK_COLORS[color][0])

    @property
    def object_pos(self) -> np.ndarray:  # nearest block (grasp feedback)
        g = self.gripper_pos
        return min((self.block(c) for c in BLOCK_COLORS),
                   key=lambda p: np.linalg.norm(p - g))

    @property
    def goal_pos(self) -> np.ndarray:  # no goal marker in the puzzle scene
        return np.zeros(3)

    @property
    def task_text(self) -> str:
        return TASKS[self.task]["text"]

    @property
    def max_steps(self) -> int:
        return TASKS[self.task]["max_steps"]

    def state_text(self) -> str:
        g = self.gripper_pos
        grip = "open" if self.gripper_opening > 0.05 else "closed"
        lines = [f"gripper at ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}), fingers {grip}"]
        for color in BLOCK_COLORS:
            p = self.block(color)
            lines.append(f"{color:5s} block at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        if self.task == "swap":
            r0, b0 = self.initial["red"], self.initial["blue"]
            lines.append(f"red   started at ({r0[0]:.3f}, {r0[1]:.3f}, {r0[2]:.3f})")
            lines.append(f"blue  started at ({b0[0]:.3f}, {b0[1]:.3f}, {b0[2]:.3f})")
        lines.append("blocks are 5 cm cubes; table surface z=0.425; "
                     "a block resting on another has z ~= base z + 0.050")
        return "\n".join(lines)

    @property
    def guidance(self) -> str:
        return (
            "Rules for working with blocks:\n"
            "- Moving a block X onto a base Y is ALWAYS this exact sequence:\n"
            "    1. open_gripper\n"
            "    2. go_to X's exact (x, y, z) from the state readout\n"
            "    3. close_gripper  — the result MUST say 'holding the X block'\n"
            "    4. go_to (Y.x, Y.y, Y.z + 0.050)   [or (x, y, 0.428) for a "
            "table spot]\n"
            "    5. open_gripper   — the block is released there\n"
            "- If close_gripper reports holding the WRONG block or nothing: "
            "open_gripper, go_to the intended block's CURRENT coordinates, "
            "close_gripper again.\n"
            "- Never grasp a block that has another block resting on it.\n"
            "- If a destination spot is occupied, first move that block to a "
            "free temporary spot on the table.\n"
            "- Block positions in the state readout are live — re-read them "
            "after every placement."
        )

    def close_gripper(self) -> str:  # color-aware grasp feedback
        self._grip_cmd = -1.0
        for _ in range(10):
            self._step(np.array([0, 0, 0, -1.0]))
        g = self.gripper_pos
        held = [c for c in BLOCK_COLORS
                if np.linalg.norm(self.block(c) - g) < 0.032]
        what = (f"holding the {held[0]} block" if len(held) == 1
                else "closed on nothing — no block between the fingers")
        return f"gripper closed (width {self.gripper_opening:.3f}); {what}"

    # -------------------------------------------------------------- success

    def _on(self, top: str, base: str) -> bool:
        t, b = self.block(top), self.block(base)
        return bool(np.linalg.norm(t[:2] - b[:2]) < ON_XY_TOL
                    and 0.03 < (t[2] - b[2]) < 0.075)

    def _at(self, color: str, spot: np.ndarray, tol: float) -> bool:
        p = self.block(color)
        return bool(np.linalg.norm(p[:2] - spot[:2]) < tol
                    and abs(p[2] - TABLE_Z) < 0.015)

    def is_success(self) -> bool:
        if self.task == "stack":
            return self._on("green", "red")
        if self.task == "tower":
            return self._on("green", "red") and self._on("blue", "green")
        if self.task == "swap":
            return (self._at("red", self.initial["blue"], SWAP_XY_TOL)
                    and self._at("blue", self.initial["red"], SWAP_XY_TOL)
                    and self._at("green", self.initial["green"], 0.08))
        raise ValueError(f"unknown task {self.task!r}")
