"""Simulated Fetch arm wrapped in a small set of motion primitives.

The environment is gymnasium-robotics FetchPickAndPlace (MuJoCo). The agent
never emits raw joint/Cartesian deltas — it composes *primitives*, each of
which runs a closed-loop P-controller over many sim steps. This mirrors how
LLM agents drive real robotic platforms: language → skill primitives →
low-level control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Headless rendering: prefer EGL, fall back to OSMesa. Set before mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers Fetch envs)

TABLE_Z = 0.425          # table surface height (m)
SAFE_Z = 0.55            # travel height that clears the block
GRIP_TOL = 0.012         # position tolerance for primitives (m)
MAX_CTRL_STEPS = 60      # per-primitive control-loop budget


def _latest_env_id(prefix: str = "FetchPickAndPlace") -> str:
    """Resolve the newest registered version of the env (v2/v3/v4...)."""
    versions = [
        spec.id for spec in gym.registry.values() if spec.id.startswith(prefix + "-v")
    ]
    if not versions:
        raise RuntimeError(f"No {prefix} env registered — is gymnasium-robotics installed?")
    return max(versions, key=lambda s: int(s.rsplit("-v", 1)[1]))


@dataclass
class Robot:
    """Fetch arm + tabletop block + target site, with skill primitives."""

    seed: int | None = None
    render_width: int = 480
    render_height: int = 480
    env: gym.Env = field(default=None, repr=False)  # type: ignore[assignment]
    _obs: dict = field(default=None, repr=False)    # type: ignore[assignment]
    _last_info: dict = field(default_factory=dict, repr=False)
    _grip_cmd: float = 1.0  # last commanded gripper action: +1 open, -1 closed
    steps_taken: int = 0

    def __post_init__(self):
        self.env = gym.make(
            _latest_env_id(),
            render_mode="rgb_array",
            width=self.render_width,
            height=self.render_height,
        )
        self.reset(self.seed)

    # ---------------------------------------------------------------- state

    def reset(self, seed: int | None = None) -> None:
        self._obs, self._last_info = self.env.reset(seed=seed)
        self._grip_cmd = 1.0
        self.steps_taken = 0

    @property
    def gripper_pos(self) -> np.ndarray:
        return self._obs["observation"][:3].copy()

    @property
    def object_pos(self) -> np.ndarray:
        return self._obs["observation"][3:6].copy()

    @property
    def goal_pos(self) -> np.ndarray:
        return self._obs["desired_goal"].copy()

    @property
    def gripper_opening(self) -> float:
        # two finger joint positions; sum ≈ 0.10 fully open, ≈ 0.0 closed
        return float(self._obs["observation"][9:11].sum())

    @property
    def guidance(self) -> str:
        """Task-family-specific rules injected into the agent prompts."""
        return (
            "If the target z is above the table surface (z > 0.47), the block "
            "must be HELD at the target — do not open the gripper or it will "
            "fall. Call done() while holding it in place. Only release for "
            "table-level targets."
        )

    def is_success(self) -> bool:
        return bool(self._last_info.get("is_success", False))

    def camera(self) -> np.ndarray:
        """Current RGB camera frame (HxWx3 uint8)."""
        return self.env.render()

    def state_text(self) -> str:
        g, o, t = self.gripper_pos, self.object_pos, self.goal_pos
        grip = "open" if self.gripper_opening > 0.05 else "closed"
        return (
            f"gripper at ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}), fingers {grip}\n"
            f"block at   ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f})\n"
            f"target at  ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})\n"
            f"block-to-target distance: {np.linalg.norm(o - t):.3f} m"
        )

    # ----------------------------------------------------------- primitives

    def _step(self, action: np.ndarray) -> None:
        self._obs, _r, _te, _tr, self._last_info = self.env.step(
            np.clip(action, -1.0, 1.0).astype(np.float32)
        )
        self.steps_taken += 1

    def _servo(self, target: np.ndarray, gripper: float) -> bool:
        """P-control the end-effector to `target`; returns True on convergence."""
        for _ in range(MAX_CTRL_STEPS):
            delta = target - self.gripper_pos
            if np.linalg.norm(delta) < GRIP_TOL:
                return True
            action = np.zeros(4)
            action[:3] = delta * 20.0
            action[3] = gripper
            self._step(action)
        return bool(np.linalg.norm(target - self.gripper_pos) < GRIP_TOL * 2)

    def go_to(self, x: float, y: float, z: float) -> str:
        """Move end-effector to (x, y, z), travelling at safe height between
        columns so the arm doesn't plough through the block."""
        z = max(z, TABLE_Z - 0.005)
        hold = self._grip_cmd  # never change grip state mid-move
        here, there = self.gripper_pos, np.array([x, y, z])
        # If the lateral move is large, lift → traverse → descend.
        if np.linalg.norm(there[:2] - here[:2]) > 0.03:
            self._servo(np.array([here[0], here[1], max(SAFE_Z, here[2])]), hold)
            self._servo(np.array([x, y, max(SAFE_Z, z)]), hold)
        ok = self._servo(there, hold)
        g = self.gripper_pos
        return (
            f"{'arrived at' if ok else 'stopped near'} "
            f"({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})"
        )

    def open_gripper(self) -> str:
        self._grip_cmd = 1.0
        for _ in range(6):
            self._step(np.array([0, 0, 0, 1.0]))
        return f"gripper opened (width {self.gripper_opening:.3f})"

    def close_gripper(self) -> str:
        self._grip_cmd = -1.0
        for _ in range(10):
            self._step(np.array([0, 0, 0, -1.0]))
        grasped = self.gripper_opening > 0.01 and (
            np.linalg.norm(self.object_pos - self.gripper_pos) < 0.03
        )
        return (
            f"gripper closed (width {self.gripper_opening:.3f}); "
            f"{'block appears grasped' if grasped else 'nothing grasped'}"
        )

    def close(self) -> None:
        self.env.close()
