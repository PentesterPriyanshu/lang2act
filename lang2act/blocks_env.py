"""Multi-block Fetch environment for blocksworld-style puzzle tasks.

gymnasium-robotics only ships single-object Fetch scenes, so this module
generates a 3-block variant of the pick-and-place MJCF at runtime (patching
the stock XML with two extra colored blocks and absolute asset paths) and
subclasses MujocoFetchEnv to scatter the blocks on reset.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv

BLOCK_COLORS = {          # color name -> (object index, rgba)
    "red":   (0, "0.90 0.15 0.15 1"),
    "green": (1, "0.15 0.70 0.20 1"),
    "blue":  (2, "0.20 0.40 0.95 1"),
}
MIN_BLOCK_GAP = 0.10      # min pairwise xy distance at reset (m)

_BODY_TEMPLATE = """\
\t\t<body name="object{i}" pos="0.025 0.025 0.025">
\t\t\t<joint name="object{i}:joint" type="free" damping="0.01"></joint>
\t\t\t<geom size="0.025 0.025 0.025" type="box" condim="3" name="object{i}" material="block_mat" rgba="{rgba}" mass="2"></geom>
\t\t\t<site name="object{i}" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 0" type="sphere"></site>
\t\t</body>
"""


def _generate_xml() -> str:
    """Patch the stock pick_and_place.xml into a 3-block scene; return path."""
    import gymnasium_robotics

    assets = Path(gymnasium_robotics.__file__).parent / "envs" / "assets"
    src = (assets / "fetch" / "pick_and_place.xml").read_text()

    # Absolute asset paths so the generated file can live anywhere.
    src = src.replace('meshdir="../stls/fetch"', f'meshdir="{assets / "stls" / "fetch"}"')
    src = src.replace('texturedir="../textures"', f'texturedir="{assets / "textures"}"')
    src = src.replace('file="shared.xml"', f'file="{assets / "fetch" / "shared.xml"}"')
    src = src.replace('file="robot.xml"', f'file="{assets / "fetch" / "robot.xml"}"')

    # Hide the target marker — puzzle goals are block configurations.
    src = src.replace('rgba="1 0 0 1" type="sphere"></site>\n\t\t</body>',
                      'rgba="1 0 0 0" type="sphere"></site>\n\t\t</body>', 1)

    # Color object0 red and silence its site marker.
    src = src.replace('name="object0" material="block_mat" mass="2"',
                      f'name="object0" material="block_mat" rgba="{BLOCK_COLORS["red"][1]}" mass="2"')
    src = re.sub(r'(<site name="object0"[^>]*rgba=")[^"]*(")', r"\g<1>1 0 0 0\g<2>", src)

    # Add green + blue blocks after object0's body.
    extra = "".join(
        _BODY_TEMPLATE.format(i=idx, rgba=rgba)
        for name, (idx, rgba) in BLOCK_COLORS.items() if idx > 0
    )
    src = src.replace("\t\t<light", extra + "\t\t<light", 1)

    out = Path(tempfile.gettempdir()) / f"lang2act_blocks_{os.getpid()}.xml"
    out.write_text(src)
    return str(out)


class BlocksFetchEnv(MujocoFetchEnv, EzPickle):
    """Fetch arm + three colored 5 cm blocks, no fixed goal marker."""

    def __init__(self, **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.60, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.32, 0.75, 0.4, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.25, 0.90, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=_generate_xml(),
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.13,
            target_range=0.0,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type="sparse",
            **kwargs,
        )
        EzPickle.__init__(self, **kwargs)

    # goals are block configurations checked by the task layer, not a site
    def _sample_goal(self):
        return np.zeros(3)

    def _render_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Scatter the three blocks: pairwise-separated, away from the gripper.
        center = self.initial_gripper_xpos[:2]
        spots: list[np.ndarray] = []
        while len(spots) < 3:
            cand = center + self.np_random.uniform(-self.obj_range, self.obj_range, 2)
            if np.linalg.norm(cand - center) < 0.07:
                continue
            if any(np.linalg.norm(cand - s) < MIN_BLOCK_GAP for s in spots):
                continue
            spots.append(cand)
        for i, spot in enumerate(spots):
            joint = f"object{i}:joint"
            qpos = self._utils.get_joint_qpos(self.model, self.data, joint)
            qpos[:2] = spot
            qpos[2] = 0.425          # resting height for a 5 cm cube
            qpos[3:] = [1, 0, 0, 0]  # upright
            self._utils.set_joint_qpos(self.model, self.data, joint, qpos)

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def block_pos(self, index: int) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, f"object{index}").copy()
