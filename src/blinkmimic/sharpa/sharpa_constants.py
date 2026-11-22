"""Sharpa right-hand RobotSpec.

The MJCF is generated from the licensed FlashDex Sharpa source asset. Its six
serial wrist joints are folded into a free root by ``tools/build_sharpa_mjcf.py``.
Only the 22 finger joints are action dimensions in this adapter.
"""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from blinkmimic.robot import MotionArrays, RobotSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ASSET_DIR = os.path.join(REPO, "assets", "sharpa")
MJCF_PATH = os.path.join(ASSET_DIR, "sharpa_right.xml")
MOTION_DIR = os.path.join(REPO, "data", "oakink2_sharpa")

JOINT_NAMES = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
]  # fmt: skip

TRACKED_BODIES = [
    "sharpa_root", "right_thumb_DP", "right_index_DP", "right_middle_DP", "right_ring_DP", "right_pinky_DP",
]  # fmt: skip

# (stiffness, effort limit, armature). The stiffness and effort values are copied
# from the source XML position actuators. Damping uses its 0.9 source dampratio
# against the joint armature as the first BlinkMimic actuator approximation.
_ACTUATOR = {
    "right_thumb_CMC_FE": (6.95, 3.3, 0.0032),
    "right_thumb_CMC_AA": (13.2, 3.3, 0.0032),
    "right_thumb_MCP_FE": (4.76, 1.864, 0.00265),
    "right_thumb_MCP_AA": (6.62, 1.864, 0.00265),
    "right_thumb_IP": (0.9, 0.638, 0.0006),
    "right_index_MCP_FE": (4.76, 1.864, 0.00265),
    "right_index_MCP_AA": (6.62, 1.864, 0.00265),
    "right_index_PIP": (0.9, 0.638, 0.0006),
    "right_index_DIP": (0.9, 0.18937, 0.00042),
    "right_middle_MCP_FE": (4.76, 1.864, 0.00265),
    "right_middle_MCP_AA": (6.62, 1.864, 0.00265),
    "right_middle_PIP": (0.9, 0.638, 0.0006),
    "right_middle_DIP": (0.9, 0.18937, 0.00042),
    "right_ring_MCP_FE": (4.76, 1.864, 0.00265),
    "right_ring_MCP_AA": (6.62, 1.864, 0.00265),
    "right_ring_PIP": (0.9, 0.638, 0.0006),
    "right_ring_DIP": (0.9, 0.18937, 0.00042),
    "right_pinky_CMC": (1.38, 0.5285, 0.00012),
    "right_pinky_MCP_FE": (4.76, 1.864, 0.00265),
    "right_pinky_MCP_AA": (6.62, 1.864, 0.00265),
    "right_pinky_PIP": (0.9, 0.638, 0.0006),
    "right_pinky_DIP": (0.9, 0.18937, 0.00042),
}


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(MJCF_PATH)


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    out = []
    for name in JOINT_NAMES:
        stiffness, effort, armature = _ACTUATOR[name]
        damping = 2.0 * 0.9 * math.sqrt(stiffness * armature)
        out.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort,
            armature=armature,
        ))
    return tuple(out)


def _load_npz(path: str) -> MotionArrays:
    """Load a preprocessed Sharpa clip, keeping OakInk2 retargeting explicit.

    The raw OakInk2-derived files currently contain XHand bimanual ``qpos`` and
    are intentionally rejected here. A converted clip must provide root pose and
    the 22 Sharpa joint positions under the names below.
    """
    with np.load(path, allow_pickle=False) as data:
        required = ("root_pos", "root_quat_wxyz", "joint_pos", "fps")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing Sharpa fields {missing}; convert from OakInk2 first")
        p = np.asarray(data["root_pos"], dtype=np.float32)
        quat = np.asarray(data["root_quat_wxyz"], dtype=np.float32)
        joints = np.asarray(data["joint_pos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).item())
    if p.ndim != 2 or p.shape[1] != 3 or quat.shape != (len(p), 4) or joints.shape != (len(p), len(JOINT_NAMES)):
        raise ValueError(f"{path}: expected root (T,3), quaternion (T,4), joints (T,22)")
    if not np.isfinite(np.concatenate((p, quat, joints), axis=1)).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: non-finite values or invalid fps")
    return p, quat, joints, fps


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )


SPEC = RobotSpec(
    name="sharpa",
    mjcf_path=MJCF_PATH,
    joint_names=tuple(JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    default_joint_pos={".*": 0.0},
    action_scale={name: 0.25 * _ACTUATOR[name][1] / _ACTUATOR[name][0] for name in JOINT_NAMES},
    com_body="sharpa_root",
    com_range=((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)),
    robot_cfg=get_robot_cfg,
    motion_dir=MOTION_DIR,
    motion_glob="*.npz",
    load_motion=_load_npz,
    njmax=800,
    nconmax=256,
)
