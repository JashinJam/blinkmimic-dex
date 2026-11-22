"""Validate converted OakInk2 Sharpa clips against the source hand model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


TRACKED = {
    "sharpa_root": "R_forearm_rot_y_link",
    "right_thumb_DP": "right_thumb_DP",
    "right_index_DP": "right_index_DP",
    "right_middle_DP": "right_middle_DP",
    "right_ring_DP": "right_ring_DP",
    "right_pinky_DP": "right_pinky_DP",
}


def quat_to_mat_wxyz(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.concatenate((quat[..., 1:], quat[..., :1]), axis=-1)).as_matrix()


def validate_one(path: Path, source_xml: Path, target_xml: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        root_quat = np.asarray(data["root_quat_wxyz"], dtype=np.float64)
        wrist_q = np.asarray(data["wrist_joint_pos"], dtype=np.float64)
        joints = np.asarray(data["joint_pos"], dtype=np.float64)
        wrist_pos = np.asarray(data["human_wrist_pos"], dtype=np.float64)
        wrist_rot = np.asarray(data["human_wrist_rot"], dtype=np.float64)
    if wrist_q.shape != (len(root_pos), 6):
        raise ValueError(f"{path.name}: expected saved six-joint wrist trace, got {wrist_q.shape}")
    source = mujoco.MjModel.from_xml_path(str(source_xml))
    source_data = mujoco.MjData(source)
    target = mujoco.MjModel.from_xml_path(str(target_xml))
    target_data = mujoco.MjData(target)
    source_wrist = [source.jnt_qposadr[i] for i in range(source.njnt) if "forearm" in source.joint(i).name]
    target_fingers = [target.jnt_qposadr[i] for i in range(target.njnt) if target.joint(i).name.startswith("right_")]
    target_body_ids = {name: target.body(name).id for name in TRACKED}
    source_root_id = source.body("R_forearm_rot_y_link").id
    if len(source_wrist) != 6:
        raise ValueError(f"{path.name}: expected six source wrist joints, got {len(source_wrist)}")
    max_wrist_pos = 0.0
    max_wrist_rot = 0.0
    max_body_pos = 0.0
    max_body_rot = 0.0
    for frame in range(len(root_pos)):
        source_data.qpos[:] = 0.0
        source_data.qpos[source_wrist] = wrist_q[frame]
        mujoco.mj_forward(source, source_data)
        source_root_pos = source_data.xpos[source_root_id].copy()
        source_root_rot = source_data.xmat[source_root_id].reshape(3, 3).copy()
        max_wrist_pos = max(max_wrist_pos, float(np.linalg.norm(source_root_pos - root_pos[frame])))
        max_wrist_rot = max(max_wrist_rot, float(np.linalg.norm(source_root_rot - quat_to_mat_wxyz(root_quat[frame]))))
        target_data.qpos[:] = 0.0
        target_data.qpos[0:3] = root_pos[frame]
        target_data.qpos[3:7] = root_quat[frame]
        target_data.qpos[target_fingers] = joints[frame]
        mujoco.mj_forward(target, target_data)
        if not np.isfinite(target_data.xpos).all() or not np.isfinite(target_data.xmat).all():
            raise ValueError(f"{path.name}: non-finite target FK at frame {frame}")
        target_root_pos = target_data.xpos[target_body_ids["sharpa_root"]]
        target_root_rot = target_data.xmat[target_body_ids["sharpa_root"]].reshape(3, 3)
        max_body_pos = max(max_body_pos, float(np.linalg.norm(target_root_pos - root_pos[frame])))
        max_body_rot = max(max_body_rot, float(np.linalg.norm(target_root_rot - quat_to_mat_wxyz(root_quat[frame]))))
    return {
        "clip": path.stem,
        "frames": int(len(root_pos)),
        "finite": True,
        "generated_root_pose_error_max_m": max_body_pos,
        "generated_root_rotation_error_max": max_body_rot,
        "source_wrist_pose_error_max_m": max_wrist_pos,
        "source_wrist_rotation_error_max": max_wrist_rot,
        "human_wrist_finite": bool(np.isfinite(wrist_pos).all() and np.isfinite(wrist_rot).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-xml", type=Path, required=True)
    parser.add_argument("--target-xml", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = [validate_one(path, args.source_xml, args.target_xml) for path in sorted(args.data_dir.glob("*.npz"))]
    if not rows:
        raise SystemExit("no converted clips found")
    report = {"status": "passed", "clips": rows, "clip_count": len(rows)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
