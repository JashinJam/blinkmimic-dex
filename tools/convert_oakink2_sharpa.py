"""Convert OakInk2 right-hand clips to the BlinkMimic Sharpa motion format.

The input files are the local OakInk2-derived records produced by the FlashDex
retargeting pipeline.  The conversion keeps the source human and object fields
in each output file for provenance, while the four fields consumed by the
BlinkMimic motion loader are generated explicitly:

``root_pos`` and ``root_quat_wxyz``
    Pose of the folded Sharpa wrist body in world coordinates.
``joint_pos``
    The 22 Sharpa finger joints in ``sharpa_constants.JOINT_NAMES`` order.
``fps``
    Source frame rate.  BlinkMimic performs the 50 Hz resampling.

The hand mapping is frame-causal.  It uses only the current MANO frame and the
previous valid Sharpa finger solution for continuity.  Object fields are copied
for the later object-conditioned task and are not used to compute the hand
command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_JOINT_NAMES = tuple(
    sorted(f"{finger}_{part}" for finger in FINGERS for part in ("proximal", "intermediate", "distal", "tip"))
)
SHARPA_JOINT_NAMES = (
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
)


def _mat_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to MuJoCo's scalar-first quaternion order."""
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.concatenate((xyzw[..., 3:4], xyzw[..., :3]), axis=-1).astype(np.float32)


class CausalSharpaMapper:
    """Small self-contained copy of the causal MANO-to-Sharpa geometry solver."""

    def __init__(self, xml: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.finger_ids = np.asarray(
            [i for i, name in enumerate(self.joint_names) if name.startswith("right_")], dtype=np.int64
        )
        self.wrist_ids = np.asarray(
            [i for i, name in enumerate(self.joint_names) if "forearm" in name], dtype=np.int64
        )
        if len(self.finger_ids) != 22 or len(self.wrist_ids) != 6:
            raise ValueError(f"unexpected Sharpa joint layout: {len(self.wrist_ids)} wrist, {len(self.finger_ids)} finger")
        actual = tuple(self.joint_names[i] for i in self.finger_ids)
        if actual != SHARPA_JOINT_NAMES:
            raise ValueError(f"Sharpa joint order mismatch: {actual}")
        self.finger_qpos = self.model.jnt_qposadr[self.finger_ids].copy()
        self.wrist_qpos = self.model.jnt_qposadr[self.wrist_ids].copy()
        self.lower, self.upper = self.model.jnt_range[self.finger_ids].T.copy()
        self.ranges = self.upper - self.lower
        self.neutral = np.clip(np.zeros(22), self.lower, self.upper)
        self.palm = self.model.site("right_palm").id
        self.target_body = self.model.body("R_forearm_rot_y_link").id
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros_like(self._jp)
        self.points: list[tuple[str, int]] = []
        for finger in FINGERS:
            bones = ["thumb_MC", "thumb_PP", "thumb_DP"] if finger == "thumb" else [
                f"{finger}_MCP_VL", f"{finger}_MP", f"{finger}_DP"
            ]
            self.points.extend(("body", self.model.body(f"right_{bone}").id) for bone in bones)
            self.points.append(("site", self.model.site(f"track_hand_right_{finger}_tip").id))
        self.human_order = [
            MANO_JOINT_NAMES.index(f"{finger}_{part}")
            for finger in FINGERS
            for part in ("proximal", "intermediate", "distal", "tip")
        ]
        self.robot_neutral_points = self._kinematics(self.neutral, jacobian=False)[0].reshape(5, 4, 3)
        self.robot_lengths = np.linalg.norm(np.diff(self.robot_neutral_points, axis=1), axis=-1)
        self.reset()

    def reset(self) -> None:
        self.previous = self.neutral.copy()
        self.started = False
        self.human_lengths: np.ndarray | None = None
        self.fallback_count = 0
        self.previous_wrist = np.zeros(6, dtype=np.float64)

    def _kinematics(self, q: np.ndarray, jacobian: bool = True):
        self.data.qpos[self.finger_qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        origin = self.data.site_xpos[self.palm]
        rotation = self.data.site_xmat[self.palm].reshape(3, 3)
        points, jac = [], []
        for kind, index in self.points:
            if kind == "body":
                point = self.data.xpos[index]
                if jacobian:
                    mujoco.mj_jac(self.model, self.data, self._jp, self._jr, point, index)
            else:
                point = self.data.site_xpos[index]
                if jacobian:
                    mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, index)
            points.append((point - origin) @ rotation)
            if jacobian:
                jac.append(rotation.T @ self._jp[:, self.finger_ids])
        return np.stack(points), np.stack(jac) if jacobian else None

    def finger_step(self, local_joints: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool]]:
        human = np.asarray(local_joints, dtype=np.float64)[self.human_order].reshape(5, 4, 3)
        bones = np.diff(human, axis=1)
        lengths = np.linalg.norm(bones, axis=-1)
        if not np.isfinite(human).all() or np.any(lengths < 1e-5):
            self.fallback_count += 1
            return self.previous.copy(), {"valid": False, "tip_error_mean_m": float("nan")}
        if self.human_lengths is None:
            self.human_lengths = lengths.copy()
        scaled_bones = bones * (self.robot_lengths / self.human_lengths)[..., None]
        targets = self.robot_neutral_points[:, 0] + scaled_bones.sum(axis=1)
        desired_direction = bones / lengths[..., None]
        q = self.previous.copy()
        iterations = 5 if self.started else 20
        for _ in range(iterations):
            points, jac = self._kinematics(q)
            points = points.reshape(5, 4, 3)
            jac = jac.reshape(5, 4, 3, 22)
            tip_r = (points[:, -1] - targets) / (0.1 * np.sqrt(5))
            tip_j = jac[:, -1] / (0.1 * np.sqrt(5))
            vectors = np.diff(points, axis=1)
            vector_jac = np.diff(jac, axis=1)
            lengths_r = np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)
            unit = vectors / lengths_r
            direction_jac = np.einsum(
                "...ij,...jk->...ik",
                np.eye(3) - unit[..., :, None] * unit[..., None, :],
                vector_jac,
            ) / lengths_r[..., None]
            direction_weight = np.sqrt(0.1 / 15)
            residual = np.concatenate(
                [
                    tip_r.ravel(),
                    ((unit - desired_direction) * direction_weight).ravel(),
                    (q - self.previous) / self.ranges * np.sqrt(0.005 / 22),
                    (q - self.neutral) / self.ranges * np.sqrt(0.002 / 22),
                ]
            )
            jacobian = np.concatenate(
                [
                    tip_j.reshape(-1, 22),
                    (direction_jac * direction_weight).reshape(-1, 22),
                    np.diag(np.sqrt(0.005 / 22) / self.ranges),
                    np.diag(np.sqrt(0.002 / 22) / self.ranges),
                ]
            )
            delta = np.linalg.solve(jacobian.T @ jacobian + 0.001 * np.eye(22), -jacobian.T @ residual)
            q = np.clip(q + np.clip(delta, -0.1, 0.1), self.lower, self.upper)
        valid = bool(np.isfinite(q).all())
        if valid:
            self.previous = q
            self.started = True
        else:
            self.fallback_count += 1
        points = self._kinematics(self.previous, jacobian=False)[0].reshape(5, 4, 3)
        error = np.linalg.norm(points[:, -1] - targets, axis=-1)
        return self.previous.copy(), {"valid": valid, "tip_error_mean_m": float(error.mean())}

    def wrist_step(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the folded wrist-body world pose for one human wrist frame."""
        self.data.qpos[:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)
        root = self.model.jnt_bodyid[self.wrist_ids[0]]
        origin = self.data.site_xpos[self.palm].copy()
        A = self.data.xmat[root].reshape(3, 3).copy()
        D = A.T @ self.data.site_xmat[self.palm].reshape(3, 3)
        xyz = A.T @ (np.asarray(position, dtype=np.float64) - origin)
        angles = Rotation.from_matrix(A.T @ np.asarray(rotation) @ D.T).as_euler("ZXY")
        angles[2] *= -1.0
        angles += 2 * np.pi * np.round((self.previous_wrist[3:] - angles) / (2 * np.pi))
        wrist = np.concatenate((xyz, angles))
        self.previous_wrist = wrist
        self.data.qpos[self.wrist_qpos] = wrist
        mujoco.mj_kinematics(self.model, self.data)
        root_pos = self.data.xpos[self.target_body].copy()
        root_rot = self.data.xmat[self.target_body].reshape(3, 3).copy()
        return root_pos.astype(np.float32), _mat_to_wxyz(root_rot)


def _required(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], path: Path) -> None:
    missing = [key for key in keys if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")


def convert_one(path: Path, output: Path, mapper: CausalSharpaMapper) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        _required(data, ("fps", "mano_right_wrist_pos", "mano_right_wrist_rot", "mano_right_joints", "obj_right_pose", "obj_right_id"), path)
        fps = float(np.asarray(data["fps"]).item())
        wrist_pos = np.asarray(data["mano_right_wrist_pos"], dtype=np.float32)
        wrist_rot = np.asarray(data["mano_right_wrist_rot"], dtype=np.float32)
        mano_joints = np.asarray(data["mano_right_joints"], dtype=np.float32)
        obj_pose = np.asarray(data["obj_right_pose"], dtype=np.float32)
        obj_id = str(np.asarray(data["obj_right_id"]).item())
    T = len(wrist_pos)
    if fps <= 0 or wrist_pos.shape != (T, 3) or wrist_rot.shape != (T, 3, 3) or mano_joints.shape != (T, 20, 3):
        raise ValueError(f"{path}: invalid hand shapes or fps")
    if obj_pose.shape != (T, 4, 4) or not np.isfinite(np.concatenate((wrist_pos, wrist_rot.reshape(T, -1), mano_joints.reshape(T, -1), obj_pose.reshape(T, -1)), axis=1)).all():
        raise ValueError(f"{path}: invalid object pose or non-finite values")
    mapper.reset()
    roots_p, roots_q, wrist_q, joints_q, tip_errors = [], [], [], [], []
    for frame in range(T):
        local = np.einsum("ji,kj->ki", wrist_rot[frame], mano_joints[frame] - wrist_pos[frame])
        q, metrics = mapper.finger_step(local)
        if not metrics["valid"]:
            raise ValueError(f"{path}: invalid causal hand solution at frame {frame}")
        p, quat = mapper.wrist_step(wrist_pos[frame], wrist_rot[frame])
        roots_p.append(p)
        roots_q.append(quat)
        wrist_q.append(mapper.previous_wrist.astype(np.float32).copy())
        joints_q.append(q.astype(np.float32))
        tip_errors.append(float(metrics["tip_error_mean_m"]))
    obj_rot = obj_pose[:, :3, :3]
    out = {
        "root_pos": np.asarray(roots_p, dtype=np.float32),
        "root_quat_wxyz": np.asarray(roots_q, dtype=np.float32),
        "wrist_joint_pos": np.asarray(wrist_q, dtype=np.float32),
        "joint_pos": np.asarray(joints_q, dtype=np.float32),
        "object_pos": obj_pose[:, :3, 3].astype(np.float32),
        "object_quat_wxyz": _mat_to_wxyz(obj_rot),
        "human_wrist_pos": wrist_pos,
        "human_wrist_rot": wrist_rot,
        "human_mano_joints": mano_joints,
        "object_pose": obj_pose,
        "fps": np.asarray(fps, dtype=np.float32),
        "object_id": np.asarray(obj_id),
        "source_clip": np.asarray(path.stem),
        "joint_names": np.asarray(SHARPA_JOINT_NAMES),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(temp, **out)
    os.replace(temp, output)
    return {
        "clip": path.stem,
        "source": str(path),
        "output": str(output),
        "frames": T,
        "fps": fps,
        "object_id": obj_id,
        "causal_fallback_count": mapper.fallback_count,
        "causal_tip_error_mean_m": float(np.mean(tip_errors)),
        "causal_tip_error_p95_m": float(np.percentile(tip_errors, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/sharpa/source/right.xml"))
    parser.add_argument("--clips", nargs="*", default=None)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.clips:
        wanted = set(args.clips)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit("no input clips found")
    mapper = CausalSharpaMapper(args.xml)
    rows = []
    for path in paths:
        rows.append(convert_one(path, args.output_dir / f"{path.stem}.npz", mapper))
    train = [row["clip"] for row in rows if "5a5d66e7a358b59c9fee" in str(row["clip"])]
    holdout = [row["clip"] for row in rows if "ce2f09a1ccbe225a69aa" in str(row["clip"])]
    if len(train) != 3 or len(holdout) != 1:
        raise ValueError(f"expected 3 train clips from one sequence and 1 holdout clip from the other, got {len(train)} and {len(holdout)}")
    manifest = {
        "format_version": 1,
        "robot": "sharpa_right",
        "source": "OakInk2-derived right-hand clips already present on the server",
        "object_id": "S20005",
        "train": sorted(train),
        "holdout": sorted(holdout),
        "clips": rows,
        "actor_command_fields": ["human_wrist_pos", "human_wrist_rot", "human_mano_joints"],
        "privileged_fields": ["object_pos", "object_quat_wxyz", "object_pose"],
        "license_reference": "OakInk2 preview data card: CC BY-SA 4.0, https://oakink.net/v2/",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"clips": len(rows), "train": len(train), "holdout": len(holdout), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
