"""Check that a causal policy goal is invariant to a clip's unseen future suffix.

The test keeps the same converted Sharpa prefix in a second temporary file and truncates it at a
control-frame boundary.  Causal goals at the last retained frame must match the full clip, while the
released central-difference goal must change because its velocity uses a future frame.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--source_frames", type=int, default=25)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--goal_frames", type=int, default=3)
    args = parser.parse_args()
    if args.frame >= args.source_frames:
        raise ValueError("--frame must be inside the retained prefix")

    robot = get_robot("sharpa")
    kin = Kinematics(robot, "cpu")
    full = MotionLib([str(args.clip)], kin, "cpu", robot.load_motion, causal_goal=True)
    with np.load(args.clip, allow_pickle=False) as data, tempfile.TemporaryDirectory() as tmp:
        truncated = Path(tmp) / args.clip.name
        np.savez_compressed(
            truncated,
            root_pos=np.asarray(data["root_pos"])[: args.source_frames],
            root_quat_wxyz=np.asarray(data["root_quat_wxyz"])[: args.source_frames],
            joint_pos=np.asarray(data["joint_pos"])[: args.source_frames],
            fps=np.asarray(data["fps"]),
        )
        prefix = MotionLib([str(truncated)], kin, "cpu", robot.load_motion, causal_goal=True)
        f = torch.tensor([args.frame], dtype=torch.long)
        causal_error = (full.goal(f, args.goal_frames) - prefix.goal(f, args.goal_frames)).abs().max().item()

        full.set_causal_goal(False)
        prefix.set_causal_goal(False)
        central_error = (full.goal(f, args.goal_frames) - prefix.goal(f, args.goal_frames)).abs().max().item()

    if causal_error > 1e-6:
        raise AssertionError(f"causal goal changed after truncation: {causal_error:.3e}")
    if central_error <= 1e-6:
        raise AssertionError(f"central-difference control did not expose the future suffix: {central_error:.3e}")
    print(f"passed: causal_max_diff={causal_error:.3e}, central_max_diff={central_error:.3e}")


if __name__ == "__main__":
    main()
