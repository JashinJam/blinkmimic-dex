"""Evaluate privileged next-frame reference actions as a control upper bound."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="sharpa")
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--object_mjcf", default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be positive")

    dev = "cuda:0"
    robot = get_robot(args.robot)
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(args.motion_dir, robot, kin, dev, causal_goal=True)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim,
              robot.sensing, include_root_position_error=robot.include_root_position_error)
    clip_of = torch.arange(len(motion.names) * args.reps, device=dev) % len(motion.names)
    env = TrackingSim(make_sim(robot), len(clip_of), robot, motion, mdp, dev, eval_mode=True,
                      reset_noise=False, randomize=False, object_mjcf=args.object_mjcf)
    starts = motion.clip_start[clip_of].clone()
    clip_end = motion.clip_end[clip_of]
    clip_len = motion.clip_len[clip_of]
    env.reset(torch.arange(len(clip_of), device=dev), starts)

    ret = torch.zeros(len(clip_of), device=dev)
    counts = torch.zeros(len(clip_of), device=dev)
    body_sum = torch.zeros(len(clip_of), device=dev)
    body_max = torch.zeros(len(clip_of), device=dev)
    root_sum = torch.zeros(len(clip_of), device=dev)
    object_sum = torch.zeros(len(clip_of), device=dev)
    object_max = torch.zeros(len(clip_of), device=dev)
    while not bool(env.done_eval.all()):
        alive = ~env.done_eval.clone()
        f1 = env.f + 1
        action = (motion.q[f1] - env.default_q) / env.alpha
        out = env.step(action)
        ret += torch.where(alive, out["r"], 0.0)
        counts += alive
        body_sum += torch.where(alive, out["terms"]["e_body_pos"], 0.0)
        body_max = torch.where(alive, torch.maximum(body_max, out["terms"]["e_body_pos"]), body_max)
        root_sum += torch.where(alive, out["terms"]["e_root_pos"], 0.0)
        if motion.object_p is not None:
            state = env.state()
            body_p = kin.forward(*state)[0]
            rel = body_p[:, 1:, :] - motion.object_p[f1].unsqueeze(-2)
            rel_ref = motion.body_p[f1][:, 1:, :] - motion.object_p[f1].unsqueeze(-2)
            e_object = ((rel - rel_ref) ** 2).sum(-1).mean(-1)
            object_sum += torch.where(alive, e_object, 0.0)
            object_max = torch.where(alive, torch.maximum(object_max, e_object), object_max)
        env.f.copy_(torch.minimum(env.f, clip_end - 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(motion.names):
        for rep in range(args.reps):
            j = rep * len(motion.names) + i
            denom = max(float(clip_len[j].item() - 1) * MAX_STEP_REWARD, 1e-8)
            n = max(float(counts[j].item()), 1.0)
            rows.append({
                "clip": name,
                "rep": rep,
                "norm_return": float(ret[j].item() / denom),
                "failed": int(env.failed_eval[j].item()),
                "steps": int(counts[j].item()),
                "mean_body_pos_sq": float(body_sum[j].item() / n),
                "max_body_pos_sq": float(body_max[j].item()),
                "mean_root_pos_sq": float(root_sum[j].item() / n),
                "mean_object_rel_sq": float(object_sum[j].item() / n),
                "max_object_rel_sq": float(object_max[j].item()),
            })
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"oracle clips={len(motion.names)} reps={args.reps} output={args.output}")
    print(f"oracle norm_return_mean={sum(r['norm_return'] for r in rows) / len(rows):.6f} "
          f"failed={sum(r['failed'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
