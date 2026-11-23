"""Record dynamic-object stage diagnostics for a privileged reference-action rollout.

The rollout uses the next-frame joint reference as a short diagnostic upper bound.  The
object state and contact array are read only for evaluation and never enter the action,
observation, or reward path.  The output separates raw traces from configurable stage
candidates:

* approach ends at the first control step with an object-involving contact;
* a hold candidate starts after ``hold_window`` consecutive contact steps with object
  speed no greater than ``hold_speed``;
* a lift candidate starts after a hold candidate once the object has risen by at least
  ``lift_delta`` from its first-contact height.

These thresholds provide an auditable first diagnostic.  They are not a task benchmark
and do not turn a contact into a grasp-success label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot


def object_contact_counts(env: TrackingSim, object_geom_ids: list[int]) -> np.ndarray:
    """Return the number of object-involving contact pairs for each simulation world."""
    ncon = int(env.sim.data.nacon.detach().cpu().item())
    if ncon <= 0:
        return np.zeros(env.N, dtype=np.int64)
    geom = env.sim.data.contact.geom.detach().cpu().numpy()[:ncon]
    world = env.sim.data.contact.worldid.detach().cpu().numpy()[:ncon]
    mask = np.isin(geom[:, 0], object_geom_ids) | np.isin(geom[:, 1], object_geom_ids)
    return np.bincount(world[mask], minlength=env.N)


def _first_or_none(value: int) -> int | None:
    return None if value < 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="sharpa")
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--object_mjcf", required=True)
    parser.add_argument("--output", type=Path, required=True, help="summary JSON path")
    parser.add_argument("--hold_window", type=int, default=10)
    parser.add_argument("--hold_speed", type=float, default=0.25, help="m/s")
    parser.add_argument("--lift_delta", type=float, default=0.02, help="m above first-contact height")
    args = parser.parse_args()
    if args.hold_window < 1 or args.hold_speed <= 0 or args.lift_delta <= 0:
        raise ValueError("hold_window must be positive and hold_speed/lift_delta must be greater than zero")

    dev = "cuda:0"
    robot = get_robot(args.robot)
    kin = Kinematics(robot, dev)
    motion = MotionLib.from_dir(args.motion_dir, robot, kin, dev, causal_goal=True)
    if motion.object_p is None:
        raise ValueError("dynamic stage evaluation requires object_pos/object_quat_wxyz in the motion files")
    mdp = MDP(
        kin,
        motion,
        torch.tensor(robot.default_joint_pos_list(), device=dev),
        XiLayout(robot, dev).dim,
        robot.sensing,
        include_root_position_error=robot.include_root_position_error,
    )
    clip_of = torch.arange(len(motion.names), device=dev)
    env = TrackingSim(
        make_sim(robot),
        len(clip_of),
        robot,
        motion,
        mdp,
        dev,
        eval_mode=True,
        randomize=False,
        object_mjcf=args.object_mjcf,
        dynamic_object=True,
    )
    model = env.sim.mj_model
    object_ids = [
        i for i in range(model.ngeom)
        if (model.geom(i).name or "").startswith("object/object_collision_")
    ]
    if not object_ids:
        raise ValueError("dynamic object has no named collision geoms")

    env.reset(torch.arange(env.N, device=dev), motion.clip_start[clip_of].clone())
    env.refresh()
    clip_start = motion.clip_start[clip_of].clone()
    clip_end = motion.clip_end[clip_of]
    first_contact = [-1] * env.N
    first_contact_z = [None] * env.N
    contact_run = [0] * env.N
    stable_run = [0] * env.N
    max_contact_run = [0] * env.N
    max_stable_run = [0] * env.N
    hold_start = [-1] * env.N
    lift_start = [-1] * env.N
    rows: list[dict] = []
    total_return = torch.zeros(env.N, device=dev)
    total_steps = torch.zeros(env.N, device=dev)
    max_lift = np.full(env.N, np.nan, dtype=np.float64)
    max_speed = np.zeros(env.N, dtype=np.float64)
    min_body_object = np.full(env.N, np.inf, dtype=np.float64)
    contact_frames = np.zeros(env.N, dtype=np.int64)
    stable_frames = np.zeros(env.N, dtype=np.int64)

    with torch.no_grad():
        while not bool(env.done_eval.all()):
            alive = ~env.done_eval.clone()
            f = env.f.clone()
            f1 = torch.minimum(f + 1, clip_end)
            action = (motion.q[f1] - env.default_q) / env.alpha
            out = env.step(action)
            env.refresh()
            object_pos = env.object.data.root_link_pos_w - env.scene.env_origins
            object_vel = env.object.data.root_link_vel_w
            body_pos = kin.forward(*env.state())[0]
            min_dist = (body_pos[:, 1:, :] - object_pos.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
            contacts = object_contact_counts(env, object_ids)
            speed = object_vel[:, :3].norm(dim=-1).detach().cpu().numpy()
            z = object_pos[:, 2].detach().cpu().numpy()
            ref_z = motion.object_p[f1, 2].detach().cpu().numpy()
            min_dist_np = min_dist.detach().cpu().numpy()
            f_np = f.detach().cpu().numpy()
            f1_np = f1.detach().cpu().numpy()
            alive_np = alive.detach().cpu().numpy()

            for i in range(env.N):
                if not alive_np[i]:
                    continue
                step = int(f_np[i] - clip_start[i].item())
                has_contact = int(contacts[i]) > 0
                contact_frames[i] += int(has_contact)
                max_speed[i] = max(max_speed[i], float(speed[i]))
                min_body_object[i] = min(min_body_object[i], float(min_dist_np[i]))
                if has_contact:
                    contact_run[i] += 1
                    if first_contact[i] < 0:
                        first_contact[i] = step
                        first_contact_z[i] = float(z[i])
                    max_contact_run[i] = max(max_contact_run[i], contact_run[i])
                else:
                    contact_run[i] = 0
                stable = has_contact and float(speed[i]) <= args.hold_speed
                if stable:
                    stable_run[i] += 1
                    stable_frames[i] += 1
                    max_stable_run[i] = max(max_stable_run[i], stable_run[i])
                    if hold_start[i] < 0 and stable_run[i] >= args.hold_window:
                        hold_start[i] = step - args.hold_window + 1
                else:
                    stable_run[i] = 0
                if first_contact_z[i] is not None:
                    delta = float(z[i] - first_contact_z[i])
                    max_lift[i] = delta if np.isnan(max_lift[i]) else max(max_lift[i], delta)
                    if hold_start[i] >= 0 and lift_start[i] < 0 and delta >= args.lift_delta:
                        lift_start[i] = step
                rows.append({
                    "clip": motion.names[i],
                    "step": step,
                    "frame": int(f1_np[i]),
                    "reference_frame": int(f1_np[i]),
                    "contact_count": int(contacts[i]),
                    "contact": int(has_contact),
                    "stable_contact": int(stable),
                    "object_x": float(object_pos[i, 0].item()),
                    "object_y": float(object_pos[i, 1].item()),
                    "object_z": float(z[i]),
                    "reference_object_z": float(ref_z[i]),
                    "object_speed_m_s": float(speed[i]),
                    "min_tracked_body_distance_m": float(min_dist_np[i]),
                    "normalized_step_reward": float(out["r"][i].item() / MAX_STEP_REWARD),
                })
            total_return += torch.where(alive, out["r"], 0.0)
            total_steps += alive
            env.f = torch.minimum(env.f, clip_end - 1)

    summaries = []
    for i, name in enumerate(motion.names):
        steps = max(int(total_steps[i].item()), 1)
        length = max(int(motion.clip_len[i].item()) - 1, 1)
        summaries.append({
            "clip": name,
            "clip_frames": int(motion.clip_len[i].item()),
            "rollout_steps": steps,
            "norm_return": float(total_return[i].item() / (length * MAX_STEP_REWARD)),
            "failed": int(env.failed_eval[i].item()),
            "approach_frames_before_first_contact": first_contact[i] if first_contact[i] >= 0 else steps,
            "first_contact_step": _first_or_none(first_contact[i]),
            "contact_frames": int(contact_frames[i]),
            "max_contact_run": int(max_contact_run[i]),
            "stable_contact_frames": int(stable_frames[i]),
            "max_stable_contact_run": int(max_stable_run[i]),
            "hold_candidate_step": _first_or_none(hold_start[i]),
            "lift_candidate_step": _first_or_none(lift_start[i]),
            "max_object_lift_from_first_contact_m": None if np.isnan(max_lift[i]) else float(max_lift[i]),
            "max_object_speed_m_s": float(max_speed[i]),
            "min_tracked_body_distance_m": None if np.isinf(min_body_object[i]) else float(min_body_object[i]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeseries_path = args.output.with_suffix(".csv")
    if rows:
        with timeseries_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "robot": args.robot,
        "motion_dir": args.motion_dir,
        "object_mjcf": args.object_mjcf,
        "dynamic_object": True,
        "contact_geom_ids": object_ids,
        "thresholds": {
            "hold_window_steps": args.hold_window,
            "hold_speed_m_s": args.hold_speed,
            "lift_delta_m": args.lift_delta,
        },
        "definitions": {
            "approach": "steps before the first detected object-involving contact",
            "first_contact": "first control step with at least one object-involving contact pair",
            "hold_candidate": "first window of hold_window consecutive contact steps with object speed <= hold_speed",
            "lift_candidate": "first step after hold_candidate with object height >= lift_delta above first-contact height",
        },
        "timeseries_csv": str(timeseries_path),
        "clips": summaries,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
