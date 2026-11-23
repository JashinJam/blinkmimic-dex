"""Check FK vs simulator, state round-trip, and reward on reference playback."""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--causal_goal", action="store_true", help="use the current-frame causal goal path")
parser.add_argument("--object_mjcf", type=str, default=None, help="reference or dynamic object MJCF")
parser.add_argument("--dynamic_object", action="store_true",
                    help="treat object_mjcf as a free-root collidable object and leave it to physics after reset")
parser.add_argument("--object_rel_reward", type=float, default=0.0,
                    help="optional dense hand-object relative-position reward weight")
parser.add_argument("--object_rel_std", type=float, default=0.05,
                    help="standard deviation used by the hand-object relative-position reward")
args = parser.parse_args()

if args.dynamic_object and args.object_rel_reward:
    raise ValueError("--object_rel_reward currently compares against the recorded object reference; leave it at 0 for dynamic smoke")

import torch
import numpy as np

from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP, RewardCfg
from blinkmimic.motion import MotionLib
from blinkmimic.robot import get_robot
from blinkmimic.rotations import quat_to_mat, rot_angle_sq


def _object_contact_counts(env, object_geom_ids: list[int]) -> np.ndarray:
    """Read per-world contacts involving the dynamic object's collision geoms."""
    ncon = int(env.sim.data.nacon.detach().cpu().item())
    if ncon <= 0:
        return np.zeros(env.num_envs, dtype=np.int64)
    geom = env.sim.data.contact.geom.detach().cpu().numpy()[:ncon]
    world = env.sim.data.contact.worldid.detach().cpu().numpy()[:ncon]
    mask = np.isin(geom[:, 0], object_geom_ids) | np.isin(geom[:, 1], object_geom_ids)
    return np.bincount(world[mask], minlength=env.num_envs)

dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
t0 = time.time()
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev, causal_goal=args.causal_goal)
print(f"motion: {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
reward_cfg = RewardCfg(w_object_rel=args.object_rel_reward, object_rel_std=args.object_rel_std)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing,
          cfg=reward_cfg, include_root_position_error=robot.include_root_position_error)
env = TrackingSim(make_sim(robot), args.num_envs, robot, motion, mdp, dev, eval_mode=True,
                  object_mjcf=args.object_mjcf, dynamic_object=args.dynamic_object)
frames = motion.clip_start[torch.arange(args.num_envs, device=dev) % len(motion.names)] + 100
if args.object_mjcf:
    if motion.object_p is None:
        raise ValueError("--object_mjcf requires motion files with object_pos/object_quat_wxyz")
    print(f"object: mocap={env.object.is_mocap} fixed_base={env.object.is_fixed_base} dynamic={args.dynamic_object} path={args.object_mjcf}")
    if args.dynamic_object:
        model = env.sim.mj_model
        robot_ids = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("robot/collision_")]
        object_ids = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("object/object_collision_")]
        print("dynamic collision geoms:", "robot", len(robot_ids), "object", len(object_ids),
              "robot contype", sorted({int(model.geom_contype[i]) for i in robot_ids}),
              "robot conaffinity", sorted({int(model.geom_conaffinity[i]) for i in robot_ids}),
              "object contype", sorted({int(model.geom_contype[i]) for i in object_ids}),
              "object conaffinity", sorted({int(model.geom_conaffinity[i]) for i in object_ids}))
env.reset(torch.arange(args.num_envs, device=dev), frames)
env.refresh()
if args.object_mjcf:
    if env.object.is_mocap:
        mocap_id = env.object.indexing.mocap_id
        object_pos = env.sim.data.mocap_pos[:, mocap_id] - env.scene.env_origins
    else:
        object_pos = env.object.data.root_link_pos_w - env.scene.env_origins
    object_pos_error = (object_pos - motion.object_p[frames]).abs().max().item()
    print(f"object reset position max error: {object_pos_error:.3e} m")
S = env.state()
ref = motion.ref_state(frames)
names = ["p", "R", "v", "w", "q", "qd"]
for n, a, b in zip(names, S, ref):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")
# FK vs sim body frames
d = env.robot.data
ids = [env.robot.body_names.index(b) for b in robot.tracked_bodies]
sp = d.body_link_pos_w[:, ids]
sR = quat_to_mat(d.body_link_quat_w[:, ids])  # wxyz
bp, bR, bv, bw = kin.forward(*S)
print("FK pos err", (bp - sp).norm(dim=-1).max().item())
print("FK rot err", rot_angle_sq(bR, sR).sqrt().max().item())
sv = d.body_link_vel_w[:, ids]
print("FK linvel err", (bv - sv[..., :3]).norm(dim=-1).max().item(), " angvel err", (bw - sv[..., 3:]).norm(dim=-1).max().item())
print("ref body vs FK(sim) pos err", (motion.body_p[frames] - bp).norm(dim=-1).max().item())
# open-loop reference playback via joint targets
rs = []
for t in range(150):
    f1 = env.f + 1
    a = (motion.q[f1] - env.default_q) / env.alpha
    out = env.step(a)
    rs.append(out["r"].mean().item() / 0.1)
    if t % 25 == 0:
        line = (t, "norm reward", rs[-1], "term frac", env.failed_eval.float().mean().item(),
                "e_body_pos", out["terms"]["e_body_pos"].mean().item())
        if "object_rel" in out["terms"]:
            line += ("e_object_rel", out["terms"]["object_rel"].mean().item())
        if args.dynamic_object:
            env.refresh()
            object_pos = env.object.data.root_link_pos_w - env.scene.env_origins
            contact_counts = _object_contact_counts(env, object_ids)
            line += ("object_z", object_pos[:, 2].mean().item(),
                     "object_contacts", float(contact_counts.mean()),
                     "worlds_with_object_contact", int(np.count_nonzero(contact_counts)))
        print(*line)
