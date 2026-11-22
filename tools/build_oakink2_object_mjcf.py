"""Build a fixed-base mocap MJCF for one OakInk2 object.

The generated scene object follows the recorded object pose and has collision disabled.  This is a
reference/visualization object for the first tracking stage.  A dynamic object and contact semantics are
introduced only after the hand-only tracking loop is stable.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _float(attr: ET.Element, name: str, default: float = 0.0) -> float:
    return float(attr.attrib.get(name, default))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object_id", required=True)
    args = parser.parse_args()

    root = ET.parse(args.urdf).getroot()
    link = root.find("link")
    if link is None:
        raise ValueError(f"{args.urdf}: URDF has no link")
    inertial = link.find("inertial")
    if inertial is None or inertial.find("mass") is None or inertial.find("inertia") is None:
        raise ValueError(f"{args.urdf}: missing inertial mass or inertia")
    origin = inertial.find("origin")
    xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    mass = _float(inertial.find("mass"), "value")
    inertia = inertial.find("inertia")
    ixx, ixy, ixz = (_float(inertia, n) for n in ("ixx", "ixy", "ixz"))
    iyy, iyz, izz = (_float(inertia, n) for n in ("iyy", "iyz", "izz"))
    mesh_dir = args.urdf.parent.parent
    if not (mesh_dir / "visual.obj").exists() or not (mesh_dir / "convex/0.obj").exists() or not (mesh_dir / "convex/1.obj").exists():
        raise FileNotFoundError(f"{args.urdf}: expected visual.obj and convex meshes under {mesh_dir}")

    xml = f'''<?xml version="1.0"?>
<mujoco model="object_{args.object_id}">
  <compiler angle="radian" meshdir="../" />
  <default>
    <geom contype="0" conaffinity="0" condim="1" />
  </default>
  <asset>
    <mesh name="object_visual" file="visual.obj" />
    <mesh name="object_collision_0" file="convex/0.obj" />
    <mesh name="object_collision_1" file="convex/1.obj" />
  </asset>
  <worldbody>
    <body name="object_{args.object_id}" pos="0 0 0" quat="1 0 0 0">
      <inertial pos="{xyz}" mass="{mass:.9g}" fullinertia="{ixx:.9g} {iyy:.9g} {izz:.9g} {ixy:.9g} {ixz:.9g} {iyz:.9g}" />
      <geom name="object_visual" type="mesh" mesh="object_visual" contype="0" conaffinity="0" density="0" rgba="0.75 0.70 0.60 1" />
      <geom name="object_collision_0" type="mesh" mesh="object_collision_0" contype="0" conaffinity="0" density="0" />
      <geom name="object_collision_1" type="mesh" mesh="object_collision_1" contype="0" conaffinity="0" density="0" />
    </body>
  </worldbody>
</mujoco>
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(xml)
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "object_id": args.object_id,
        "source_urdf": str(args.urdf),
        "mesh_dir": str(mesh_dir),
        "mass_kg": mass,
        "inertia_urdf": {"ixx": ixx, "ixy": ixy, "ixz": ixz, "iyy": iyy, "iyz": iyz, "izz": izz},
        "mode": "fixed_base_mocap_reference",
        "collision_enabled": False,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} and {sidecar}")


if __name__ == "__main__":
    main()
