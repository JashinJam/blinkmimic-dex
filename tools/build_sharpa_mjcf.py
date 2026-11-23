"""Build the BlinkMimic single-hand MJCF from the licensed Sharpa source asset.

The FlashDex source XML uses six serial wrist joints. BlinkMimic represents the tracked
root as a free joint, so this conversion folds the zero-angle wrist chain into the free
root pose and keeps only the 22 finger joints as actions.
"""

from __future__ import annotations

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _vec(text: str | None, size: int, default: tuple[float, ...]) -> list[float]:
    values = [float(x) for x in (text or "").split()]
    return values if values else list(default)


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_apply(q: list[float], v: list[float]) -> list[float]:
    _, x, y, z = q
    qv = [0.0, *v]
    return _quat_mul(_quat_mul(q, qv), [q[0], -x, -y, -z])[1:]


def _fmt(values: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print XML on Python versions before ``ElementTree.indent``."""
    pad = "\n" + "  " * level
    child_pad = "\n" + "  " * (level + 1)
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_pad
        for child in children:
            _indent(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = pad


def _chain_to(root: ET.Element, target_name: str) -> list[ET.Element]:
    chain = [root]
    current = root
    while current.get("name") != target_name:
        children = [body for body in current.findall("./body")]
        if len(children) != 1:
            names = [body.get("name") for body in children]
            raise ValueError(f"cannot follow wrist chain below {current.get('name')}: {names}")
        current = children[0]
        chain.append(current)
    return chain


def build(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("source MJCF has no worldbody")
    source_root = worldbody.find("./body")
    if source_root is None or source_root.get("name") != "R_forearm_pos_x_link":
        raise ValueError("unexpected Sharpa source root body")

    chain = _chain_to(source_root, "R_forearm_rot_y_link")
    target = chain[-1]

    # Compose the source wrist chain at its zero joint configuration. Each source body
    # position is expressed in its parent's frame and each source quaternion is wxyz.
    position = [0.0, 0.0, 0.0]
    orientation = [1.0, 0.0, 0.0, 0.0]
    for body in chain:
        local_pos = _vec(body.get("pos"), 3, (0.0, 0.0, 0.0))
        local_quat = _vec(body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        position = [a + b for a, b in zip(position, _quat_apply(orientation, local_pos))]
        orientation = _quat_mul(orientation, local_quat)
    norm = math.sqrt(sum(x * x for x in orientation))
    orientation = [x / norm for x in orientation]

    new_root = copy.deepcopy(target)
    new_root.set("name", "sharpa_root")
    new_root.set("pos", _fmt(position))
    new_root.set("quat", _fmt(orientation))
    # The source asset assumes a robot arm supports the hand.  This branch models
    # the wrist root as the externally commanded hand pose, so every hand body
    # receives full gravity compensation instead of the source 0.8 approximation.
    for body in new_root.iter("body"):
        body.set("gravcomp", "1.0")
    new_root.attrib.pop("class", None)
    new_root.attrib.pop("mocap", None)
    new_root.set("pos", _fmt(position))
    new_root.set("quat", _fmt(orientation))
    new_root.insert(0, ET.Element("freejoint", {"name": "floating_base_joint"}))

    wrist_joint = new_root.find("./joint")
    if wrist_joint is None or wrist_joint.get("name") != "R_forearm_rot_y_joint":
        raise ValueError("unexpected target wrist joint")
    new_root.remove(wrist_joint)
    # The copied target body was already expressed in its own frame. Its children
    # therefore retain their local transforms after the chain is folded into the root.
    new_root.set("pos", _fmt(position))
    new_root.set("quat", _fmt(orientation))

    worldbody.clear()
    worldbody.append(new_root)

    compiler = root.find("./compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "source/meshes/")

    # Actuator gains are supplied by the BlinkMimic RobotSpec. Keeping the source
    # position actuators here would create a second, XML-backed actuator group in
    # mjlab, so remove the entire source block.
    actuators = root.find("./actuator")
    if actuators is not None:
        root.remove(actuators)

    contacts = root.find("./contact")
    if contacts is not None:
        for exclude in contacts.findall("./exclude"):
            for key in ("body1", "body2"):
                if exclude.get(key) == "R_forearm_rot_y_link":
                    exclude.set(key, "sharpa_root")
                elif exclude.get(key, "").startswith("R_forearm_"):
                    contacts.remove(exclude)
                    break

    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # Validate the structural contract without requiring a GPU.
    check = ET.parse(output).getroot()
    free = check.findall(".//freejoint")
    joints = [x.get("name") for x in check.findall(".//joint") if x.get("name")]
    positions = [x.get("joint") for x in check.findall("./actuator/position")]
    if [x.get("name") for x in free] != ["floating_base_joint"]:
        raise ValueError("generated MJCF does not have exactly one expected freejoint")
    if any(name.startswith("R_forearm_") for name in joints + positions):
        raise ValueError("generated MJCF still contains a wrist action joint")
    if len(joints) != 22 or positions:
        raise ValueError(f"expected 22 hand joints and no XML actuators, got {len(joints)} and {len(positions)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("assets/sharpa/source/right.xml"))
    parser.add_argument("--output", type=Path, default=Path("assets/sharpa/sharpa_right.xml"))
    args = parser.parse_args()
    build(args.source, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
