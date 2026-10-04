#!/usr/bin/env python3
"""FR3 URDF without Isaac Sim.

`isaac/build_surgical_cell.py:retarget_urdf()` gets its FR3 out of two Isaac Sim
extensions -- the URDF from `isaacsim.robot_motion.motion_generation`, the meshes
from the URDF importer's bundled `franka_description`. Neither exists on a
machine without Isaac Sim, and Isaac Sim does not install on this H100 host
(no RT cores; see VLA_HANDOFF.md 1.1).

So the FR3 comes from the manufacturer instead: `frankarobotics/franka_description`
on GitHub, expanded from xacro. That repo is where Isaac's copy came from in the
first place, and the numbers that matter are checked here rather than assumed --
`verify_against_kinematics()` compares every expanded joint origin against
`robots/fr3/kinematics.yaml`, which is the manufacturer's own table. If the two
ever disagree the build stops, because `laparo_control.BEST` was grid-searched
against these origins and a silent millimetre would invalidate the tuning.

The hand is stripped (`hand:=false`): both arms carry a motorised laparoscopic
unit bolted straight to the bare flange, exactly as the Isaac path does.

    from newton_cell.fr3_urdf import build_fr3_local
    path = build_fr3_local()          # -> out/fr3_local.urdf
"""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRANKA = ROOT / "assets" / "franka_description"
OUT = ROOT / "out" / "fr3_local.urdf"

REPO = "https://github.com/frankarobotics/franka_description.git"
BRANCH = "main"

# `laparo_ik.Chain` walks exactly these, in this order, and nothing else in the
# FR3 subtree is used by the controller. joint8 is the fixed link7 -> flange
# step; the tool URDF builder appends fr3_link8_to_tool and tool_flange_to_tip.
CHAIN_JOINTS = ("fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                "fr3_joint5", "fr3_joint6", "fr3_joint7", "fr3_joint8")

# The Franka hand subtree, removed here so the motorised unit can take the
# flange. Same lists as build_surgical_cell.HAND_LINKS / HAND_JOINTS; the xacro
# is expanded with hand:=false so they should already be absent, and their
# removal below is a belt-and-braces pass that also catches ee variants.
HAND_LINKS = ("fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger",
              "gripper_center")
HAND_JOINTS = ("fr3_hand_joint", "fr3_hand_tcp_joint", "fr3_finger_joint1",
               "fr3_finger_joint2", "gripper_center_joint")


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_franka_description(ref: str = BRANCH) -> Path:
    """Clone franka_description if it is not already beside the repo."""
    if (FRANKA / "robots" / "fr3" / "fr3.urdf.xacro").is_file():
        return FRANKA
    FRANKA.parent.mkdir(parents=True, exist_ok=True)
    log(f"  fetching {REPO} ({ref}) -> {FRANKA}")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref,
                    REPO, str(FRANKA)], check=True)
    return FRANKA


# --------------------------------------------------------------------------
# xacro
# --------------------------------------------------------------------------
def _resolve_package_paths(src_root: Path, work: Path) -> Path:
    """Copy the xacro tree and hard-substitute `$(find franka_description)`.

    xacro resolves `$(find pkg)` through the ROS package index, which is not
    installed here. Rewriting the literal is equivalent and keeps the expansion
    a pure file operation -- and it lands ABSOLUTE mesh paths in the output,
    which is what the Isaac path's retarget_urdf() produces too.
    """
    import shutil
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src_root, work, ignore=shutil.ignore_patterns(".git"))
    # `$(find pkg)` is xacro's own spelling and has to be gone before xacro
    # parses the includes. The other spelling, `package://pkg/`, is never
    # literal in the source -- utils.xacro builds it from a ${description_pkg}
    # property -- so it can only be caught on the expanded output, which
    # resolve_package_uris() does.
    n = 0
    for f in list(work.rglob("*.xacro")) + list(work.rglob("*.urdf")):
        text = f.read_text()
        if "$(find franka_description)" not in text:
            continue
        f.write_text(text.replace("$(find franka_description)", str(work)))
        n += 1
    log(f"  resolved $(find franka_description) in {n} file(s)")
    return work


def resolve_package_uris(xml: str, work: Path) -> str:
    """Turn `package://franka_description/...` into absolute paths.

    Nothing downstream speaks ROS: `laparo_ik` parses the URDF with
    ElementTree, and the Newton scene loads the meshes with trimesh. Both want
    a plain filesystem path, which is also what the Isaac path's
    retarget_urdf() ends up writing.
    """
    return xml.replace("package://franka_description/", f"{work}/")


def expand_xacro(robot_type: str = "fr3") -> str:
    src = ensure_franka_description()
    work = ROOT / "out" / "_franka_xacro"
    work.parent.mkdir(parents=True, exist_ok=True)
    _resolve_package_paths(src, work)

    xacro_file = work / "robots" / robot_type / f"{robot_type}.urdf.xacro"
    if not xacro_file.is_file():
        raise FileNotFoundError(xacro_file)

    import xacro
    doc = xacro.process_file(
        str(xacro_file),
        mappings={"hand": "false",          # the motorised unit replaces it
                  "robot_type": robot_type,
                  "arm_prefix": "",
                  "no_prefix": "false",     # keep the fr3_* joint names
                  "gazebo": "false",
                  "ros2_control": "false"},
    )
    return resolve_package_uris(doc.toprettyxml(indent="  "), work)


# --------------------------------------------------------------------------
# verification against the manufacturer's own kinematics table
# --------------------------------------------------------------------------
def _load_kinematics_yaml(robot_type: str = "fr3") -> dict:
    import yaml
    p = ensure_franka_description() / "robots" / robot_type / "kinematics.yaml"
    d = yaml.safe_load(p.read_text())
    out = {}
    for name, spec in d.items():
        k = spec["kinematic"]
        out[name] = (np.array([k["x"], k["y"], k["z"]], float),
                     np.array([k["roll"], k["pitch"], k["yaw"]], float))
    return out


def verify_against_kinematics(root: ET.Element, robot_type: str = "fr3",
                              tol: float = 1e-9) -> None:
    """Every chain joint origin must equal kinematics.yaml exactly.

    `laparo_control.BEST` is the winner of a 182-configuration grid search run
    against the Isaac copy of this chain. If this URDF's origins differ from the
    manufacturer's table the search result no longer describes this robot, so
    this is an assertion, not a warning.
    """
    table = _load_kinematics_yaml(robot_type)
    joints = {j.get("name"): j for j in root.findall("joint")}
    bad = []
    for i, jname in enumerate(CHAIN_JOINTS, start=1):
        key = f"joint{i}"
        if jname not in joints:
            bad.append(f"{jname}: missing from expanded URDF")
            continue
        if key not in table:
            bad.append(f"{jname}: no {key} in kinematics.yaml")
            continue
        o = joints[jname].find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0")).split()])
        rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0")).split()])
        exyz, erpy = table[key]
        dxyz = float(np.abs(xyz - exyz).max())
        drpy = float(np.abs(rpy - erpy).max())
        if dxyz > tol or drpy > tol:
            bad.append(f"{jname}: xyz off by {dxyz:.3e} m, rpy off by "
                       f"{drpy:.3e} rad (urdf {xyz}/{rpy} vs yaml {exyz}/{erpy})")
    if bad:
        raise SystemExit("FR3 kinematics do NOT match kinematics.yaml:\n  "
                         + "\n  ".join(bad))
    log(f"  verified {len(CHAIN_JOINTS)} joint origins against "
        f"{robot_type}/kinematics.yaml (exact)")


def verify_meshes(root: ET.Element) -> None:
    """Every mesh the URDF references must exist on disk."""
    missing = []
    n = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        fn = fn.replace("file://", "")
        if not fn:
            continue
        n += 1
        if not Path(fn).is_file():
            missing.append(fn)
    if missing:
        raise SystemExit("FR3 URDF references missing meshes:\n  "
                         + "\n  ".join(sorted(set(missing))[:10]))
    log(f"  verified {n} mesh reference(s) resolve on disk")


# --------------------------------------------------------------------------
def build_fr3_local(dest: Path = OUT, robot_type: str = "fr3") -> Path:
    """Expand, strip the hand, verify, write `out/fr3_local.urdf`."""
    log(f"FR3 URDF from franka_description ({robot_type})")
    xml = expand_xacro(robot_type)
    root = ET.fromstring(xml)

    dropped = []
    for el in list(root):
        name = el.get("name")
        if (el.tag == "link" and name in HAND_LINKS) or \
           (el.tag == "joint" and name in HAND_JOINTS):
            root.remove(el)
            dropped.append(name)
    if dropped:
        log(f"  removed hand subtree: {', '.join(dropped)}")

    verify_against_kinematics(root, robot_type)
    verify_meshes(root)

    links = [l.get("name") for l in root.findall("link")]
    if "fr3_link8" not in links:
        raise SystemExit(f"no fr3_link8 in expanded URDF; links={links}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(dest), xml_declaration=True, encoding="utf-8")
    log(f"  wrote {dest}  ({len(links)} links, "
        f"{len(root.findall('joint'))} joints)")
    return dest


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-type", default="fr3")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    build_fr3_local(Path(a.out), a.robot_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
