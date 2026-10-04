#!/usr/bin/env python3
"""FR3 + motorised laparoscopic instrument, as one URDF per arm.

A port of `isaac/build_surgical_cell.py:build_arm_tool_urdf()` with the Isaac
dependency removed. That function is already pure `xml.etree` -- it only
touched Isaac through `retarget_urdf()`, which pulled the FR3 out of an Isaac
extension. `newton_cell/fr3_urdf.py` supplies the FR3 instead; everything below
follows the original structure and comments, and reads its numbers from
`cell_spec.constants()` so the two engines share one set.

The output is the file the handoff calls "the single URDF both engines share":

    out/fr3_left_grasper.urdf      FR3 (hand removed) + motorised grasper
    out/fr3_right_cutter.urdf      FR3 (hand removed) + motorised cutter

`laparo_ik.Chain` parses these, so this is what makes the numpy IK and
`laparo_control.BEST` run identically on both machines.

    .venv-newton/bin/python newton_cell/tool_urdf.py
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cell_spec import (ROOT, constants, mesh_material_name, staged_mesh_path,
                       tools)
from fr3_urdf import build_fr3_local


def log(msg: str) -> None:
    print(msg, flush=True)


def build_arm_tool_urdf(T, fr3_urdf: Path) -> Path:
    """FR3 (hand removed) + one of the motorised tools on the bare flange.

    Both arms are built by this one function from a TOOLS spec; the left
    grasper and the right cutter are the same instrument family and differ
    only in the numbers (T.vis_rpy is turned 180 deg about Z on the cutter,
    its tip is 21.8 mm shorter, and its blades open 50 deg instead of 74).
    Keeping one builder is deliberate -- two copies of this would drift.

    Neither arm has a Franka hand: the unit bolts straight onto the flange
    (fr3_link8). So the hand/finger subtree is stripped from the FR3 and one
    rigid `tool_flange` link -- carrying the CAD meshes -- is added coincident
    with fr3_link8, plus a massless `tool_tip` frame at the measured SE(3)
    T_flange->tip.

    The endtip is articulated: the two blades/jaws are their own links on
    revolute joints about the clevis pin, 0 = closed as exported, +half the
    full opening each. Everything else is one rigid body, position-driven.
    """
    C = constants()
    tree = ET.parse(fr3_urdf)
    root = tree.getroot()
    for el in list(root):
        name = el.get("name")
        if (el.tag == "link" and name in C["HAND_LINKS"]) or \
           (el.tag == "joint" and name in C["HAND_JOINTS"]):
            root.remove(el)

    if not T.mesh_dir.is_dir():
        raise FileNotFoundError(f"{T.name} meshes not found: {T.mesh_dir}")
    meshes = sorted(p for p in T.mesh_dir.glob("*.stl")
                    if p.name not in set(T.jaw_meshes.values()))
    if not meshes:
        raise FileNotFoundError(f"no STL meshes under {T.mesh_dir}")
    have = {p.name for p in meshes}
    lost = [m for m in C["LAPARO_EXTRA_MESHES"] if m not in have]
    if lost:
        raise FileNotFoundError(f"{T.name}: casing meshes missing: {lost}")

    # per-part materials matching the photo of the real unit: black housing +
    # shaft, silver metal endtip, dark fasteners.
    for mname, rgb in C["TOOL_MATERIALS"].items():
        m = ET.SubElement(root, "material", {"name": mname})
        ET.SubElement(m, "color", {"rgba": "%g %g %g 1" % rgb})

    link = ET.SubElement(root, "link", {"name": "tool_flange"})
    inr = ET.SubElement(link, "inertial")
    ET.SubElement(inr, "origin", {"xyz": "%g %g %g" % T.com, "rpy": "0 0 0"})
    ET.SubElement(inr, "mass", {"value": "%g" % T.mass})
    ET.SubElement(inr, "inertia", T.inertia)
    vis_xyz = "%g %g %g" % T.vis_xyz
    vis_rpy = "%g %g %g" % T.vis_rpy
    scale = "%g %g %g" % ((C["LAPARO_MESH_SCALE"],) * 3)
    n_col = 0
    for mesh in meshes:
        fn = str(staged_mesh_path(mesh))
        mname = mesh_material_name(mesh.stem)
        for tag in ("visual", "collision"):
            if tag == "collision" and mesh.name not in C["LAPARO_COLLIDE_MESHES"]:
                continue
            node = ET.SubElement(link, tag)
            ET.SubElement(node, "origin", {"xyz": vis_xyz, "rpy": vis_rpy})
            geo = ET.SubElement(node, "geometry")
            ET.SubElement(geo, "mesh", {"filename": fn, "scale": scale})
            if tag == "visual":
                ET.SubElement(node, "material", {"name": mname})
            else:
                n_col += 1

    # the two blades: revolute about the clevis pin, 0 = closed (as exported),
    # +half the full opening each. The blade link frame is the flange frame
    # TRANSLATED to the pivot (no rotation), so the same one visual origin
    # places the mesh after subtracting the pivot, and the joint axis is the
    # pin axis expressed directly in the flange frame. POSITIVE q opens BOTH,
    # with the opposite signs carried by the two axes.
    half_open = math.radians(T.open_total_deg) / 2.0
    jaw_off = tuple(v - p for v, p in zip(T.vis_xyz, T.pivot_xyz))
    for side, sign in (("left", 1.0), ("right", -1.0)):
        jl = ET.SubElement(root, "link", {"name": f"{T.link_prefix}_{side}"})
        inr = ET.SubElement(jl, "inertial")
        ET.SubElement(inr, "origin", {"xyz": "%g %g %g" % C["JAW_COM"],
                                      "rpy": "0 0 0"})
        ET.SubElement(inr, "mass", {"value": "%g" % C["JAW_MASS"]})
        ET.SubElement(inr, "inertia", C["JAW_INERTIA"])
        fn = str(T.mesh_dir / T.jaw_meshes[side])
        for tag in ("visual", "collision"):
            node = ET.SubElement(jl, tag)
            ET.SubElement(node, "origin", {"xyz": "%g %g %g" % jaw_off,
                                           "rpy": vis_rpy})
            geo = ET.SubElement(node, "geometry")
            ET.SubElement(geo, "mesh", {"filename": fn, "scale": scale})
            if tag == "visual":
                ET.SubElement(node, "material", {"name": "stainless_steel"})
        j = ET.SubElement(root, "joint", {"name": f"{T.joint_prefix}_{side}",
                                          "type": "revolute"})
        ET.SubElement(j, "parent", {"link": "tool_flange"})
        ET.SubElement(j, "child", {"link": f"{T.link_prefix}_{side}"})
        ET.SubElement(j, "origin", {"xyz": "%g %g %g" % T.pivot_xyz,
                                    "rpy": "0 0 0"})
        ET.SubElement(j, "axis", {"xyz": "%g %g %g" %
                                  tuple(sign * a for a in T.pin_axis)})
        ET.SubElement(j, "limit", {"lower": "0", "upper": "%.6f" % half_open,
                                   "effort": "20", "velocity": "20"})
        ET.SubElement(j, "dynamics", {"damping": "0.01", "friction": "0"})

    # fixed joint: the tool_flange frame IS the bare flange (identity transform)
    j = ET.SubElement(root, "joint", {"name": "fr3_link8_to_tool", "type": "fixed"})
    ET.SubElement(j, "parent", {"link": "fr3_link8"})
    ET.SubElement(j, "child", {"link": "tool_flange"})
    ET.SubElement(j, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    # massless tip frame at the measured SE(3) T_flange->tip
    ET.SubElement(root, "link", {"name": "tool_tip"})
    jt = ET.SubElement(root, "joint",
                       {"name": "tool_flange_to_tip", "type": "fixed"})
    ET.SubElement(jt, "parent", {"link": "tool_flange"})
    ET.SubElement(jt, "child", {"link": "tool_tip"})
    ET.SubElement(jt, "origin", {"xyz": "%g %g %g" % T.tip_xyz,
                                 "rpy": "%g %g %g" % T.tip_rpy})

    out = ROOT / T.urdf
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    log(f"  built {T.name} urdf -> {shown}  "
        f"({len(meshes)} static tool meshes, {n_col} collider(s) + 2 "
        f"articulated {T.link_prefix}s (0..{T.open_total_deg/2:.2f} deg each, "
        f"{T.open_total_deg:.0f} deg full opening), hand removed, "
        f"tip {T.tip_xyz[2]*1000:.1f} mm out along flange +Z)")
    return out


# --------------------------------------------------------------------------
# verification: the tip frame the handoff documents, checked from the URDF
# --------------------------------------------------------------------------
def verify_tip_frame(urdf: Path, arm: str) -> None:
    """Re-derive VLA_HANDOFF.md section 2's tip-frame table from this URDF.

    The handoff states, and the teleop and every policy rely on:

        tip +x = (-0.707,  0.707,  0)    jaw closing direction
        tip +y = ( 0.707,  0.707,  0)    clevis pin axis
        tip +z = ( 0,      0,     -1)    shaft outward  ->  insertion = -z

    all expressed in the FLANGE frame, on BOTH arms. `dz < 0` inserting the
    instrument is the single most load-bearing sign in the action interface, so
    it is checked here rather than trusted.

    TOL is 1e-5, not machine epsilon, and the reason is worth writing down.
    `LAPARO_TIP_RPY` is authored to six decimals (2.356194 for 3*pi/4), and the
    URDF writer above formats it with "%g", which keeps six SIGNIFICANT digits
    and drops it again to 2.35619. The two roundings put the tip axes 3.2e-6
    from the exact table -- 0.0002 degrees. That is inherited from the Isaac
    builder, which formats the same way, and it is kept rather than fixed
    precisely so this URDF stays byte-identical to the one built on the RTX
    machine. Anything larger than 1e-5 is a real frame error, not formatting.
    """
    import sys
    sys.path.insert(0, str(ROOT / "isaac"))
    from laparo_ik import Chain

    TOL = 1e-5
    ch = Chain(urdf)
    # T_flange->tip is the tail of the chain: the two fixed steps after joint8.
    T = np.eye(4)
    for Tf, ax in ch.steps[-2:]:
        T = T @ Tf
    R = T[:3, :3]
    want = {"x": np.array([-0.7071068, 0.7071068, 0.0]),
            "y": np.array([0.7071068, 0.7071068, 0.0]),
            "z": np.array([0.0, 0.0, -1.0])}
    bad = []
    worst = 0.0
    for i, (nm, w) in enumerate(want.items()):
        got = R[:, i]
        d = float(np.abs(got - w).max())
        worst = max(worst, d)
        if d > TOL:
            bad.append(f"tip +{nm} = {got.round(6)}, expected {w} (off {d:.2e})")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-6:
        bad.append(f"det(R) = {det:.9f}, expected +1")
    if bad:
        raise SystemExit(f"{arm}: tool_tip frame does NOT match "
                         f"VLA_HANDOFF.md section 2:\n  " + "\n  ".join(bad))
    log(f"  verified {arm} tool_tip frame vs VLA_HANDOFF.md section 2 "
        f"(insertion = -z, det=+1, worst axis error {worst:.1e})")


def chain_signature(urdf: Path) -> np.ndarray:
    """The numbers `laparo_ik.Chain` actually reads, flattened.

    Used to decide whether an existing URDF is equivalent to the one this
    builder would write, without caring about XML formatting or element order.
    """
    import sys
    sys.path.insert(0, str(ROOT / "isaac"))
    from laparo_ik import Chain
    ch = Chain(urdf)
    parts = [ch.lo, ch.hi]
    for Tf, ax in ch.steps:
        parts.append(Tf.ravel())
        parts.append(np.zeros(3) if ax is None else np.asarray(ax, float))
    return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def build_all(force: bool = False) -> dict:
    """Build the two arm URDFs, or validate the ones already there.

    `out/fr3_left_grasper.urdf` and `out/fr3_right_cutter.urdf` are the ONE
    interface the Isaac and Newton backends share -- `laparo_ik.Chain` parses
    them on both machines, and `laparo_control.BEST` is only meaningful if the
    two agree. On the RTX workstation `isaac/build_surgical_cell.py` writes
    them as part of the full cell build.

    So this does not blindly overwrite. If a URDF is already present, it is
    parsed and compared against what would be written; an equivalent one is
    kept (and reported), a differing one is a hard error naming `--force`.
    That makes running this on the RTX machine a VALIDATION of the Isaac build
    rather than a silent replacement of it.
    """
    fr3 = build_fr3_local()
    out, kept = {}, []
    log("arm + tool URDFs")
    for arm, T in tools().items():
        dest = ROOT / T.urdf
        if dest.is_file() and not force:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                probe = ToolSpecProbe(T, Path(td) / dest.name)
                build_arm_tool_urdf(probe, fr3)
                a, b = chain_signature(dest), chain_signature(probe.abs_dest)
            if a.shape == b.shape and np.abs(a - b).max() < 1e-12:
                log(f"  {arm}: kept existing {dest.relative_to(ROOT)} "
                    f"(kinematically identical to this build)")
                out[arm], _ = dest, kept.append(arm)
                continue
            raise SystemExit(
                f"{dest.relative_to(ROOT)} already exists and does NOT match "
                f"what newton_cell would build (max chain difference "
                f"{np.abs(a - b).max() if a.shape == b.shape else float('nan'):.3e}).\n"
                f"That URDF is shared with the Isaac backend, so it is not "
                f"overwritten automatically. Re-run with --force to replace "
                f"it, or delete it and rebuild.")
        out[arm] = build_arm_tool_urdf(T, fr3)
    for arm, p in out.items():
        verify_tip_frame(p, arm)
    return out


class ToolSpecProbe:
    """A ToolSpec that writes somewhere else, for the compare-don't-clobber path."""

    def __init__(self, T, dest: Path):
        self._T = T
        self.abs_dest = dest
        self.urdf = str(dest)

    def __getattr__(self, k):
        return getattr(self._T, k)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing out/*.urdf instead of validating "
                         "them (they are shared with the Isaac backend)")
    a = ap.parse_args(argv)
    build_all(force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
