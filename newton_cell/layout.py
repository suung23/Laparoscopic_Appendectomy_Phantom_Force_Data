#!/usr/bin/env python3
"""Where everything in the surgical cell sits, in world coordinates.

A port of the placement half of `isaac/build_surgical_cell.py` -- the functions
that turn measured numbers into world poses (`phantom_placement_offset`,
`port_positions_local/world`, `phantom_apex_world_z`, the robot base line, the
D405 pose) with the USD authoring stripped out. The arithmetic is the original's;
the constants come from `cell_spec`, so both backends place the cell from one
set of numbers.

World frame, as the Isaac cell defines it: +z up, table top = robot-base plane
= z = 0, the phantom dome centred over the origin with its base on z = 0 and
then raised by `--lift`.

    from newton_cell.layout import CellLayout
    L = CellLayout()
    L.ports_world          # (11,3) trocar holes
    L.arm_base("left")     # (4,4) T_world_base
    L.T_world_rig          # (4,4) the frame policies use
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

from cell_spec import ROOT, cli_defaults, constants


def rot_z(pts, deg):
    """Rotate an (N,3) array about the world Z axis through the origin."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(pts, np.float64) @ R.T


def Rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_xyz(deg3):
    """USD RotateXYZ: X then Y then Z, applied in that order (R = Rz@Ry@Rx)."""
    rx, ry, rz = (math.radians(v) for v in deg3)
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rzz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rzz @ Ry @ Rx


def T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


class CellLayout:
    """Every world pose in the cell, derived the way the Isaac build derives it."""

    def __init__(self):
        self.C = constants()
        self.D = cli_defaults()

    # ---- phantom ---------------------------------------------------------
    @functools.cached_property
    def _phantom_verts_m(self) -> np.ndarray:
        import trimesh
        ph = trimesh.load(ROOT / "phantom.stl", process=False)
        return np.asarray(ph.vertices, np.float64) * 0.001

    @functools.cached_property
    def phantom_placement_offset(self) -> np.ndarray:
        """build_surgical_cell.phantom_placement_offset, verbatim.

        phantom.stl in metres, then z shifted so the base sits on z=0 and X/Y
        shifted to their mean, so the dome is centred over the origin. The
        measured trocar holes live in that same STL frame and take the same
        transform.
        """
        v = self._phantom_verts_m
        return np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()])

    @functools.cached_property
    def phantom_apex_world_z(self) -> float:
        """World Z of the dome top. The 180 deg spin does not touch Z, so this
        is just the mesh height plus the plate lift."""
        v = self._phantom_verts_m
        return float(v[:, 2].max() - v[:, 2].min()) + self.D["--lift"]

    def phantom_points_world(self) -> np.ndarray:
        """The dome as the cell places it: centred, based on z=0, spun, lifted."""
        v = self._phantom_verts_m - self.phantom_placement_offset
        v = rot_z(v, self.D["--rotate-phantom"])
        v[:, 2] += self.D["--lift"]
        return v

    # ---- trocar ports ----------------------------------------------------
    @functools.cached_property
    def ports_local(self) -> np.ndarray:
        """Trocar holes in the phantom's placed (pre-rotation, unlifted) frame."""
        off = self.phantom_placement_offset
        return np.asarray(self.C["TROCAR_PORTS_MM"], np.float64) * 0.001 - off

    @functools.cached_property
    def ports_world(self) -> np.ndarray:
        """Final world positions: phantom spin + plate lift.

        The ports are drilled through the phantom, so they carry its 180 deg
        Z-spin and the lift. This is what the robot placement and the D405 are
        measured against.
        """
        p = rot_z(self.ports_local, self.D["--rotate-phantom"])
        p[:, 2] += self.D["--lift"]
        return p

    @property
    def used_ports(self) -> tuple:
        return tuple(self.C["USED_PORT_IDX"])

    def port_of_arm(self, arm: str) -> int:
        """Which port index each arm uses.

        The whole specimen is spun 180 deg, which swaps the two used ports'
        physical sides -- so build_surgical_cell decides left/right "from the
        final world X below, not from the port number". Same rule here.
        """
        idx = list(self.used_ports)
        xs = self.ports_world[idx][:, 0]
        order = [idx[int(np.argmin(xs))], idx[int(np.argmax(xs))]]
        return order[0] if arm == "left" else order[1]

    # ---- organs / field --------------------------------------------------
    # The reconstruction lives in the D405 camera frame: +x right, +y DOWN the
    # image, +z away from the lens. The D405 looked straight DOWN at the animal,
    # so the camera's depth axis is the world's DOWN axis. Same matrix as
    # sim/scene.py's CAM_TO_WORLD; mapping depth to a horizontal axis instead
    # stands the rabbit on end inside the phantom.
    CAM_TO_WORLD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

    # build_deformable_usd.py's --drop-height: metres above the floor the
    # captured stack is dropped at.
    DROP_HEIGHT = 0.005

    @functools.cached_property
    def capture_offset(self) -> np.ndarray:
        """sim/scene.py's `placement_offset`, over the union of all organs.

        ONE shared translation for every organ and for the field. Centring each
        body on the phantom axis independently would throw away the relative
        anatomy the reconstruction captured -- the appendix ends up inside the
        cecum, and the field floats above the organs with nothing under them.
        """
        import numpy as _np
        vs = []
        for n in ("cecum", "bowel1", "bowel2", "bowel3", "stomach", "appendix"):
            p = ROOT / "tet" / f"{n}.npz"
            if p.exists():
                vs.append(_np.load(p)["vertices"].astype(_np.float64))
        stacked = np.vstack([v @ self.CAM_TO_WORLD.T for v in vs])
        return np.array([-stacked[:, 0].mean(),
                         -stacked[:, 1].mean(),
                         -stacked[:, 2].min() + self.DROP_HEIGHT])

    @property
    def organ_offset(self) -> np.ndarray:
        """The extra translation the CELL bakes on top of the capture placement.

        build_surgical_cell bakes spin+lift into the POINTS rather than an
        ancestor transform, because PhysX deformables step from cooked
        world-space points and snap back if the transform is an xformOp. Newton
        soft bodies take their vertices the same way, so the same bake applies.
        """
        e = np.asarray(self.C["ORGAN_EXTRA_TRANSLATE"], np.float64)
        return np.array([e[0], e[1], e[2] + self.D["--lift"]])

    def organs_to_world(self, pts: np.ndarray) -> np.ndarray:
        """Captured organ/field points (metres, camera frame) -> world.

        Two stages, and BOTH are needed -- this was wrong at first with only the
        second, which left the organs y/z-flipped and 70 mm above the reachable
        volume, so the instruments passed underneath them and never touched
        tissue:

        1. `build_deformable_usd.py`: camera frame -> world via CAM_TO_WORLD,
           then the shared capture offset that centres the stack over the
           phantom axis and drops it to `DROP_HEIGHT`.
        2. `build_surgical_cell.py:transform_specimen`: spin by
           `--rotate-organs` and translate by ORGAN_EXTRA_TRANSLATE + `--lift`.
        """
        p = np.asarray(pts, np.float64) @ self.CAM_TO_WORLD.T + self.capture_offset
        p = rot_z(p, self.D["--rotate-organs"])
        return p + self.organ_offset

    # ---- endoscope -------------------------------------------------------
    @property
    def T_world_endoscope(self) -> np.ndarray:
        """The scope body pose fitted to the real recording.

        build_surgical_cell authors this as `AddTranslateOp` then
        `AddRotateXYZOp`, which reads like translate-first -- R @ (p + t). It is
        not, and the scene itself settles the question: ENDO_POSE_* was FITTED
        so the scope view matches the real recording, so the correct composition
        is the one whose boresight lands on the organs.

        Measured, with the 30-degree forward-oblique bevel applied:

            R @ p + t   (this one)   boresight 10.6 deg off the organ centroid
            R @ (p + t)              boresight 74.8 deg off -- looking at nothing

        The first render used the other order and came back a flat white field:
        the tip light was sitting a couple of centimetres from white moulded
        plastic with no tissue in frame at all.
        """
        return T(rot_xyz(self.C["ENDO_POSE_ROTXYZ"]),
                 np.asarray(self.C["ENDO_POSE_TRANSLATE"], np.float64))

    # ---- robots ----------------------------------------------------------
    @functools.cached_property
    def base_y(self) -> float:
        """The base line: --port-offset in -Y from the used-ports midpoint."""
        mid = self.ports_world[list(self.used_ports)].mean(axis=0)
        return float(mid[1]) - self.D["--port-offset"]

    def arm_base(self, arm: str) -> np.ndarray:
        """T_world_base for one FR3.

        Bases are symmetric about X=0, --spacing apart, and both arms face the
        phantom (+Y), which is a +90 deg yaw of the base frame.
        """
        x = self.D["--spacing"] / 2.0
        px = -x if arm == "left" else +x
        return T(Rz(90.0), np.array([px, self.base_y, 0.0]))

    # ---- the rig frame policies use -------------------------------------
    @functools.cached_property
    def T_world_rig(self) -> np.ndarray:
        """rig <- world, per VLA_HANDOFF.md section 2.

        Origin at the midpoint of the two FR3 base origins, +x from the left
        base toward the right base, +z world up, +y = z cross x.
        """
        bl = self.arm_base("left")[:3, 3]
        br = self.arm_base("right")[:3, 3]
        origin = 0.5 * (bl + br)
        ex = br - bl
        ex[2] = 0.0
        ex /= np.linalg.norm(ex)
        ez = np.array([0.0, 0.0, 1.0])
        ey = np.cross(ez, ex)
        return T(np.column_stack([ex, ey, ez]), origin)

    def to_rig(self, p_world) -> np.ndarray:
        M = self.T_world_rig
        R, t = M[:3, :3], M[:3, 3]
        return (np.asarray(p_world, np.float64) - t) @ R

    # ---- D405 ------------------------------------------------------------
    @functools.cached_property
    def d405_position(self) -> np.ndarray:
        """XY at the midpoint of the four highest ports, Z just under the apex.

        The module looks straight down (-Z) with its image axes parallel to
        world X/Y, so its rotation is identity in USD camera convention.
        """
        pw = self.ports_world
        top4 = np.argsort(pw[:, 2])[-4:]
        xy = pw[top4][:, :2].mean(axis=0)
        z = self.phantom_apex_world_z - self.D["--camera-drop"]
        return np.array([xy[0], xy[1], z])

    # ---- table -----------------------------------------------------------
    @property
    def table_depth(self) -> float:
        """--table-depth, with the documented AUTO rule when it is 0.

        depth = round_5cm(0.80 + max(0, port_offset - 0.27)), so the bases keep
        a constant back margin as the stand-off grows.
        """
        d = self.D["--table-depth"]
        if d > 0:
            return float(d)
        base = self.C["TABLE_SIZE"][1]
        grow = max(0.0, self.D["--port-offset"] - 0.27)
        return round((base + grow) / 0.05) * 0.05

    def describe(self) -> str:
        pw = self.ports_world
        lines = [
            f"phantom   offset {self.phantom_placement_offset.round(4)} m, "
            f"apex z {self.phantom_apex_world_z:.4f} m, "
            f"spin {self.D['--rotate-phantom']:.0f} deg, lift {self.D['--lift']*1000:.0f} mm",
            f"ports     {len(pw)} holes; used {self.used_ports} -> "
            f"left=port{self.port_of_arm('left')+1} right=port{self.port_of_arm('right')+1}",
        ]
        for arm in ("left", "right"):
            i = self.port_of_arm(arm)
            lines.append(f"  {arm:5s} port {i+1}: world {pw[i].round(4)}  "
                         f"rig {self.to_rig(pw[i]).round(4)}")
        for arm in ("left", "right"):
            b = self.arm_base(arm)
            lines.append(f"{arm:9s} base at {b[:3,3].round(4)} m, yaw +90 deg")
        lines += [
            f"base line y {self.base_y:+.4f} m ({self.D['--port-offset']*100:.0f} cm "
            f"from the used-port midpoint)",
            f"rig origin  {self.T_world_rig[:3,3].round(4)} m",
            f"organs      baked offset {self.organ_offset.round(5)} m, "
            f"spin {self.D['--rotate-organs']:.0f} deg",
            f"d405        {self.d405_position.round(4)} m, looking -Z",
            f"table       {self.C['TABLE_SIZE'][0]:.2f} x {self.table_depth:.2f} x "
            f"{self.C['TABLE_SIZE'][2]:.2f} m",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print(CellLayout().describe())
