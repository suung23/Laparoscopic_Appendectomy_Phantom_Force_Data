#!/usr/bin/env python3
"""The two cameras and the scope light, placed as the Isaac cell places them.

Ports the geometry of `build_surgical_cell.py`'s `endoscope_tip_frame`,
`bevelled_tip_basis`, `add_endoscope_camera`, `add_endoscope_light` and
`add_realsense_405`, minus the USD authoring.

The endoscope is a 30-degree FORWARD-OBLIQUE laparoscope, not a 0-degree one:
the objective and the light fibre both sit on an end face ground at 30 degrees
to the plane perpendicular to the shaft, so the view direction AND the
illumination axis leave the tip 30 degrees off the shaft. Getting this wrong is
30 degrees of view direction unaccounted for, which the Isaac cell was guilty of
until it was fixed there too.
"""
from __future__ import annotations

import math

import numpy as np

from cell_spec import ROOT, cli_defaults, constants
from layout import CellLayout
from render import Camera

# The calibrated pinhole build_surgical_cell.py authors on the scope camera:
# ChArUco calibration of the real laparoscope (endoscope_interinsic/, 30 frames,
# RMS 0.58 px) at 1280x720. The renderer uses SQUARE pixels -- the Isaac cell
# notes the RTX viewport ignores verticalAperture, so the calibration's 2 %
# fx/fy anisotropy cannot be expressed and fy is taken as fx. Matching that here
# keeps the projection identical across the two backends.
ENDO_INTRINSICS = (684.128, 697.759, 611.381, 297.075)   # fx, fy, ppx, ppy


def _minus_z_rotation(direction, roll_deg: float = 0.0) -> np.ndarray:
    """Column-convention R taking local -Z onto `direction`, with a roll.

    Equivalent to build_surgical_cell._minus_z_rotation, which builds it with
    Gf (row-vector) as `Rotation(+Z, roll) * Rotation(-Z, direction)`; in column
    convention that composition reverses.
    """
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, d)
    c = float(np.dot(a, d))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        R_align = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    if roll_deg:
        t = math.radians(roll_deg)
        Rz = np.array([[math.cos(t), -math.sin(t), 0.0],
                       [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
        return R_align @ Rz
    return R_align


def bevelled_tip_basis(shaft_dir, roll_deg=180.0, tilt_deg=None, clock_deg=None):
    """(right, up, forward) for a tip ground at `tilt_deg` to the shaft normal.

    Starts from the basis a 0-degree scope would have -- boresight along
    `shaft_dir`, `roll_deg` about it -- then rolls by `clock_deg` about the
    shaft to point the bevel, and swings the boresight `tilt_deg` toward image
    DOWN. tilt 0 reproduces the 0-degree basis exactly.
    """
    C = constants()
    tilt = C["ENDO_TIP_ANGLE_DEG"] if tilt_deg is None else tilt_deg
    clock = C["ENDO_BEVEL_CLOCK_DEG"] if clock_deg is None else clock_deg
    R = _minus_z_rotation(shaft_dir, roll_deg)
    right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
    c, s = math.cos(math.radians(clock)), math.sin(math.radians(clock))
    right, up = c * right + s * up, -s * right + c * up
    ct, st = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    fwd, up = ct * fwd - st * up, st * fwd + ct * up
    return right, up, fwd


class CellCameras:
    """Endoscope + D405, and the scope tip light, in world coordinates."""

    def __init__(self, layout: CellLayout, organ_centroid_world):
        self.L = layout
        self.C = constants()
        self.D = cli_defaults()
        self.organ_c_world = np.asarray(organ_centroid_world, np.float64)
        self._tip_frame_local()

    def _tip_frame_local(self):
        """Endoscope tip + shaft axis, in the scope's LOCAL frame.

        The camera and light are children of the endoscope prim in the Isaac
        cell, so their transforms are authored in its local frame and the prim's
        own measured pose carries them to world. Same split here.
        """
        import trimesh
        v = np.asarray(trimesh.load(ROOT / "mesh" / "endoscope.ply",
                                    process=False).vertices, np.float64) * 0.001
        M = self.L.T_world_endoscope
        organ_local = np.linalg.inv(M) @ np.append(self.organ_c_world, 1.0)
        organ_local = organ_local[:3]

        c = v.mean(axis=0)
        axis = np.linalg.svd(v - c)[2][0]
        proj = (v - c) @ axis
        ends = [v[proj.argmin()], v[proj.argmax()]]
        if np.linalg.norm(ends[0] - organ_local) < np.linalg.norm(ends[1] - organ_local):
            tip, tail = ends[0], ends[1]
        else:
            tip, tail = ends[1], ends[0]
        shaft = tip - tail
        self.tip_local = tip
        self.shaft_local = shaft / np.linalg.norm(shaft)

    # ---- endoscope -------------------------------------------------------
    @property
    def endoscope(self) -> Camera:
        W, H = self.C["ENDO_CAM_RES"]
        fx, fy, ppx, ppy = ENDO_INTRINSICS
        right, up, fwd = bevelled_tip_basis(self.shaft_local)
        M = self.L.T_world_endoscope
        R_w = M[:3, :3]
        pos = R_w @ self.tip_local + M[:3, 3]
        # USD camera: local +X right, +Y UP, -Z forward. OpenCV wants +Y DOWN
        # and +Z forward, so the y and z axes flip.
        r_w = R_w @ right
        u_w = R_w @ (-up)
        f_w = R_w @ fwd
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r_w, u_w, f_w, pos
        return Camera(W, H, fx, fx, ppx, ppy, T)     # square pixels: fy = fx

    @property
    def endoscope_light(self) -> dict:
        """The fibre bundle at the scope tip.

        Shares the tip's bevel, so it illuminates what the camera looks at. The
        cone was widened from 55 to 80 degrees in the Isaac cell because the
        dome walls sit far off the axis and were reading as dim brown shadow;
        keeping that here is what stops the surround going black.
        """
        C = self.C
        _, _, fwd = bevelled_tip_basis(self.shaft_local)
        M = self.L.T_world_endoscope
        R_w = M[:3, :3]
        pos = R_w @ self.tip_local + M[:3, 3]
        half = math.radians(C["ENDO_CONE_DEG"])
        inten = self.D["--endoscope-intensity"] * (2.0 ** C["ENDO_LIGHT_EXPOSURE"])
        return dict(
            pos=pos,
            dir=R_w @ fwd,
            color=np.asarray(C["ENDO_LIGHT_COLOR"], np.float64),
            # USD intensity is nits over the emitter area; the tracer wants a
            # radiant scale, so it is normalised by the emitter's area the same
            # way the RTX sphere light integrates it.
            intensity=inten * (C["ENDO_LIGHT_RADIUS"] ** 2) * 4.0 * math.pi,
            radius=C["ENDO_LIGHT_RADIUS"],
            cos_outer=math.cos(half),
            softness=C["ENDO_CONE_SOFTNESS"],
        )

    @property
    def front(self):
        return front_camera(self.L)

    # ---- D405 ------------------------------------------------------------
    @property
    def d405(self) -> Camera:
        W, H = self.C["D405_RES"]
        pos = self.L.d405_position
        # looks straight down -Z with image axes parallel to world X/Y; in
        # OpenCV terms that is +x = world +x, +y = world -y, +z = world -z
        T = np.eye(4)
        T[:3, 0] = (1.0, 0.0, 0.0)
        T[:3, 1] = (0.0, -1.0, 0.0)
        T[:3, 2] = (0.0, 0.0, -1.0)
        T[:3, 3] = pos
        f = (W / 2.0) / math.tan(math.radians(self.C["D405_HFOV_DEG"]) / 2.0)
        return Camera(W, H, f, f, W / 2.0, H / 2.0, T)

    def d405_right(self, baseline=0.018) -> Camera:
        """The stereo partner, `baseline` along the left imager's +x.

        Authored as a rigid offset of the calibrated left camera, so the pair is
        structurally rectified and `rgb`/`rgb_right` go straight into a matcher.
        """
        c = self.d405
        T = c.T.copy()
        T[:3, 3] = T[:3, 3] + T[:3, 0] * baseline
        return Camera(c.width, c.height, c.fx, c.fy, c.cx, c.cy, T)

    def describe(self) -> str:
        e, d = self.endoscope, self.d405
        L = self.endoscope_light
        return "\n".join([
            f"endoscope {e.width}x{e.height} fx={e.fx:.1f} c=({e.cx:.1f},{e.cy:.1f}) "
            f"hfov={e.hfov_deg():.1f} vfov={e.vfov_deg():.1f} deg",
            f"  tip   {(e.T[:3,3]*1000).round(1)} mm",
            f"  fwd   {e.T[:3,2].round(4)}   (30 deg off shaft "
            f"{(self.L.T_world_endoscope[:3,:3] @ self.shaft_local).round(4)})",
            f"  light intensity {L['intensity']:.4g}, cone "
            f"{math.degrees(math.acos(L['cos_outer'])):.0f} deg, "
            f"colour {np.asarray(L['color']).round(3)}",
            f"d405      {d.width}x{d.height} fx={d.fx:.1f} hfov="
            f"{d.hfov_deg():.1f} deg at {(d.T[:3,3]*1000).round(1)} mm, looking -Z",
        ])


def front_camera(layout, width=720, height=540, hfov_deg=42.0,
                 distance=1.6, height_m=0.62, look_z=0.16):
    """The cell overview: both arms, the phantom, the table, from the front.

    Not a clinical camera -- nothing on the rig sees this. It is the view the
    reference render pairs with the endoscope feed so a reader can tell at a
    glance which way the arms are pointing, and it is what makes an eval video
    interpretable without knowing the port layout.

    Placed on the phantom's -Y side (the side the arms face across), looking
    slightly down at the specimen.
    """
    import numpy as _np
    from render import Camera
    origin = _np.array([0.0, layout.base_y - distance, height_m])
    target = _np.array([0.0, 0.0, look_z])
    f = target - origin
    f = f / _np.linalg.norm(f)
    right = _np.cross(f, _np.array([0.0, 0.0, 1.0]))
    right = right / _np.linalg.norm(right)
    up = _np.cross(f, right)                       # OpenCV: +y is DOWN
    T = _np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, f, origin
    return Camera.from_hfov(width, height, hfov_deg, T)
