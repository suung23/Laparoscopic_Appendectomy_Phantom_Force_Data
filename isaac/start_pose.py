#!/usr/bin/env python3
"""Shared start-pose geometry for the bimanual cell.

Both arms must start with the instrument **inserted `insert` metres past its own
trocar port** and with the tool tip **near the middle of the endoscope image**,
far enough apart that they do not touch.

Insertion depth is a hard constraint: the tip sits exactly `insert` from the port
centre. The only freedom is the DIRECTION the shaft takes through the port, and
that direction is what this module solves for -- inside a tilt cone around the
port's inward surface normal, so the shaft still passes through the hole at a
physically sane angle.

That spawn is a PARKED pose, not the working one: at 10 cm the tips are still
short of the organ bed, which is what keeps them from being cooked inside
tissue. The scripted APPROACH below then flies both tips to the measured
working pose (`APPROACH_TIP_WORLD`) before any operator or policy input is
allowed through -- see `approach_command`.

Used by `isaac/teleop_left_touch.py` (live start pose) and
`isaac/build_surgical_cell.py` (baked into the saved stage), so the scene opens
in the same pose the teleop starts from.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from laparo_ik import TROCAR_PORTS_MM, port_inward_normal  # noqa: E402,F401


# --------------------------------------------------------------------------
# endoscope camera
# --------------------------------------------------------------------------
ENDO_CAM_PATH = "/World/colliders/endoscope/endoscope_cam"


def endoscope_camera(stage, path: str = ENDO_CAM_PATH) -> dict:
    """The scope camera as a world-frame pinhole -- the one the RENDERER draws.

    USD cameras look along their local -Z with +X image-right and +Y image-UP;
    pixel v grows DOWNWARD, hence the sign flip on the up axis in `project`.

    The pixel scale and principal point are NOT read off the authored attributes
    naively. Both were measured against the rendered image by
    `isaac/check_projection.py` (a 3x3 emissive grid at a known depth, centroids
    fitted in the PNG), which found the RTX viewport to:

      * use SQUARE pixels: fy = fx = W*focal/horizontalAperture. The authored
        verticalAperture is ignored, so the calibration's 2 % fx/fy anisotropy
        cannot be expressed through this path -- see `fy_calibrated` below.
      * offset the principal point as
            cx = W/2 - hoff * (W/hap)
            cy = H/2 + voff * (W/hap)
        i.e. mirrored in u, and scaled by the HORIZONTAL aperture in v (square
        pixels again), not by verticalAperture/H.

    Decoding by the rendered convention is right for a stage authored either
    way: it always reports where the renderer actually puts the optical centre.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"no endoscope camera at {path}")
    M = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    T = np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    R = T[:3, :3]
    # strip any scale the ancestor chain carries
    R = R / np.linalg.norm(R, axis=0, keepdims=True)

    cam = UsdGeom.Camera(prim)
    f = float(cam.GetFocalLengthAttr().Get())
    hap = float(cam.GetHorizontalApertureAttr().Get())
    vap = float(cam.GetVerticalApertureAttr().Get())
    hoff = float(cam.GetHorizontalApertureOffsetAttr().Get() or 0.0)
    voff = float(cam.GetVerticalApertureOffsetAttr().Get() or 0.0)
    W = prim.GetAttribute("endocam:width").Get() or 1280
    H = prim.GetAttribute("endocam:height").Get() or 720
    W, H = int(W), int(H)

    px_per_ap = W / hap                     # pixels per aperture unit
    fx = fy = W * f / hap
    cx = W / 2.0 - hoff * px_per_ap
    cy = H / 2.0 + voff * px_per_ap
    return dict(pos=T[:3, 3], right=R[:, 0], up=R[:, 1], fwd=-R[:, 2],
                fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H,
                fy_calibrated=H * f / vap,   # what the USD claims; not rendered
                hfov=2.0 * math.degrees(math.atan(W / (2.0 * fx))),
                vfov=2.0 * math.degrees(math.atan(H / (2.0 * fy))))


def aperture_offsets(ppx, ppy, W, H, hap):
    """Inverse of the decode above: the (horizontal, vertical) aperture offsets
    that put the RENDERED principal point on (ppx, ppy)."""
    px_per_ap = W / hap
    return (W / 2.0 - ppx) / px_per_ap, (ppy - H / 2.0) / px_per_ap


def project(cam: dict, p) -> tuple[float, float, float]:
    """World point -> (u px, v px, depth m along the boresight)."""
    d = np.asarray(p, np.float64) - cam["pos"]
    z = float(np.dot(d, cam["fwd"]))
    if z <= 1e-6:
        return float("nan"), float("nan"), z
    x = float(np.dot(d, cam["right"]))
    y = float(np.dot(d, cam["up"]))
    return cam["cx"] + cam["fx"] * x / z, cam["cy"] - cam["fy"] * y / z, z


def in_frame(cam: dict, u, v, margin_px: float = 0.0) -> bool:
    return (margin_px <= u < cam["W"] - margin_px
            and margin_px <= v < cam["H"] - margin_px)


# --------------------------------------------------------------------------
# the aim solver
# --------------------------------------------------------------------------
def _cone_directions(n, max_tilt_deg, n_tilt=25, n_az=72):
    """Unit directions inside a cone of half-angle `max_tilt_deg` about `n`."""
    n = np.asarray(n, np.float64)
    n = n / np.linalg.norm(n)
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    tilts = np.linspace(0.0, math.radians(max_tilt_deg), n_tilt)
    az = np.linspace(0.0, 2.0 * math.pi, n_az, endpoint=False)
    T, A = np.meshgrid(tilts, az, indexing="ij")
    T, A = T.ravel(), A.ravel()
    d = (np.cos(T)[:, None] * n[None, :]
         + (np.sin(T) * np.cos(A))[:, None] * e1[None, :]
         + (np.sin(T) * np.sin(A))[:, None] * e2[None, :])
    return d / np.linalg.norm(d, axis=1, keepdims=True), T


def _aim_one(cam, p_port, n_in, insert, target_uv, max_tilt_deg,
             organ_pts=None, min_clear=0.008, w_tilt=0.6, w_clear=4.0):
    """Best insertion direction for ONE port: the tip lands `insert` past the
    port and as close as possible to `target_uv` in the endoscope image.

    Cost is in pixels; the tilt term is a mild preference for entering along the
    surface normal (in px-equivalent per radian) and the clearance term is a
    steep penalty for parking the tip inside tissue."""
    dirs, tilts = _cone_directions(n_in, max_tilt_deg)
    tips = p_port[None, :] + insert * dirs

    d = tips - cam["pos"][None, :]
    z = d @ cam["fwd"]
    x = d @ cam["right"]
    y = d @ cam["up"]
    ok = z > 1e-3
    u = np.where(ok, cam["cx"] + cam["fx"] * x / np.where(ok, z, 1.0), 1e6)
    v = np.where(ok, cam["cy"] - cam["fy"] * y / np.where(ok, z, 1.0), 1e6)

    cost = np.hypot(u - target_uv[0], v - target_uv[1])
    cost = cost + w_tilt * np.degrees(tilts) * cam["fx"] / 1000.0
    # keep the tip in front of the scope and inside the frame
    cost = np.where(ok, cost, 1e9)
    cost = cost + 1e5 * (z < 0.015)

    if organ_pts is not None and len(organ_pts):
        clear = _min_dist(tips, organ_pts)
        cost = cost + w_clear * cam["fx"] * np.maximum(0.0, min_clear - clear)

    i = int(np.argmin(cost))
    return dict(dir=dirs[i], tip=tips[i], tilt_deg=float(np.degrees(tilts[i])),
                u=float(u[i]), v=float(v[i]), z=float(z[i]),
                cost=float(cost[i]))


def _min_dist(pts, cloud, max_cloud=6000, chunk=64):
    """Per-point distance to the nearest cloud point (numpy, chunked).

    The cloud is decimated to `max_cloud` points -- this only has to be accurate
    to a few millimetres (it feeds a soft clearance penalty), and the organ
    render meshes carry 10^5 vertices."""
    cloud = np.asarray(cloud, np.float64)
    if len(cloud) > max_cloud:
        cloud = cloud[:: max(1, len(cloud) // max_cloud)]
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        block = pts[i:i + chunk]
        dd = np.linalg.norm(block[:, None, :] - cloud[None, :, :], axis=2)
        out[i:i + chunk] = dd.min(axis=1)
    return out


def solve_start_tips(cam, ports, insert=0.10, min_sep=0.045,
                     max_tilt_deg=60.0, v_frac=0.5, organ_pts=None,
                     min_clear=0.008, max_iter=40, verbose=None):
    """Start tips for both arms.

    ports: {port_no: (p_world, inward_normal_world)} -- exactly two entries.

    Both tips land `insert` past their own port, aimed at the middle of the
    endoscope image and split horizontally by whatever it takes to hold
    `min_sep` between them. Each arm keeps the image side its port naturally
    falls on, so the shafts do not cross.

    Returns {port_no: dict(tip, dir, tilt_deg, u, v, z)} plus a "_meta" entry.
    """
    say = verbose or (lambda *_: None)
    nos = sorted(ports)
    assert len(nos) == 2, "expected exactly the two used ports"

    # which port shows up on which side of the image, aiming straight in
    side = {}
    for k in nos:
        p, n = ports[k]
        u, _, _ = project(cam, p + insert * n)
        side[k] = u
    left_port = min(nos, key=lambda k: side[k])     # smaller u = image-left
    right_port = max(nos, key=lambda k: side[k])

    # Aim at the middle of the FRAME, not at the optical axis: this scope's
    # calibrated principal point sits 29 px left of and 63 px above the frame
    # centre, and "in the middle of the endoscope view" means the middle of the
    # picture the operator looks at.
    u_centre, v_target = cam["W"] / 2.0, v_frac * cam["H"]
    d_px = 0.06 * cam["W"]                          # initial half-split
    best = None
    for it in range(max_iter):
        res = {}
        for k, sgn in ((left_port, -1.0), (right_port, +1.0)):
            p, n = ports[k]
            res[k] = _aim_one(cam, p, n, insert,
                              (u_centre + sgn * d_px, v_target),
                              max_tilt_deg, organ_pts, min_clear)
        sep = float(np.linalg.norm(res[nos[0]]["tip"] - res[nos[1]]["tip"]))
        say(f"  iter {it:2d}: split {d_px:6.1f} px -> tips at "
            f"({res[left_port]['u']:.0f},{res[left_port]['v']:.0f}) / "
            f"({res[right_port]['u']:.0f},{res[right_port]['v']:.0f}) px, "
            f"separation {sep*1000:.1f} mm")
        if best is None or (sep >= min_sep and best["sep"] < min_sep) or \
                (sep >= min_sep and best["sep"] >= min_sep
                 and _img_cost(cam, res) < best["img"]):
            best = dict(res=res, sep=sep, d_px=d_px,
                        img=_img_cost(cam, res))
        if sep >= min_sep:
            break
        d_px *= 1.12
        if u_centre + d_px > cam["W"] * 0.92:
            say("  ! ran out of image width before reaching the separation")
            break

    out = dict(best["res"])
    out["_meta"] = dict(sep=best["sep"], split_px=best["d_px"],
                        left_port=left_port, right_port=right_port,
                        v_target=v_target, insert=insert)
    return out


def _img_cost(cam, res):
    """How far the two tips sit from the middle of the frame, in px (mean)."""
    return float(np.mean([math.hypot(r["u"] - cam["W"] / 2.0,
                                     r["v"] - cam["H"] / 2.0)
                          for r in res.values()]))


# --------------------------------------------------------------------------
# the working pose, and the scripted move that reaches it
# --------------------------------------------------------------------------
# Where each tool tip has to be before the operator or the policy takes over.
# MEASURED, not solved: both instruments were parked by hand in the GUI where
# they should open on the appendix, and the numbers below were read straight
# off Property > Transform on /World/robots/fr3_<arm>/robot/tool_tip.
#
# Stored in WORLD metres because that is the frame the thing they are aimed at
# lives in. The panel reports the tip LOCAL to the arm's own robot prim, and
# those readings are kept beside it for provenance -- but a local target moves
# with the arm base, so rebuilding the cell at a different --spacing would slide
# both tips off the organs while the numbers stayed the same. World does not.
#
#   left  local (0.62169, -0.45275, 0.09388) under fr3_left/robot   (base at
#   right local (0.62447,  0.42404, 0.08901) under fr3_right/robot   -+0.45,
#                                             both yawed +90 deg)  -0.62311)
APPROACH_TIP_WORLD = {
    "left":  np.array([0.00275, -0.00142, 0.09388]),
    "right": np.array([0.02596,  0.00136, 0.08901]),
}
# The pair sits 23.8 mm apart, 12.0 mm (left) and 7.0 mm (right) off the
# nearest appendix surface, both inside the scope frame at (575, 212) and
# (879, 222) px of 1280x720. Re-measure with isaac/probe_tips.py after any
# change to the specimen pose or the scope.


def approach_tip(arm: str, override=None) -> np.ndarray:
    """The working tip for one arm, world metres.

    `override` accepts "x,y,z" (or a 3-sequence) so a caller can expose the
    target as a flag without every caller re-parsing it."""
    if override is None or (isinstance(override, str) and not override.strip()):
        return APPROACH_TIP_WORLD[arm].copy()
    if isinstance(override, str):
        override = [float(v) for v in override.split(",")]
    p = np.asarray(override, np.float64)
    if p.shape != (3,):
        raise ValueError(f"approach tip for {arm} wants x,y,z, got {override!r}")
    return p


def spawn_tips(ports, arm_of_port, insert: float, override=None) -> dict:
    """The PARKED tips: each working tip, withdrawn along its own port ray.

    This is the spawn aim, and it replaces pointing the parked tools at the
    middle of the endoscope image. That objective is wrong for a pose this
    shallow, and measurably so: the scope is advanced 135 mm into the cavity,
    so "the middle of the image, 100 mm past the port" is a point level with
    the scope tip itself -- the solved left tip came out 1.5 mm from the camera
    origin and projected to (nan, nan), i.e. behind it, and the IK gave up at
    20.7 mm of tip error against its joint limits. Aiming at the image only
    works while the tools are deeper than the camera, which is why the depth
    used to have to track --endo-advance.

    Withdrawing the WORKING pose has neither problem. The direction is one the
    arm is known to reach, because it is the direction it will be holding when
    the episode starts; the parked tip is behind the working one along that
    ray, which is out of the organs by construction; and the approach that
    follows is then a pure slide down a single ray, with no pivoting against
    the trocar on the way in.

    Returns {port_no: dict(tip, dir, tilt_deg)}, matching solve_start_tips'
    shape minus the image terms, which mean nothing for a parked tool.
    """
    out = {}
    for port_no, (p_port, n_in) in ports.items():
        arm = arm_of_port[port_no]
        p_work = approach_tip(arm, (override or {}).get(arm))
        d = p_work - np.asarray(p_port, np.float64)
        r = float(np.linalg.norm(d))
        if r < 1e-9:
            raise ValueError(f"{arm}: working tip is on top of its port")
        u = d / r
        out[port_no] = dict(
            tip=np.asarray(p_port, np.float64) + insert * u, dir=u,
            tilt_deg=math.degrees(math.acos(float(np.clip(
                np.dot(u, np.asarray(n_in, np.float64)
                       / np.linalg.norm(n_in)), -1.0, 1.0)))))
    return out


def approach_command(p_port, p_from, p_to, s: float) -> np.ndarray:
    """The tip command a fraction `s` (0..1) of the way from `p_from` to `p_to`.

    NOT a straight line. The shaft pivots in the trocar, so the reachable tips
    are exactly the points on some ray out of the port: the path slerps the
    port->tip DIRECTION and lerps the DEPTH, which keeps every intermediate
    command on a pose the RCM constraint can actually hold. A straight lerp
    cuts the chord instead, and the controller spends the whole move trading
    tip error against RCM error -- on a 40 mm move with a 25 deg direction
    change the chord leaves the valid set by ~4 mm at the midpoint, which is
    the tip pushing sideways against the trocar the entire way in.
    """
    s = float(np.clip(s, 0.0, 1.0))
    p_port = np.asarray(p_port, np.float64)
    d0, d1 = np.asarray(p_from, np.float64) - p_port, \
        np.asarray(p_to, np.float64) - p_port
    r0, r1 = float(np.linalg.norm(d0)), float(np.linalg.norm(d1))
    if r0 < 1e-9 or r1 < 1e-9:                 # degenerate: nothing to pivot
        return (1.0 - s) * np.asarray(p_from, np.float64) + \
            s * np.asarray(p_to, np.float64)
    u0, u1 = d0 / r0, d1 / r1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    ang = math.acos(dot)
    if ang < 1e-6:                             # same ray: pure in/out
        u = u1
    else:
        u = (math.sin((1.0 - s) * ang) * u0 + math.sin(s * ang) * u1) \
            / math.sin(ang)
        u = u / np.linalg.norm(u)
    return p_port + ((1.0 - s) * r0 + s * r1) * u


def approach_duration(p_port, p_from, p_to, speed: float,
                      min_seconds: float = 0.5) -> float:
    """Seconds the move needs at `speed` m/s, measured ALONG the arc above --
    so a mostly-pivoting move is not rushed just because its endpoints are
    close together."""
    n = 32
    pts = [approach_command(p_port, p_from, p_to, i / n) for i in range(n + 1)]
    arc = float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))
    return max(min_seconds, arc / max(speed, 1e-6))


# --------------------------------------------------------------------------
# the start-pose IK, shared by the cell build and the teleop
# --------------------------------------------------------------------------
# SEED, MIRROR and solve_start_q live here rather than in either caller because
# the two used to carry their own copies: the teleop RE-SOLVES the start pose at
# startup instead of trusting the stage (so a cell built at a different
# --port-offset still starts correctly), which means any improvement made to one
# copy silently did nothing to the other. The elbow lift was added to the build
# and the live pose did not change -- that is the failure mode this removes.
SEED = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def solve_start_q(chain, T_base, tip, p_rcm, seed=None, roll_deg=0.0,
                  elbow_lift=0.6, elbow_point=3, ik_kw=None,
                  n_task=900, n_roll=300):
    """Joint angles for one arm's start pose: tip + RCM, then the shaft ROLL.

    tip/RCM converge first; the roll (the one orientation DOF the RCM leaves
    free) is then driven to 'tip +y as close to world +z as the shaft allows',
    plus `roll_deg` about the shaft. `elbow_lift` rides in the null space
    throughout so the elbow sits high in an arch instead of low over the table.

    Returns (q, info).
    """
    from laparo_ik import (JAW_PIN_AXIS_F, SHAFT_AXIS_F, elbow_up_gradient,
                           ik_step, perp_unit, rodrigues, task_error)
    kw = dict(ik_kw or {})
    q = (SEED if seed is None else np.asarray(seed, np.float64)).copy()

    def g_elbow(qq):
        return (elbow_up_gradient(chain, qq, T_base, elbow_lift, elbow_point)
                if elbow_lift else None)

    for _ in range(n_task):
        q = ik_step(chain, q, T_base, tip, p_rcm, w_rcm=1.0, iters=1,
                    g_extra=g_elbow(q), **kw)
    _, R_fl = chain.fk(q, T_base)
    a0w = R_fl @ SHAFT_AXIS_F
    r_cmd = perp_unit(R_fl @ JAW_PIN_AXIS_F, a0w)
    y_tip = perp_unit((R_fl @ chain.R_f2t)[:, 1], a0w)
    z_up = perp_unit(np.array([0.0, 0.0, 1.0]), a0w)
    phi = math.atan2(float(np.dot(a0w, np.cross(y_tip, z_up))),
                     float(np.dot(y_tip, z_up))) + math.radians(roll_deg)
    r_cmd = perp_unit(rodrigues(a0w, phi) @ r_cmd, a0w)
    for _ in range(n_roll):
        q = ik_step(chain, q, T_base, tip, p_rcm, w_rcm=1.0, r_cmd=r_cmd,
                    iters=1, g_extra=g_elbow(q), **kw)

    e, _, p_tip, _ = task_error(chain, q, T_base, tip, p_rcm, 1.0)
    _, R_fl = chain.fk(q, T_base)
    y_ach = (R_fl @ chain.R_f2t)[:, 1]
    a0w = R_fl @ SHAFT_AXIS_F
    info = dict(
        tip=p_tip, r_cmd=r_cmd,
        tip_err=float(np.linalg.norm(e[:3])),
        rcm_err=float(np.linalg.norm(e[3:])),
        depth=float(np.linalg.norm(p_tip - p_rcm)),
        margin=float(np.minimum(q - chain.lo, chain.hi - q).min()),
        elbow_z=float(chain.fk_points(q, T_base)[elbow_point][2]),
        y_up_deg=math.degrees(math.acos(float(np.clip(y_ach[2], -1, 1)))),
        y_best_deg=math.degrees(math.acos(float(np.clip(
            perp_unit(np.array([0.0, 0.0, 1.0]), a0w)[2], -1, 1)))))
    return q, info
