#!/usr/bin/env python3
"""The trocar port as the PHYSICAL constraint it is, not a soft target.

The rig is not a free pivot. Around each trocar hole the phantom carries a
30 mm disc of thin, nearly-rigid rubber lying flat on the shell -- take it as
the tangent plane at the port -- with a 6 mm hole punched at the centre. The
tool shaft is a 5 mm tube (`laparo_shaft.stl`, verified in
build_surgical_cell.py). So:

    hole radius  R = 3.0 mm
    shaft radius r = 2.5 mm
    clearance ON THE AXIS = 0.5 mm

and that 0.5 mm is the WHOLE budget only while the shaft is perpendicular to
the sheet. A cylinder tilted by theta cuts the plane in an ELLIPSE with semi-
axes (r/cos theta) x r, so

    lateral budget(theta) = R - r / cos(theta)

        theta      0     10     13.3   18.3    25     30    33.56 deg
        budget   0.50   0.46    0.43   0.37   0.24   0.11   0.00  mm

The shipped start poses sit at 13.3 deg (left arm) and 18.3 deg (right), and
the 10 cm/s reference swings that to about 25 deg -- so the real lateral
tolerance in this cell is 0.24-0.43 mm, and past 33.6 deg the shaft cannot
pass the hole at ANY offset. The controller sweep used to gate the RCM error
at 8 mm p95.

Everything here is pure numpy and frame-agnostic: give it the port centre and
the INWARD surface normal in the same frame as the shaft line.
"""
from __future__ import annotations

import math

import numpy as np

HOLE_R = 0.0030          # m: 6 mm hole in the rubber
SHAFT_R = 0.0025         # m: 5 mm tool shaft
RUBBER_R = 0.0150        # m: 30 mm rubber disc (outside it, bare phantom)

# how many boundary samples the ellipse-vs-circle maximum uses. 512 puts the
# discretisation error at ~0.2 um, three orders below anything measured here.
_NPHI = 512
_PHI = np.linspace(0.0, 2.0 * math.pi, _NPHI, endpoint=False)
_COSP, _SINP = np.cos(_PHI), np.sin(_PHI)


# How far the CONTROLLER lets the shaft lean, which is deliberately not the
# geometric limit below. The modelled 6 mm hole stops a 5 mm shaft at 33.56 deg
# and the shipped cone_mm=0.15 reserve brings that to 28.7 -- tight enough that
# ordinary reaching across the field spends its time clamped, with the tip
# command pinned to the cone wall and the operator pushing against nothing
# (user: "tilt cone의 제약이 너무 심해"). The cap opens that to 60 deg.
#
# This does NOT pretend the rubber allows it: `port_geometry` is untouched, so
# `bite` keeps reporting the real interference and every metric built on it
# stays honest -- past 33.56 deg it simply reads as a shaft pressed into the
# sheet, which is what it would be. What changes is that the controller no
# longer refuses to go there. Set 0 to fall back to the geometric limit.
TILT_CAP_DEG = 60.0


def max_tilt_rad(margin=0.0, hole_r=HOLE_R, shaft_r=SHAFT_R):
    """Largest shaft tilt off the port normal that still leaves `margin` metres
    of lateral room for tracking error. margin=0 gives the hard geometric
    limit, where the ellipse exactly fills the hole and ANY lateral error
    touches rubber."""
    c = shaft_r / max(hole_r - margin, 1e-9)
    return math.acos(min(1.0, max(-1.0, c))) if c <= 1.0 else 0.0


def clamp_tilt_rad(cone_mm, cap_deg=TILT_CAP_DEG):
    """The tilt cap `cone_clamp` should be driven with, radians.

    One place, because the teleop, the VLA env and laparo_control each used to
    derive it from `cone_mm` on their own -- so a change to the rule had to be
    made three times or it half-applied. Returns 0.0 (= clamp off) when
    `cone_mm` is negative, exactly as those three did.
    """
    if cone_mm is None or cone_mm < 0.0:
        return 0.0
    return math.radians(cap_deg) if cap_deg and cap_deg > 0.0 \
        else max_tilt_rad(cone_mm * 1e-3)


def margin_at_tilt(tilt_rad, hole_r=HOLE_R, shaft_r=SHAFT_R):
    """Lateral budget (m) left at this tilt: how far the shaft axis may cross
    the sheet off the hole centre before it touches rubber."""
    return hole_r - shaft_r / max(math.cos(tilt_rad), 1e-9)


def port_geometry(centre, normal, p_tip, axis, hole_r=HOLE_R,
                  shaft_r=SHAFT_R):
    """Where the shaft actually goes through the rubber sheet.

    centre, normal : port centre and INWARD unit normal (the sheet is the
                     plane through `centre` with this normal)
    p_tip, axis    : a point on the shaft and its unit direction, pointing
                     INWARD (toward the tip). Either (3,) or (N,3).

    Returns a dict of arrays:
        off      lateral distance from the hole centre to the point where the
                 shaft AXIS pierces the sheet
        tilt     angle between the shaft and the port normal, rad
        reach    furthest point of the shaft's cross-section (an ellipse) from
                 the hole centre, in the sheet
        bite     reach - hole_r. POSITIVE = the shaft is pressing into the
                 rubber, by that much radially. Negative is clearance.

    `bite` is the number that matters: it is exact for a thin sheet, it folds
    offset and tilt together the way the hardware does, and it is signed, so
    "how much margin is left" and "how hard is it jammed" are one metric.
    """
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    c = np.asarray(centre, np.float64)
    p = np.atleast_2d(np.asarray(p_tip, np.float64))
    a = np.atleast_2d(np.asarray(axis, np.float64))
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    ca = a @ n                                   # cos(tilt)
    ca_s = np.where(np.abs(ca) < 1e-6, 1e-6, ca)
    s = ((c - p) @ n) / ca_s                     # distance along the shaft
    xc = p + a * s[:, None]                      # axis-through-sheet point
    e = xc - c                                   # lies in the sheet

    # in-sheet frame: t along the tilt azimuth, v perpendicular to it
    t = a - ca[:, None] * n
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    # perfectly perpendicular shafts have no azimuth; any in-sheet direction
    # then gives the same answer (the section is a circle), so pick one
    fallback = np.cross(n, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(fallback) < 1e-6:
        fallback = np.cross(n, np.array([0.0, 1.0, 0.0]))
    fallback = fallback / np.linalg.norm(fallback)
    t = np.where(tn > 1e-9, t / np.maximum(tn, 1e-12), fallback)
    v = np.cross(np.broadcast_to(n, t.shape), t)

    A = shaft_r / np.maximum(np.abs(ca), 1e-6)   # ellipse semi-major
    B = shaft_r                                  # ellipse semi-minor
    et = np.sum(e * t, axis=1)
    ev = np.sum(e * v, axis=1)
    # furthest boundary point of the offset ellipse from the hole centre
    x = et[:, None] + A[:, None] * _COSP[None, :]
    y = ev[:, None] + B * _SINP[None, :]
    reach = np.sqrt(np.max(x * x + y * y, axis=1))

    return dict(off=np.linalg.norm(e, axis=1),
                tilt=np.arccos(np.clip(np.abs(ca), -1.0, 1.0)),
                reach=reach,
                bite=reach - hole_r,
                depth=s)


def cone_clamp(p_cmd, centre, normal, tilt_max):
    """Pull a tip command back inside the cone the aperture allows.

    Under an RCM the shaft direction IS the port-to-tip direction, so a tip
    command outside this cone is a command for the shaft to lie on the rubber
    -- the arm cannot honour it and the servo just leans on the sheet. Clamped
    the same way the minimum-insertion floor is: keep the depth, keep the
    azimuth, only pull the tilt in.

    Returns (p_clamped, was_clamped).
    """
    c = np.asarray(centre, np.float64)
    n = np.asarray(normal, np.float64)
    n = n / np.linalg.norm(n)
    v = np.asarray(p_cmd, np.float64) - c
    d = float(np.linalg.norm(v))
    if d < 1e-9 or tilt_max <= 0.0:
        return np.asarray(p_cmd, np.float64), False
    u = v / d
    ct = float(np.dot(u, n))
    th = math.acos(min(1.0, max(-1.0, ct)))
    if th <= tilt_max:
        return np.asarray(p_cmd, np.float64), False
    # rotate u toward n by (th - tilt_max) about their common perpendicular
    ax = np.cross(n, u)
    an = float(np.linalg.norm(ax))
    if an < 1e-9:                       # anti-parallel: no defined azimuth
        return c + n * d, True
    ax = ax / an
    ang = tilt_max
    u2 = n * math.cos(ang) + np.cross(ax, n) * math.sin(ang)
    return c + u2 * d, True


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The geometry is the whole point, so it gets checked against cases whose
    # answer is known by hand rather than trusted.
    n = np.array([0.0, 0.0, 1.0])
    c = np.zeros(3)

    def one(off_mm, tilt_deg, azim_deg=0.0):
        th, az = math.radians(tilt_deg), math.radians(azim_deg)
        a = np.array([math.sin(th) * math.cos(az),
                      math.sin(th) * math.sin(az), math.cos(th)])
        # a point on the axis 50 mm past the sheet, offset in the sheet
        e = np.array([off_mm * 1e-3 * math.cos(az),
                      off_mm * 1e-3 * math.sin(az), 0.0])
        return port_geometry(c, n, e + a * 0.05, a)

    print("case                      off(mm) tilt(deg)  bite(mm)  expected")
    for off, tilt, exp in [
            (0.0, 0.0, SHAFT_R - HOLE_R),
            (0.5, 0.0, 0.0),
            (0.0, 33.5573, 0.0),
            (0.0, 20.0, SHAFT_R / math.cos(math.radians(20)) - HOLE_R),
            (0.24, 25.0, 0.24e-3 + SHAFT_R / math.cos(math.radians(25)) - HOLE_R),
            (1.22, 13.3, 1.22e-3 + SHAFT_R / math.cos(math.radians(13.3)) - HOLE_R)]:
        g = one(off, tilt)
        print(f"off {off:5.2f} mm tilt {tilt:6.2f} deg  "
              f"{g['off'][0]*1e3:7.3f} {math.degrees(g['tilt'][0]):8.3f} "
              f"{g['bite'][0]*1e3:9.4f}  {exp*1e3:9.4f}")
        assert abs(g["bite"][0] - exp) < 2e-6, (off, tilt)
        assert abs(g["off"][0] - off * 1e-3) < 1e-9
        assert abs(math.degrees(g["tilt"][0]) - tilt) < 1e-6

    # offset PERPENDICULAR to the tilt azimuth is cheaper than along it, which
    # is exactly why the ellipse has to be carried around instead of a radius
    th = math.radians(25.0)
    a = np.array([math.sin(th), 0.0, math.cos(th)])
    along = port_geometry(c, n, np.array([0.4e-3, 0, 0]) + a * 0.05, a)
    across = port_geometry(c, n, np.array([0.0, 0.4e-3, 0]) + a * 0.05, a)
    print(f"\n0.4 mm offset at 25 deg: along the tilt "
          f"{along['bite'][0]*1e3:.4f} mm, across {across['bite'][0]*1e3:.4f} mm")
    assert along["bite"][0] > across["bite"][0]

    print("\ntilt budget table")
    for d in (0, 5, 10, 13.3, 18.3, 20, 25, 30, 33.56):
        print(f"  {d:6.2f} deg -> {margin_at_tilt(math.radians(d))*1e3:6.3f} mm")
    for m in (0.0, 0.0001, 0.00015, 0.00025, 0.0003):
        print(f"  margin {m*1e3:.2f} mm -> max tilt "
              f"{math.degrees(max_tilt_rad(m)):6.2f} deg")

    # cone clamp: a command 45 deg off the normal comes back to exactly the cap
    p = c + 0.1 * np.array([math.sin(math.radians(45)), 0.0,
                            math.cos(math.radians(45))])
    q, hit = cone_clamp(p, c, n, math.radians(25.0))
    ang = math.degrees(math.acos(np.dot(q - c, n) / np.linalg.norm(q - c)))
    print(f"\ncone clamp 45 -> {ang:.3f} deg (cap 25), depth "
          f"{np.linalg.norm(q-c)*1e3:.2f} mm (was 100.00), clamped={hit}")
    assert abs(ang - 25.0) < 1e-6 and abs(np.linalg.norm(q - c) - 0.1) < 1e-12
    q2, hit2 = cone_clamp(p, c, n, math.radians(60.0))
    assert not hit2 and np.allclose(q2, p)
    print("\nall trocar geometry checks passed")
