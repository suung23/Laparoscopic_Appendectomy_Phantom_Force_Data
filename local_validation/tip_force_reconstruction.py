"""Tip force from the proximal (extracorporeal) bending station.

The instrument carries ONE strain-gauge station, outside the trocar, measuring
bending in two planes. The tip force is not read directly: the trocar sits
between the gauge and the tip and carries a reaction, so the gauge sees the tip
load levered back through that support. Recovering the tip force needs the
insertion depth, which the MCU supplies per sample.

Geometry, measured from the proximal support point (the handle, which by design
transmits no moment):

    support ──── gauge ──────── trocar ──── tip
            b              a           d          L = b + a + d

With a moment-free support the bending moment is zero there, which anchors the
linear moment profile and fixes the shear. Then

    M_gauge = F_tip * b * d / L'        where L' = b + a = L - d
    F_tip   = c * M_gauge * (L - d) / d,     c = 1/b

`c` and `L` are fixed by construction but are NOT measured physically -- a
calibration sweep recovers both, because

    F*d / M = c*L - c*d

is a straight line in d: slope -c, intercept c*L. The same line validates the
whole model, including the moment-free support assumption:

    straight through the expected form  -> model holds
    curvature                           -> the trocar is applying a couple
    scatter                             -> the support point is not reproducible

Sensitivity rises with insertion depth (S = M/F = b*d/(L-d) is monotone in d),
so the detection limit is depth-dependent and every estimate carries one.

What this module does NOT do: axial (thrust) force. The half-bridges difference
opposing gauges, which cancels the common-mode axial strain by construction.
Only the two lateral bending components exist. Loads applied along the shaft
axis are invisible.
"""
from __future__ import annotations

import numpy as np

# Trocar-station block of the legacy decoupling matrix, recovered from the
# 261021 synced files. It references ONLY the two live channels (btx, bty), so
# it carries no noise from the four removed gauges. The cross-axis shape is a
# property of gauge placement and transfers; the SCALE was fitted for a
# different load geometry and must be refitted -- which `fit_geometry` does,
# since `c` absorbs any constant scale.
TROCAR_DECOUPLE = np.array([[-2.3237, 0.3471],
                            [0.7880, 2.9510]])
TROCAR_OFFSET = np.array([0.083, -0.160])


def bending_from_raw(btx, bty, matrix=None, offset=None) -> np.ndarray:
    """(N,2) decoupled bending in the two gauge planes, arbitrary scale."""
    m = TROCAR_DECOUPLE if matrix is None else np.asarray(matrix, float)
    o = TROCAR_OFFSET if offset is None else np.asarray(offset, float)
    raw = np.column_stack([np.asarray(btx, float), np.asarray(bty, float)])
    return raw @ m.T + o


def sensitivity(d_mm, c: float, length_mm: float) -> np.ndarray:
    """S(d) = M/F in the module's moment units. Monotone increasing in d."""
    d = np.asarray(d_mm, float)
    out = np.full(d.shape, np.nan)
    ok = np.isfinite(d) & (d > 0.0) & (d < length_mm)
    out[ok] = d[ok] / (c * (length_mm - d[ok]))
    return out


def tip_force(bending_xy, d_mm, c: float, length_mm: float) -> np.ndarray:
    """(N,2) lateral tip force in the gauge planes.

    Samples with d outside (0, L) come back NaN rather than silently wrong:
    d >= L would put the trocar at or behind the support, which is not a
    configuration this model describes.
    """
    m = np.asarray(bending_xy, float)
    d = np.asarray(d_mm, float)
    gain = np.full(d.shape, np.nan)
    ok = np.isfinite(d) & (d > 0.0) & (d < length_mm)
    gain[ok] = c * (length_mm - d[ok]) / d[ok]
    return m * gain[:, None]


def detection_limit(sigma_moment: float, d_mm, c: float,
                    length_mm: float) -> np.ndarray:
    """Smallest resolvable tip force at each depth, same chain as `tip_force`."""
    d = np.asarray(d_mm, float)
    out = np.full(d.shape, np.inf)
    ok = np.isfinite(d) & (d > 0.0) & (d < length_mm)
    out[ok] = sigma_moment * c * (length_mm - d[ok]) / d[ok]
    return out


def fit_geometry(bending_mag, d_mm, force_known) -> dict:
    """Recover (c, L) from a calibration sweep, and judge the model.

    Apply a KNOWN lateral tip load at several insertion depths, then fit

        y = F*d/M = c*L - c*d

    Needs at least two distinct depths; four or more make the linearity
    judgement meaningful. Load the x plane deliberately -- with only two axes it
    is half the information, and an earlier campaign left cross-validation at
    R^2 = 0.62 because x was under-loaded.
    """
    m = np.asarray(bending_mag, float)
    d = np.asarray(d_mm, float)
    f = np.asarray(force_known, float)
    ok = np.isfinite(m) & np.isfinite(d) & np.isfinite(f) & (np.abs(m) > 0)
    if ok.sum() < 2 or np.unique(np.round(d[ok], 3)).size < 2:
        raise ValueError("need loaded samples at two or more distinct depths")
    y = f[ok] * d[ok] / m[ok]
    slope, intercept = np.polyfit(d[ok], y, 1)
    if slope >= 0:
        raise ValueError("non-negative slope: F*d/M must fall with depth; "
                         "check sign conventions or the load direction")
    c = float(-slope)
    length = float(intercept / c)
    pred = slope * d[ok] + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # curvature test: a quadratic term that the straight line cannot absorb
    quad = np.polyfit(d[ok], y, 2) if ok.sum() >= 3 else [np.nan] * 3
    return {
        "c_per_mm": c,
        "length_support_to_tip_mm": length,
        "implied_support_to_gauge_mm": 1.0 / c,
        "r2": r2,
        "n_points": int(ok.sum()),
        "distinct_depths": int(np.unique(np.round(d[ok], 3)).size),
        "depth_range_mm": [float(d[ok].min()), float(d[ok].max())],
        "quadratic_coefficient": float(quad[0]),
        "verdict": _verdict(r2, length),
    }


def _verdict(r2: float, length_mm: float) -> str:
    if not np.isfinite(r2):
        return "indeterminate"
    if r2 < 0.95:
        return ("poor linearity: the trocar may be applying a couple, or the "
                "support point is not reproducible")
    if not (0.0 < length_mm < 2000.0):
        return "fitted length is not physical; check units (mm) and load sign"
    return "linear as the model predicts"


def project_to_world(force_xy, quat_xyzw) -> np.ndarray:
    """Lift the two in-plane components into world coordinates using the tool
    attitude. The gauge planes are fixed to the shaft, so a tip force expressed
    there means nothing until the shaft orientation is applied."""
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.where(n > 0, n, 1.0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    f = np.column_stack([np.asarray(force_xy, float),
                         np.zeros(len(force_xy))])
    return np.einsum("nij,nj->ni", r, f)


def noise_floor(bending_xy, rate_hz: float = 100.0,
                windows_s=(0.1, 0.5, 1.0, 2.0)) -> dict:
    """Characterise the measurement floor from an unloaded recording.

    Reports the moment-unit standard deviation after block averaging, and how
    closely it tracks 1/sqrt(n). Tracking it all the way means the average is
    converging on zero -- there is no signal underneath, which is the honest
    reading of an unloaded session and the right baseline for the next one.
    """
    m = np.asarray(bending_xy, float)
    out = {"n_samples": int(len(m)), "per_axis": {}}
    for ax, name in enumerate(("plane_x", "plane_y")):
        s0 = float(np.std(m[:, ax]))
        rows = []
        for ws in windows_s:
            n = max(1, int(round(ws * rate_hz)))
            if n >= len(m):
                continue
            k = np.ones(n) / n
            s = float(np.std(np.convolve(m[:, ax], k, "valid")))
            rows.append({"window_s": ws, "samples": n, "std": s,
                         "ideal_white": s0 / np.sqrt(n),
                         "ratio_measured_over_white": s / (s0 / np.sqrt(n))})
        out["per_axis"][name] = {"std_single_sample": s0, "averaging": rows}
    mag = np.hypot(m[:, 0], m[:, 1])
    out["magnitude_mean"] = float(mag.mean())
    out["magnitude_max"] = float(mag.max())
    return out
