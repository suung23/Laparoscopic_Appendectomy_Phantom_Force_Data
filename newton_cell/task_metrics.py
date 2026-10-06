"""Engine-independent geometric metrics shared by the Newton task path.

These observations describe grasp location and whether appendix vertices lie
between a cutter's blades. They do not assert cutting or tissue severance.
"""
from __future__ import annotations

import numpy as np


JAW_BLADE_LEN = 0.021
JAW_HALF_WIDTH = 0.0025
JAW_SKIN = 0.0015


def signed_tet_volumes(points, tet_indices):
    """Signed tetrahedron volumes in m^3 for four-index cells."""
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(tet_indices, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 4:
        raise ValueError("Expected Nx3 points and Mx4 tetrahedron indices")
    if not np.isfinite(t).all() or (t < 0).any() or (t >= len(p)).any():
        raise ValueError("Tetrahedron indices are invalid for the point array")
    v = p[t]
    return np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0],
                                           v[:, 2] - v[:, 0]),
                     v[:, 3] - v[:, 0]) / 6.0


def deformation_health(points, velocities, rest_points, tet_indices):
    """Summarize finite state, signed-volume health, and nodal speed."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    rest = np.asarray(rest_points, dtype=np.float64)
    tets = np.asarray(tet_indices, dtype=np.int64)
    shape_ok = (p.ndim == 2 and p.shape[1:] == (3,) and
                v.shape == p.shape and rest.shape == p.shape and
                tets.ndim == 2 and tets.shape[1:] == (4,))
    empty = {
        "physics_state_finite": False,
        "tissue_min_signed_volume_ratio": float("nan"),
        "tissue_inverted_tets": -1,
        "tissue_nonfinite_tets": -1,
        "tissue_max_speed_m_s": float("nan"),
        "tissue_worst_tet_index": -1,
    }
    if not shape_ok or not len(p) or not len(tets):
        return empty
    finite_nodes = np.isfinite(p).all(axis=1) & np.isfinite(v).all(axis=1)
    valid_indices = (tets >= 0).all(axis=1) & (tets < len(p)).all(axis=1)
    valid_tet_ids = np.flatnonzero(valid_indices)
    safe_tets = tets[valid_indices]
    rest_vol = signed_tet_volumes(rest, safe_tets)
    finite_tets = finite_nodes[safe_tets].all(axis=1)
    current_vol = np.full(len(safe_tets), np.nan)
    if finite_tets.any():
        current_vol[finite_tets] = signed_tet_volumes(
            p, safe_tets[finite_tets])
    usable = np.isfinite(rest_vol) & (np.abs(rest_vol) > 1e-15)
    ratio = current_vol[usable] / rest_vol[usable]
    finite_ratio = np.isfinite(ratio)
    tet_ids = valid_tet_ids[usable][finite_ratio]
    inverted = int(np.sum(ratio[finite_ratio] <= 0.0))
    nonfinite = int(len(tets) - finite_ratio.sum())
    speeds = np.linalg.norm(v[finite_nodes], axis=1)
    result = {
        "physics_state_finite": bool(finite_nodes.all() and nonfinite == 0),
        "tissue_min_signed_volume_ratio": (
            float(ratio[finite_ratio].min()) if finite_ratio.any() else float("nan")),
        "tissue_inverted_tets": inverted,
        "tissue_nonfinite_tets": nonfinite,
        "tissue_max_speed_m_s": float(speeds.max()) if len(speeds) else float("nan"),
        "tissue_worst_tet_index": (
            int(tet_ids[int(np.argmin(ratio[finite_ratio]))])
            if finite_ratio.any() else -1),
    }
    return result


def partition_tets_by_spans(tet_indices, organ_spans):
    """Return local tet indices grouped by contiguous tissue particle spans.

    Newton's scene builder appends each organ's particles and tets in order.
    Keeping a per-organ health view lets a failed global integrity gate identify
    which reconstructed mesh caused it, instead of merely reporting a total.
    Cross-organ cells are returned under ``__unassigned__`` and remain covered
    by the global integrity check.
    """
    tets = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    owner = np.full(len(tets), -1, np.int32)
    result = {}
    for organ_i, (name, span) in enumerate(organ_spans.items()):
        start, end = map(int, span)
        mask = ((tets >= start) & (tets < end)).all(axis=1)
        result[name] = (start, end, tets[mask] - start)
        owner[mask] = organ_i
    other = owner < 0
    if other.any():
        result["__unassigned__"] = (0, max((int(s[1]) for s in
                                               organ_spans.values()), default=0),
                                    tets[other])
    return result


def deformation_health_by_organ(points, velocities, rest_points, groups):
    """Signed-volume health for each organ using prepartitioned local tets."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    rest = np.asarray(rest_points, dtype=np.float64)
    return {
        name: deformation_health(p[start:end], v[start:end], rest[start:end], local_tets)
        for name, (start, end, local_tets) in groups.items()
    }


def tissue_probe(points, p_tip, shaft_axis, flange_rotation, grip_half,
                 jaw_pin_axis_f):
    """Mirror the Isaac jaw-mouth test using current tissue vertices."""
    if points is None:
        return False, float("nan"), None
    points = np.asarray(points, dtype=np.float64)
    p_tip = np.asarray(p_tip, dtype=np.float64)
    shaft_axis = np.asarray(shaft_axis, dtype=np.float64)
    flange_rotation = np.asarray(flange_rotation, dtype=np.float64)
    jaw_pin_axis_f = np.asarray(jaw_pin_axis_f, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
            p_tip.shape != (3,) or shaft_axis.shape != (3,) or
            flange_rotation.shape != (3, 3) or jaw_pin_axis_f.shape != (3,) or
            not np.isfinite(points).all() or not np.isfinite(p_tip).all() or
            not np.isfinite(shaft_axis).all() or
            not np.isfinite(flange_rotation).all() or
            not np.isfinite(jaw_pin_axis_f).all() or not np.isfinite(grip_half)):
        return False, float("nan"), None
    axis_norm = np.linalg.norm(shaft_axis)
    if axis_norm < 1e-9:
        return False, float("nan"), None
    a = shaft_axis / axis_norm
    w = points - p_tip
    along = w @ (-a)
    n = flange_rotation @ jaw_pin_axis_f
    n = n - a * float(n @ a)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return False, float("nan"), None
    n /= nn
    u = np.cross(a, n)
    s_u, s_n = w @ u, w @ n
    half_gap = (JAW_BLADE_LEN - along) * np.sin(max(float(grip_half), 0.0))
    mouth = ((along > -JAW_SKIN) & (half_gap > JAW_SKIN) &
             (np.abs(s_u) < half_gap + JAW_SKIN) &
             (np.abs(s_n) < JAW_HALF_WIDTH + JAW_SKIN))
    if int(mouth.sum()) < 5:
        return False, float("nan"), None
    su = s_u[mouth]
    if su.min() > JAW_SKIN or su.max() < -JAW_SKIN:
        return False, float("nan"), None
    thickness = float(np.percentile(su, 95) - np.percentile(su, 5))
    return True, thickness, points[mouth].mean(axis=0)


def resection_observations(appendix_vertices, handle_positions, held_index,
                           pin_world, right_tip_world, *, grasp_band=(0.30, 0.70),
                           resect_offset_m=0.010):
    """Describe ordered grasp/resection geometry on the live appendix surface."""
    vertices = np.asarray(appendix_vertices, dtype=np.float64)
    handles = np.asarray(handle_positions, dtype=np.float64)
    pin = np.asarray(pin_world, dtype=np.float64)
    right_tip = np.asarray(right_tip_world, dtype=np.float64)
    count = len(handles) if handles.ndim == 2 else 0
    valid = (vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) and
             handles.ndim == 2 and handles.shape[1] == 3 and count > 0 and
             np.isfinite(vertices).all() and np.isfinite(handles).all() and
             pin.shape == (3,) and right_tip.shape == (3,) and
             np.isfinite(pin).all() and np.isfinite(right_tip).all())
    held = int(held_index) if held_index is not None else -1
    valid_held = valid and 0 <= held < count
    gfrac = (held + 0.5) / count if valid_held else float("nan")
    lo, hi = map(float, grasp_band)
    out = {
        "n_grip_clusters": int(count),
        "grasp_index": int(held if valid_held else -1),
        "grasp_frac": float(gfrac),
        "grasp_in_band": bool(valid_held and lo <= gfrac <= hi),
        "grasp_band": np.asarray([lo, hi], dtype=np.float32),
        "resect_seg_n": 0,
        "resect_target_world": np.full(3, np.nan, dtype=np.float32),
        "right_tip_to_resect_m": float("nan"),
        "right_tissue_in_jaw": False,
        "right_jaw_tissue_m": float("nan"),
        "cut_done": False,
        "cut_count": 0,
        "cut_frac": float("nan"),
        "appendix_severed": False,
    }
    if not valid_held:
        return out
    p_held = handles[held]
    axis = pin - p_held
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return out
    direction = axis / length
    t = (vertices - p_held) @ direction
    segment = vertices[(t > 0.0) & (t < length)]
    if len(segment):
        target = p_held + direction * float(resect_offset_m)
        target = segment[int(np.argmin(np.linalg.norm(segment - target, axis=1)))]
        out["resect_seg_n"] = int(len(segment))
        out["resect_target_world"] = target.astype(np.float32)
        out["right_tip_to_resect_m"] = float(
            np.linalg.norm(segment - right_tip, axis=1).min())
    return out
