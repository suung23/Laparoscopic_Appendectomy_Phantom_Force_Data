"""Bake registered multi-view endoscopic observations onto mesh UVs.

This transfers observed RGB only. It is not intrinsic-albedo recovery. Each
view must provide calibrated camera intrinsics/extrinsics, metric camera-Z
depth, and a reviewed target-tissue mask. Pixels are accepted only when the
registered mesh point agrees with depth and is front-facing/visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegisteredView:
    rgb: np.ndarray
    depth_z_m: np.ndarray
    tissue_mask: np.ndarray
    intrinsics: np.ndarray
    camera_from_world: np.ndarray
    name: str = "view"


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0., 1.)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1. / 2.4) - 0.055)


def _validate_view(view: RegisteredView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(view.rgb)
    depth = np.asarray(view.depth_z_m, dtype=np.float64)
    mask = np.asarray(view.tissue_mask, dtype=bool)
    k = np.asarray(view.intrinsics, dtype=np.float64)
    t = np.asarray(view.camera_from_world, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError("RGB, depth, and tissue mask must share HxW dimensions; RGB must have 3 channels")
    if rgb.dtype.kind not in "uif" or not np.isfinite(rgb).all():
        raise ValueError("RGB must contain finite numeric values")
    rgb = rgb.astype(np.float64)
    if rgb.min(initial=0.) < 0.:
        raise ValueError("RGB values must be nonnegative")
    if rgb.max(initial=0.) > 1.:
        if rgb.max() > 255.:
            raise ValueError("RGB values must be in [0, 1] or [0, 255]")
        rgb /= 255.
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Intrinsics must be a finite 3x3 matrix with positive focal lengths")
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("camera_from_world must be a finite 4x4 rigid transform")
    if not np.allclose(t[3], [0., 0., 0., 1.], atol=1e-8):
        raise ValueError("camera_from_world must use homogeneous row [0, 0, 0, 1]")
    rotation = t[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("camera_from_world rotation must be a proper orthonormal rotation")
    return rgb, depth, mask


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax, ay = x - x0, y - y0
    if image.ndim == 2:
        return ((1 - ax) * (1 - ay) * image[y0, x0] +
                ax * (1 - ay) * image[y0, x1] +
                (1 - ax) * ay * image[y1, x0] + ax * ay * image[y1, x1])
    return (((1 - ax) * (1 - ay))[:, None] * image[y0, x0] +
            (ax * (1 - ay))[:, None] * image[y0, x1] +
            ((1 - ax) * ay)[:, None] * image[y1, x0] +
            (ax * ay)[:, None] * image[y1, x1])


def bake_registered_surface(vertices: np.ndarray, faces: np.ndarray, face_uvs: np.ndarray,
                            views: list[RegisteredView], *, atlas_size: int = 1024,
                            depth_tolerance_m: float = 0.003,
                            relative_depth_tolerance: float = 0.01,
                            incidence_power: float = 2.) -> dict:
    """Fuse visible, registered view pixels into face-varying UVs.

    ``camera_from_world`` uses a pinhole camera with +Z forward, +X right,
    +Y down. Depth is metric camera-space Z. ``face_uvs`` has shape Mx3x2.
    Output ``coverage`` is authoritative; uncovered atlas texels stay black
    so downstream material code cannot silently mistake padding for evidence.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if raw_faces.dtype.kind not in "iu" or raw_faces.dtype.kind == "b":
        raise ValueError("faces must contain integer vertex indices")
    faces = raw_faces.astype(np.int64, copy=False)
    face_uvs = np.asarray(face_uvs, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3 world coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces must be nonempty in-range Mx3 triangles")
    if face_uvs.shape != (len(faces), 3, 2) or not np.isfinite(face_uvs).all():
        raise ValueError("face_uvs must be finite Mx3x2 corner coordinates")
    if np.any(face_uvs < -1e-7) or np.any(face_uvs > 1. + 1e-7):
        raise ValueError("UV coordinates must lie in [0, 1]")
    if isinstance(atlas_size, bool) or not isinstance(atlas_size, (int, np.integer)) or atlas_size < 16:
        raise ValueError("atlas_size must be an integer >= 16")
    if not views:
        raise ValueError("At least one calibrated registered view is required")
    if (not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0 or
            not np.isfinite(relative_depth_tolerance) or relative_depth_tolerance < 0 or
            not np.isfinite(incidence_power) or incidence_power < 0):
        raise ValueError("Depth tolerances and incidence power must be finite and nonnegative")
    checked = [(_validate_view(view), view) for view in views]

    accum = np.zeros((atlas_size, atlas_size, 3), dtype=np.float64)
    weights = np.zeros((atlas_size, atlas_size), dtype=np.float64)
    view_counts = np.zeros((atlas_size, atlas_size), dtype=np.uint16)
    per_view_accepted = {view.name: 0 for _, view in checked}

    for tri_ids, tri_uv in zip(faces, face_uvs):
        tri_xyz = vertices[tri_ids]
        cross = np.cross(tri_xyz[1] - tri_xyz[0], tri_xyz[2] - tri_xyz[0])
        normal_len = np.linalg.norm(cross)
        if normal_len <= 1e-12:
            continue
        normal = cross / normal_len
        tri_px = np.column_stack([tri_uv[:, 0] * atlas_size - .5,
                                  (1. - tri_uv[:, 1]) * atlas_size - .5])
        lo = np.maximum(np.ceil(tri_px.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(tri_px.max(axis=0)).astype(int), atlas_size - 1)
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        atlas_xy = np.column_stack([xx.ravel(), yy.ravel()])
        edge1, edge2 = tri_px[1] - tri_px[0], tri_px[2] - tri_px[0]
        det = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        if abs(det) <= 1e-12:
            continue
        delta = atlas_xy - tri_px[0]
        b1 = (delta[:, 0] * edge2[1] - delta[:, 1] * edge2[0]) / det
        b2 = (edge1[0] * delta[:, 1] - edge1[1] * delta[:, 0]) / det
        bary = np.column_stack([1. - b1 - b2, b1, b2])
        inside = np.all(bary >= -1e-7, axis=1)
        if not inside.any():
            continue
        atlas_xy, bary = atlas_xy[inside], bary[inside]
        points = bary @ tri_xyz
        flat_indices = atlas_xy[:, 1] * atlas_size + atlas_xy[:, 0]

        for (rgb, depth, mask), view in checked:
            t = np.asarray(view.camera_from_world, dtype=np.float64)
            k = np.asarray(view.intrinsics, dtype=np.float64)
            xyz_cam = points @ t[:3, :3].T + t[:3, 3]
            z = xyz_cam[:, 2]
            positive = z > 1e-6
            if not positive.any():
                continue
            projected = xyz_cam[positive] @ k.T
            px = projected[:, 0] / projected[:, 2]
            py = projected[:, 1] / projected[:, 2]
            good_ids = np.flatnonzero(positive)
            h, w = depth.shape
            in_frame = (px >= 0.) & (py >= 0.) & (px <= w - 1.) & (py <= h - 1.)
            if not in_frame.any():
                continue
            good_ids, px, py = good_ids[in_frame], px[in_frame], py[in_frame]
            ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
            depth_obs = depth[iy, ix]
            mask_ok = mask[iy, ix]
            tol = depth_tolerance_m + relative_depth_tolerance * z[good_ids]
            depth_ok = np.isfinite(depth_obs) & (depth_obs > 0.) & (np.abs(depth_obs - z[good_ids]) <= tol)
            camera_center = -t[:3, :3].T @ t[:3, 3]
            to_camera = camera_center - points[good_ids]
            to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
            incidence = np.maximum(0., to_camera @ normal)
            valid = mask_ok & depth_ok & (incidence > 0.)
            if not valid.any():
                continue
            selected_ids = good_ids[valid]
            sampled = _sample_bilinear(rgb, px[valid], py[valid])
            sampled_linear = _srgb_to_linear(sampled)
            sample_weights = incidence[valid] ** incidence_power
            ys, xs = atlas_xy[selected_ids, 1], atlas_xy[selected_ids, 0]
            np.add.at(accum, (ys, xs), sampled_linear * sample_weights[:, None])
            np.add.at(weights, (ys, xs), sample_weights)
            np.add.at(view_counts, (ys, xs), 1)
            per_view_accepted[view.name] += int(valid.sum())

    coverage = weights > 0.
    atlas = np.zeros_like(accum, dtype=np.float32)
    atlas[coverage] = _linear_to_srgb(accum[coverage] / weights[coverage, None]).astype(np.float32)
    return {
        "rgb": atlas,
        "coverage": coverage,
        "contributing_views": view_counts,
        "weight_sum": weights.astype(np.float32),
        "report": {
            "status": "registered_observation_bake_complete" if coverage.any() else "no_registered_texels",
            "atlas_size": int(atlas_size),
            "covered_texels": int(coverage.sum()),
            "coverage_fraction": float(coverage.mean()),
            "multi_view_texel_fraction": float(np.mean(view_counts > 1)),
            "accepted_projection_count_by_view": per_view_accepted,
            "depth_tolerance_m": float(depth_tolerance_m),
            "relative_depth_tolerance": float(relative_depth_tolerance),
            "incidence_weight_power": float(incidence_power),
            "source_semantics": "registered observed RGB; not intrinsic albedo, BRDF, or clinical anatomy confirmation",
            "required_inputs": ["reviewed tissue masks", "camera calibration", "metric depth", "mesh-to-camera registration"],
        },
    }
