"""Provenance checks and FEM-node appearance transfer for Newton renders."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def loop_subdivide_surface(vertices, faces, levels=1):
    """Create a smooth render-only Loop surface embedded in FEM rest nodes.

    Every generated render vertex is a sparse linear stencil of the original
    surface vertices. Applying the same stencil to the current FEM particles
    makes the denser appearance surface follow deformation without adding FEM
    degrees of freedom or changing contact geometry.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(vertices, np.float64)
    triangles = np.asarray(faces, np.int64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4 or
            triangles.ndim != 2 or triangles.shape[1:] != (3,) or
            len(triangles) == 0):
        raise ValueError("surface requires vertices (N,3) and triangles (M,3)")
    if not np.isfinite(points).all() or not np.isfinite(levels):
        raise ValueError("surface coordinates and subdivision level must be finite")
    if isinstance(levels, bool) or int(levels) != levels or not 0 <= levels <= 3:
        raise ValueError("Loop subdivision levels must be an integer in [0, 3]")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("surface face index is outside the vertex array")
    if np.any(np.sort(triangles, axis=1)[:, 1:] ==
              np.sort(triangles, axis=1)[:, :-1]):
        raise ValueError("surface contains a degenerate triangle")

    # Sparse stencils are dicts over the original FEM-linked render vertices.
    stencils = [{i: 1.0} for i in range(len(points))]
    for _ in range(int(levels)):
        neighbors = [set() for _ in stencils]
        edge_opposites = {}
        for a, b, c in triangles.tolist():
            for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
                key = (u, v) if u < v else (v, u)
                edge_opposites.setdefault(key, []).append(opposite)
                neighbors[u].add(v)
                neighbors[v].add(u)
        if any(len(opposite) != 2 for opposite in edge_opposites.values()):
            raise ValueError("Loop subdivision requires a closed 2-manifold surface")
        if any(not adjacent for adjacent in neighbors):
            raise ValueError("surface contains an isolated vertex")

        def combine(weighted_stencils):
            result = {}
            for stencil, scale in weighted_stencils:
                for index, weight in stencil.items():
                    result[index] = result.get(index, 0.0) + scale * weight
            return {index: weight for index, weight in result.items()
                    if abs(weight) > 1e-14}

        updated = []
        for vertex, adjacent in enumerate(neighbors):
            valence = len(adjacent)
            beta = 3.0 / 16.0 if valence == 3 else 3.0 / (8.0 * valence)
            updated.append(combine(
                [(stencils[vertex], 1.0 - valence * beta)] +
                [(stencils[n], beta) for n in sorted(adjacent)]))

        edge_indices = {}
        for edge, opposites in sorted(edge_opposites.items()):
            u, v = edge
            edge_indices[edge] = len(updated)
            updated.append(combine(
                [(stencils[u], 3.0 / 8.0), (stencils[v], 3.0 / 8.0),
                 (stencils[opposites[0]], 1.0 / 8.0),
                 (stencils[opposites[1]], 1.0 / 8.0)]))

        refined = []
        for a, b, c in triangles.tolist():
            ab = edge_indices[(a, b) if a < b else (b, a)]
            bc = edge_indices[(b, c) if b < c else (c, b)]
            ca = edge_indices[(c, a) if c < a else (a, c)]
            refined.extend(((a, ab, ca), (b, bc, ab),
                            (c, ca, bc), (ab, bc, ca)))
        triangles = np.asarray(refined, np.int32)
        stencils = updated

    width = max(len(stencil) for stencil in stencils)
    indices = np.zeros((len(stencils), width), np.int32)
    weights = np.zeros((len(stencils), width), np.float64)
    for row, stencil in enumerate(stencils):
        total = float(sum(stencil.values()))
        if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise RuntimeError("Loop stencil does not preserve affine coordinates")
        for column, (index, weight) in enumerate(sorted(stencil.items())):
            indices[row, column] = index
            weights[row, column] = weight
    refined_points = np.einsum("nk,nkj->nj", weights,
                               points[indices], optimize=True)
    _, nearest = cKDTree(points).query(refined_points, k=1)
    displacement = np.linalg.norm(refined_points - points[nearest], axis=1)
    report = {
        "subdivision_levels": int(levels),
        "source_vertices": int(len(points)),
        "source_faces": int(len(faces)),
        "render_vertices": int(len(refined_points)),
        "render_faces": int(len(triangles)),
        "maximum_stencil_terms": int(width),
        "p95_rest_smoothing_displacement_m": float(np.quantile(displacement, .95)),
        "max_rest_smoothing_displacement_m": float(displacement.max()),
        "simulation_geometry_changed": False,
        "contact_geometry_changed": False,
        "binding": "fixed sparse Loop stencils over source render/FEM-linked vertices",
    }
    return (refined_points.astype(np.float64), triangles.astype(np.int32),
            indices, weights.astype(np.float32), report)


def bind_surface_vertices_to_fem_nodes(surface_rest_vertices, fem_rest_vertices,
                                       max_error_m=1.0e-5):
    """Bind a captured render surface to its matching FEM rest nodes.

    The CT-derived PLY surfaces in this repository reuse FEM node positions.
    This strict nearest-node binding lets the renderer draw their authored
    surface topology while following the live FEM without changing physics.
    It refuses ambiguous duplicate bindings or a geometric mismatch rather
    than silently stretching a render mesh onto another anatomy.
    """
    from scipy.spatial import cKDTree

    surface = np.asarray(surface_rest_vertices, np.float64)
    fem = np.asarray(fem_rest_vertices, np.float64)
    if (surface.ndim != 2 or fem.ndim != 2 or surface.shape[1:] != (3,) or
            fem.shape[1:] != (3,) or len(surface) == 0 or len(fem) == 0):
        raise ValueError("render and FEM rest vertices must be nonempty (N, 3) arrays")
    if not np.isfinite(surface).all() or not np.isfinite(fem).all():
        raise ValueError("render or FEM rest vertices contain non-finite values")
    if not np.isfinite(max_error_m) or max_error_m < 0.0:
        raise ValueError("max_error_m must be finite and nonnegative")

    error, indices = cKDTree(fem).query(surface, k=1)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("render surface has ambiguous duplicate FEM-node bindings")
    if float(error.max()) > float(max_error_m):
        raise ValueError(
            f"render surface does not match FEM rest nodes: max error "
            f"{float(error.max()):.9g} m exceeds {float(max_error_m):.9g} m")
    return indices.astype(np.int32), {
        "render_vertices": int(len(surface)),
        "fem_nodes": int(len(fem)),
        "max_rest_error_m": float(error.max()),
        "p95_rest_error_m": float(np.quantile(error, 0.95)),
        "unique_binding": True,
    }


def appendix_texture_vertex_colors(rest_vertices, texture_path, base_colors=None,
                                   mix=1.0):
    """Sample a video-derived RGB candidate onto appendix FEM rest vertices.

    A deterministic cylindrical parameterisation is built from the appendix's
    principal axis. The resulting per-node colors stay attached to FEM
    particles as they deform. This does not register pixels to patient-space
    geometry or recover intrinsic albedo/BRDF.
    """
    from PIL import Image

    pts = np.asarray(rest_vertices, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("appendix rest vertices must be an (N>=4, 3) array")
    if not np.isfinite(pts).all():
        raise ValueError("appendix rest vertices contain non-finite values")
    if not np.isfinite(mix) or not 0.0 <= float(mix) <= 1.0:
        raise ValueError("texture mix must be finite and in [0, 1]")

    image = np.asarray(Image.open(texture_path).convert("RGB"), np.float64) / 255.0
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("appendix texture must be at least 2 x 2 pixels")
    image = np.where(image <= 0.04045, image / 12.92,
                     ((image + 0.055) / 1.055) ** 2.4)

    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    axis = vh[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    axis /= np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    radial_u = up - np.dot(up, axis) * axis
    if np.linalg.norm(radial_u) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        radial_u = fallback - np.dot(fallback, axis) * axis
    radial_u /= np.linalg.norm(radial_u)
    radial_v = np.cross(axis, radial_u)

    rel = pts - center
    axial = rel @ axis
    axial_span = float(axial.max() - axial.min())
    if axial_span <= 1e-10:
        raise ValueError("appendix rest vertices have no longitudinal extent")
    v = np.clip((axial - axial.min()) / axial_span, 0.0, 1.0)
    theta = np.arctan2(rel @ radial_v, rel @ radial_u)
    u = np.mod((theta + np.pi) / (2.0 * np.pi), 1.0)

    # Bilinear sampling, wrapping around the tube and clamping at its ends.
    h, w = image.shape[:2]
    xf = u * w
    x0 = np.floor(xf).astype(np.int64) % w
    x1 = (x0 + 1) % w
    yf = (1.0 - v) * (h - 1)
    y0 = np.floor(yf).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xf - np.floor(xf))[:, None]
    fy = (yf - y0)[:, None]
    sampled = ((1.0 - fy) * ((1.0 - fx) * image[y0, x0] + fx * image[y0, x1])
               + fy * ((1.0 - fx) * image[y1, x0] + fx * image[y1, x1]))
    if base_colors is not None:
        base = np.asarray(base_colors, np.float64).reshape(-1, 3)
        if base.shape != sampled.shape or not np.isfinite(base).all():
            raise ValueError("base vertex colours must match the appendix nodes")
        sampled = (1.0 - float(mix)) * base + float(mix) * sampled
    return sampled.astype(np.float32)


def appendix_texture_uv(rest_vertices):
    """Stable cylindrical UVs for a tube-like appendix rest surface.

    UVs are bound to rest FEM nodes. This is an appearance projection, not
    camera-to-mesh registration.
    """
    pts = np.asarray(rest_vertices, np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError("appendix rest vertices must be an (N>=4, 3) array")
    if not np.isfinite(pts).all():
        raise ValueError("appendix rest vertices contain non-finite values")
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    axis = vh[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
    axis /= np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    radial_u = up - np.dot(up, axis) * axis
    if np.linalg.norm(radial_u) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        radial_u = fallback - np.dot(fallback, axis) * axis
    radial_u /= np.linalg.norm(radial_u)
    radial_v = np.cross(axis, radial_u)
    rel = pts - center
    axial = rel @ axis
    span = float(axial.max() - axial.min())
    if span <= 1e-10:
        raise ValueError("appendix rest vertices have no longitudinal extent")
    u = np.mod((np.arctan2(rel @ radial_v, rel @ radial_u) + np.pi) /
               (2.0 * np.pi), 1.0)
    v = (axial - axial.min()) / span
    return np.stack([u, v], axis=1).astype(np.float32)


def appendix300_texture_provenance(texture_path):
    """Require an adjacent Appendix300 report whose recorded hash matches."""
    import hashlib
    import json

    path = Path(texture_path).expanduser()
    report_path = path.parent / "appearance.json"
    if not path.is_file() or not report_path.is_file():
        raise ValueError("Appendix300 texture requires a sibling appearance.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (str(report.get("dataset", "")).casefold() != "appendix300" or
            report.get("license") != "CC BY 4.0"):
        raise ValueError("appearance.json must identify licensed Appendix300 data")
    doi = report.get("source_doi", report.get("doi"))
    if doi != "10.25532/OPARA-1173":
        raise ValueError("appearance.json does not identify the Appendix300 DOI")
    expected = None
    for key in ("rois_and_textures", "rois", "texture"):
        section = report.get(key) or {}
        if not isinstance(section, dict):
            continue
        data = section if key == "texture" else (section.get("appendix") or {})
        expected = data.get("texture_sha256")
        if expected:
            break
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError("Appendix300 texture hash does not match appearance.json")
    mask = report.get("mask") or {}
    return {
        "dataset": "Appendix300",
        "license": report["license"],
        "attribution": report.get(
            "attribution", "Appendix300, DOI 10.25532/OPARA-1173, CC BY 4.0"),
        "source_doi": doi,
        "source_video": report.get("source_video"),
        "source_video_sha256": report.get("source_video_sha256"),
        "source_frame_sha256": report.get("source_frame_sha256"),
        "source_mask_sha256": mask.get("mask_sha256"),
        "source_status": report.get("status"),
        "source_texture_sha256": actual,
        "provenance_report_sha256": hashlib.sha256(
            report_path.read_bytes()).hexdigest(),
    }
