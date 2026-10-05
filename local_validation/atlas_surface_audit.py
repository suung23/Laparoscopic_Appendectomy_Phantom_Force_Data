"""Audit atlas winding and normal chords; never infer anatomical wall thickness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom


def uv_seams(usd):
    primvar = UsdGeom.PrimvarsAPI(usd).GetPrimvar('st')
    if not primvar or not primvar.HasValue():
        return None
    if primvar.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise ValueError('Expected face-varying UVs')
    uv = np.asarray(primvar.ComputeFlattened()).reshape(-1, 3, 2)
    faces = np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    edges, gaps = {}, []
    for face, coords in zip(faces, uv):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(face[a]), int(face[b]))))
            pair = coords[[a, b]] if face[a] < face[b] else coords[[b, a]]
            if key in edges:
                gaps.append(float(np.max(np.linalg.norm(pair - edges[key], axis=1))))
            else:
                edges[key] = pair
    return dict(shared_edges=len(gaps), discontinuous_uv_edges=int(np.sum(np.asarray(gaps) > 1e-5)),
        interpretation='UV seams are permitted, but unrelated image samples across them can produce visible colour discontinuities. This is not proof of a shading defect.')


def normal_chords(mesh, faces):
    origins = mesh.triangles_center[faces]
    directions = -mesh.face_normals[faces]
    locations, rays, hit_faces = mesh.ray.intersects_location(
        origins, directions, multiple_hits=True)
    distances = np.einsum('ij,ij->i', locations - origins[rays], directions[rays])
    # Starting on the face avoids an epsilon that can cross a thin surface.
    # Trimesh can return the emitting face: explicitly exclude it and zero hits.
    valid = (hit_faces != faces[rays]) & (distances > 1e-8)
    result = np.full(len(faces), np.inf)
    np.minimum.at(result, rays[valid], distances[valid])
    return result


def audit(scene):
    stage = Usd.Stage.Open(str(scene))
    records = {}
    for name in ('appendix', 'mesoappendix', 'appendicular_artery'):
        usd = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Atlas/' + name + '/render'))
        if not np.all(np.asarray(usd.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError('Expected triangular atlas surface')
        mesh = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError('Closed consistently outward atlas surface required: ' + name)
        faces = np.random.default_rng(47).choice(len(mesh.faces),
            size=min(1024, len(mesh.faces)), replace=False)
        chords = normal_chords(mesh, faces)
        finite = chords[np.isfinite(chords)]
        normals = np.asarray(usd.GetNormalsAttr().Get())
        records[name] = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
            watertight=True, winding_consistent=True, signed_volume_ml=mesh.volume * 1e6,
            normal_interpolation=str(usd.GetNormalsInterpolation()),
            normal_agreement_min=float(np.min(np.sum(normals * mesh.vertex_normals, axis=1))),
            requested_rays=len(faces), valid_rays=len(finite),
            uv_seams=uv_seams(usd),
            inward_chord_mm_p05_p50_p95=(np.quantile(finite, [.05,.5,.95])*1000).tolist())
    return dict(scene=str(scene).replace('\\', '/'),
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(), seed=47,
        sampling='Uniform faces, not uniform surface area',
        correction='Excludes emitting face and zero-distance intersections; supersedes epsilon-offset audit',
        scope='Local-coordinate normal chords, not anatomical wall thickness. Curvature and folds affect chords. No patient calibration.',
        meshes=records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=Path('out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda'))
    p.add_argument('--out', type=Path, default=Path('validation/atlas-local-thickness-audit.json'))
    args = p.parse_args()
    # Analytic regression: every inward face-normal chord of a cube is its width.
    cube = trimesh.creation.box(extents=[.01]*3)
    np.testing.assert_allclose(normal_chords(cube, np.arange(len(cube.faces))), .01, atol=1e-10)
    result = audit(args.scene)
    result['analytic_cube_regression_passed'] = True
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
