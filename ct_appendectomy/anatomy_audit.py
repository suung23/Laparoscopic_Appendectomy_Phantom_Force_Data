"""Mesh overlap diagnostics using deterministic ray parity, without native rtree."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom


def load_meshes(path):
    stage = Usd.Stage.Open(str(path))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Readable metre-scale USD required')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        if not np.all(np.asarray(geom.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError('Triangle geometry required')
        vertices = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3]
        faces = np.asarray(geom.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        meshes[str(prim.GetPath())] = trimesh.Trimesh(vertices, faces, process=False)
    return meshes


def contains(mesh, points):
    # Two non-axis-aligned parity rays; disagreement is explicitly reported.
    points = np.asarray(points)
    results = []
    a = mesh.triangles[:, 0]
    edge1 = mesh.triangles[:, 1] - a
    edge2 = mesh.triangles[:, 2] - a
    for direction in (np.array([1., .371, .529]), np.array([.213, 1., .417])):
        direction /= np.linalg.norm(direction)
        p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        det = np.einsum('ij,ij->i', edge1, p)
        valid = np.abs(det) > 1e-14
        inverse = np.divide(1., det, out=np.zeros_like(det), where=valid)
        parity = []
        for point in points:
            if np.any(point < mesh.bounds[0]) or np.any(point > mesh.bounds[1]):
                parity.append(False)
                continue
            delta = point-a
            u = np.einsum('ij,ij->i', delta, p)*inverse
            q = np.cross(delta, edge1)
            v = (q @ direction)*inverse
            t = np.einsum('ij,ij->i', edge2, q)*inverse
            hits = valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-9)
            parity.append(np.count_nonzero(hits) % 2 == 1)
        results.append(np.array(parity))
    return results[0] & results[1], results[0] != results[1]


def overlap_report(path, samples=128):
    meshes = load_meshes(path)
    manifest=json.loads((Path(path).parent/'scene_manifest.json').read_text(encoding='utf-8'))
    anchor=np.asarray(manifest['ct_anchor_m'])
    report = {'schema_version': 1, 'scene_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'method': 'Deterministic vertex samples, two-ray parity, not volume overlap',
              'registration_validated': False, 'structures': {}}
    for name, mesh in meshes.items():
        if not name.startswith('/World/Atlas/'):
            continue
        selected = np.unique(np.linspace(0, len(mesh.vertices)-1, min(samples, len(mesh.vertices)), dtype=int))
        points = mesh.vertices[selected]
        distal=np.linalg.norm(points-anchor,axis=1)>.004
        checks = {}
        for other, target in meshes.items():
            if other.startswith('/World/CT/'):
                inside, uncertain = contains(target, points)
                checks[other] = {'inside_sample_fraction': float(inside.mean()),
                                 'distal_inside_sample_fraction':float(inside[distal].mean()) if np.any(distal) else None,
                                 'parity_disagreement_fraction': float(uncertain.mean()),
                                 'target_watertight': bool(target.is_watertight)}
        report['structures'][name] = {'sample_count': len(points), 'ct_overlap': checks}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = overlap_report(args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
