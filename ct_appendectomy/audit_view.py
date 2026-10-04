"""Deterministic geometric camera audit; not an RTX or anatomy validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom


def first_hits(triangles, origins, directions, max_distances):
    """Bounded-memory Moller-Trumbore intersection without native rtree state."""
    a = triangles[:, 0]
    edge1 = triangles[:, 1] - a
    edge2 = triangles[:, 2] - a
    result = np.full(len(origins), -1, dtype=int)
    for i, (origin, direction) in enumerate(zip(origins, directions)):
        p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum('ij,ij->i', edge1, p)
        valid = np.abs(determinant) > 1e-14
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
        delta = origin - a
        u = np.einsum('ij,ij->i', delta, p) * inverse
        q = np.cross(delta, edge1)
        v = (q @ direction) * inverse
        distance = np.einsum('ij,ij->i', edge2, q) * inverse
        valid &= (u >= 0) & (v >= 0) & (u+v <= 1)
        valid &= (distance >= 0) & (distance <= max_distances[i])
        if np.any(valid):
            result[i] = np.argmin(np.where(valid, distance, np.inf))
    return result


def audit(scene, camera_path='/World/Endoscope', width=96, height=54):
    if width < 1 or height < 1:
        raise ValueError('Image dimensions must be positive')
    stage = Usd.Stage.Open(str(scene))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Expected a readable USD stage in metres')
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    if not camera or camera.GetProjectionAttr().Get() != 'perspective':
        raise ValueError('Expected a perspective camera')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    camera_transform = cache.GetLocalToWorldTransform(camera.GetPrim())
    meshes, paths, face_owner = [], [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        if geom.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if geom.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        counts = np.asarray(geom.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f'Triangles required: {prim.GetPath()}')
        vertices = np.asarray(geom.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3]
        faces = np.asarray(geom.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        meshes.append(trimesh.Trimesh(vertices, faces, process=False))
        face_owner.extend([len(paths)] * len(faces))
        paths.append(str(prim.GetPath()))
    if not meshes:
        raise ValueError('No visible triangle meshes')
    mesh = trimesh.util.concatenate(meshes)
    focal = float(camera.GetFocalLengthAttr().Get())
    aperture_x = float(camera.GetHorizontalApertureAttr().Get())
    aperture_y = float(camera.GetVerticalApertureAttr().Get())
    offset_x = float(camera.GetHorizontalApertureOffsetAttr().Get())
    offset_y = float(camera.GetVerticalApertureOffsetAttr().Get())
    x, y = np.meshgrid((np.arange(width) + .5) / width - .5,
                       .5 - (np.arange(height) + .5) / height)
    local = np.c_[x.ravel()*aperture_x + offset_x,
                  y.ravel()*aperture_y + offset_y, np.full(width*height, -focal)]
    local /= np.linalg.norm(local, axis=1)[:, None]
    rays = np.array([camera_transform.TransformDir(Gf.Vec3d(*v)) for v in local])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    eye = np.asarray(camera_transform.ExtractTranslation())
    near, far = camera.GetClippingRangeAttr().Get()
    near_distance = float(near) / -local[:, 2]
    far_distance = float(far) / -local[:, 2]
    origins = eye + rays*near_distance[:, None]
    hits = first_hits(mesh.triangles, origins, rays, far_distance-near_distance)
    ray_ids = np.flatnonzero(hits >= 0)
    triangle_ids = hits[ray_ids]
    owner = np.full(width*height, -1, dtype=int)
    owner[ray_ids] = np.asarray(face_owner)[triangle_ids]
    fractions = {path: float(np.mean(owner == i)) for i, path in enumerate(paths)}
    return {
        'schema_version': 1,
        'scene_sha256': hashlib.sha256(Path(scene).read_bytes()).hexdigest(),
        'camera': camera_path, 'sample_resolution': [width, height],
        'horizontal_fov_deg': float(np.degrees(2*np.arctan(aperture_x/(2*focal)))),
        'first_hit_frame_fraction': fractions,
        'background_fraction': float(np.mean(owner == -1)),
        'appendix_visible_in_sample': fractions.get('/World/Atlas/appendix/render', 0) > 0,
        'method': 'Pixel-centre pinhole rays; opaque triangle first hit at default USD time',
        'limitations': ['No RTX materials, distortion, transparency or dynamic deformation',
                       'Geometric visibility does not validate clinical anatomy or realism',
                       'Small structures may fall between sampled rays'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=54)
    args = parser.parse_args()
    report = audit(args.scene, width=args.width, height=args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
