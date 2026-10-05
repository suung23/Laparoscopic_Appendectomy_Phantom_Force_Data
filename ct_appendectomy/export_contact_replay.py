"""Bake CPU reference contact history into a time-sampled USD, not an RL rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Vt
from attached_visual_layer import AttachedVisualLayer
from build_scene import mesh_prim
from build_fat_layer import evaluate


def write_mesh(mesh, points, faces, time=None):
    geometry = trimesh.Trimesh(points, faces, process=False)
    code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    for attr, values in ((mesh.GetPointsAttr(), points),
                         (mesh.GetNormalsAttr(), geometry.vertex_normals),
                         (mesh.GetExtentAttr(), geometry.bounds)):
        attr.Set(Vt.Vec3fArray.FromNumpy(np.asarray(values, np.float32)), code)


def export(scene, parent, layer, trajectory, approach, out, artery=None):
    states = np.load(trajectory)
    points, dt = states['points'], float(states['dt'])
    rest = np.load(parent)
    motion = np.load(approach)
    poses = motion['poses']
    if points.ndim != 3 or points.shape[1:] != rest['points'].shape or len(points) < 2:
        raise ValueError('Invalid tissue trajectory')
    if not np.isfinite(points).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid trajectory values')
    if not np.allclose(points[0], rest['points'], atol=1e-10, rtol=0):
        raise ValueError('Trajectory must begin at supplied rest shape')
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Invalid tool trajectory')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out.resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(), out.parent.resolve()).replace('\\', '/'))
    stage.SetTimeCodesPerSecond(1 / dt)
    stage.SetFramesPerSecond(1 / dt)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(points) - 1)
    fat = AttachedVisualLayer(stage, '/SurgicalRegion/mesoappendix_FJ3397',
                              '/SurgicalRegion/authored_fat_layer', parent, layer)
    vessel = AttachedVisualLayer(stage, '/SurgicalRegion/mesoappendix_FJ3397',
                                 '/SurgicalRegion/appendicular_artery_FJ3410', parent, artery) if artery else None
    if vessel:
        vessel.shape_regularization = True
    jaw = mesh_prim(stage, '/World/FixedCADContactJaw',
                    trimesh.Trimesh(motion['jaw_points'], motion['jaw_faces'], process=False),
                    (.45, .48, .5), roughness=.2)
    for index, state in enumerate(points):
        # Solver stores pre-step rest at index 0, then post-step states. Tool
        # pose used for post-step state i is approach[min(i-1, last)].
        pose = poses[min(max(index - 1, 0), len(poses) - 1)]
        write_mesh(fat.parent, state, rest['faces'])
        write_mesh(fat.parent, state, rest['faces'], index)
        for follower in (fat, vessel):
            if follower:
                positions = follower.update()
                write_mesh(follower.layer, positions, follower.faces, index)
        positions = motion['jaw_points'] @ pose[:3, :3].T + pose[:3, 3]
        write_mesh(jaw, positions, motion['jaw_faces'], index)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(out.resolve()))
    mesh = UsdGeom.Mesh(reopened.GetPrimAtPath(str(fat.parent.GetPath())))
    error = max(float(np.max(np.abs(np.asarray(mesh.GetPointsAttr().Get(i)) - state)))
                for i, state in enumerate(points))
    if error > 1e-8:
        raise RuntimeError('Saved tissue replay mismatch')
    fat_mesh = UsdGeom.Mesh(reopened.GetPrimAtPath(str(fat.layer.GetPath())))
    jaw_mesh = UsdGeom.Mesh(reopened.GetPrimAtPath(str(jaw.GetPath())))
    fat_error = jaw_error = 0.
    for index, state in enumerate(points):
        pose = poses[min(max(index-1, 0), len(poses)-1)]
        expected_jaw = motion['jaw_points'] @ pose[:3, :3].T + pose[:3, 3]
        # The follower consumes float32 USD parent positions, as at runtime.
        expected_fat = evaluate(state.astype(np.float32).astype(float), fat.nodes, fat.weights, fat.offset)
        jaw_error = max(jaw_error, float(np.max(np.abs(np.asarray(jaw_mesh.GetPointsAttr().Get(index))-expected_jaw))))
        fat_error = max(fat_error, float(np.max(np.abs(np.asarray(fat_mesh.GetPointsAttr().Get(index))-expected_fat))))
    if max(jaw_error, fat_error) > 1e-8:
        raise RuntimeError('Saved tool or fat replay mismatch')
    sources = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in [('scene', scene), ('parent', parent), ('layer', layer),
                                 ('trajectory', trajectory), ('approach', approach)]}
    if artery:
        sources['artery'] = {'sha256': hashlib.sha256(artery.read_bytes()).hexdigest()}
    report = {'status': 'USD replay exported and tissue samples reopened',
              'samples': len(points), 'duration_seconds': (len(points)-1)*dt,
              'dt': dt, 'max_saved_tissue_error_m': error, 'sources': sources,
              'max_saved_jaw_error_m': jaw_error, 'max_saved_fat_error_m': fat_error,
              'scope': 'CPU reference single-jaw contact replay; time codes are solver steps',
              'RL': False, 'Isaac_physics': False, 'render_verified': False,
              'material_calibrated': False, 'artery_is_visual_only': bool(artery)}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'parent', 'layer', 'trajectory', 'approach', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--artery', type=Path)
    export(**vars(parser.parse_args()))
