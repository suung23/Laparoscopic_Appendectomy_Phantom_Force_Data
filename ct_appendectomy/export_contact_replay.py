"""Bake CPU reference contact history into a time-sampled USD, not an RL rollout."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Vt, Gf
from attached_visual_layer import AttachedVisualLayer
from build_scene import mesh_prim
from build_fat_layer import evaluate
from cad_penalty_contact import part_pose


def write_mesh(mesh, points, faces, time=None):
    geometry = trimesh.Trimesh(points, faces, process=False)
    code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    for attr, values in ((mesh.GetPointsAttr(), points),
                         (mesh.GetNormalsAttr(), geometry.vertex_normals),
                         (mesh.GetExtentAttr(), geometry.bounds)):
        attr.Set(Vt.Vec3fArray.FromNumpy(np.asarray(values, np.float32)), code)


def export(scene, parent, layer, trajectory, approach, out, artery=None, cad=None):
    states = np.load(trajectory)
    points, dt = states['points'], float(states['dt'])
    rest = np.load(parent)
    motion = np.load(approach)
    poses = motion['poses']
    jaw_angles=motion['jaw_half_angles'] if 'jaw_half_angles' in motion else np.zeros(len(poses))
    if jaw_angles.shape!=(len(poses),):raise ValueError('Jaw trajectory length mismatch')
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
    # Solver trajectories and source anatomy coordinates are explicitly metres.
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
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
    tool_op = None;hinge_ops={}
    if cad:
        from cad_penalty_contact import load_jaw
        hull = load_jaw(cad)
        if not np.allclose(hull.vertices, motion['jaw_points']) or not np.array_equal(hull.faces, motion['jaw_faces']):
            raise ValueError('Full tool CAD does not match recorded contact hull')
        source = Usd.Stage.Open(str(cad.resolve()))
        cad_units = UsdGeom.GetStageMetersPerUnit(source)
        stage_units = UsdGeom.GetStageMetersPerUnit(stage)
        tool = UsdGeom.Xform.Define(stage, '/World/RosotaReplayTool')
        tool.GetPrim().GetReferences().AddReference(os.path.relpath(cad.resolve(), out.parent.resolve()).replace('\\', '/'), '/Tool')
        tool_op = tool.MakeMatrixXform()
        for name in ('gripper_tip_left','gripper_tip_right'):
            hinge_ops[name]=UsdGeom.Xformable(stage.GetPrimAtPath('/World/RosotaReplayTool/'+name)).MakeMatrixXform()
        UsdGeom.Imageable(jaw).CreateVisibilityAttr().Set('invisible')
    for index, state in enumerate(points):
        # Solver stores pre-step rest at index 0, then post-step states. Tool
        # pose used for post-step state i is approach[min(i-1, last)].
        pose = poses[min(max(index - 1, 0), len(poses) - 1)]
        angle=float(jaw_angles[min(max(index-1,0),len(poses)-1)])
        write_mesh(fat.parent, state, rest['faces'])
        write_mesh(fat.parent, state, rest['faces'], index)
        for follower in (fat, vessel):
            if follower:
                positions = follower.update()
                write_mesh(follower.layer, positions, follower.faces, index)
        left_pose=part_pose(pose,'gripper_tip_left',angle)
        positions = motion['jaw_points'] @ left_pose[:3, :3].T + left_pose[:3, 3]
        write_mesh(jaw, positions, motion['jaw_faces'], index)
        if tool_op:
            transform = pose.copy()
            transform[:3, :3] *= cad_units / stage_units
            tool_op.Set(Gf.Matrix4d(transform.T.tolist()), index)
            for name,op in hinge_ops.items():op.Set(Gf.Matrix4d(part_pose(np.eye(4),name,angle).T.tolist()),index)
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
        angle=float(jaw_angles[min(max(index-1,0),len(poses)-1)])
        left_pose=part_pose(pose,'gripper_tip_left',angle)
        expected_jaw = motion['jaw_points'] @ left_pose[:3, :3].T + left_pose[:3, 3]
        # The follower consumes float32 USD parent positions, as at runtime.
        expected_fat = evaluate(state.astype(np.float32).astype(float), fat.nodes, fat.weights, fat.offset)
        jaw_error = max(jaw_error, float(np.max(np.abs(np.asarray(jaw_mesh.GetPointsAttr().Get(index))-expected_jaw))))
        fat_error = max(fat_error, float(np.max(np.abs(np.asarray(fat_mesh.GetPointsAttr().Get(index))-expected_fat))))
    if max(jaw_error, fat_error) > 1e-8:
        raise RuntimeError('Saved tool or fat replay mismatch')
    tool_error = 0.
    part_count = 0
    if cad:
        for prim in source.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            part_count += 1
            original = UsdGeom.Mesh(prim)
            local = np.asarray(original.GetPointsAttr().Get(), float)
            source_matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
            original_m = (np.column_stack([local, np.ones(len(local))]) @ source_matrix)[:, :3] * cad_units
            target = reopened.GetPrimAtPath(str(prim.GetPath()).replace('/Tool/', '/World/RosotaReplayTool/', 1))
            for index in (0, len(points)-1):
                pose = poses[min(max(index-1, 0), len(poses)-1)]
                pose = part_pose(pose,prim.GetParent().GetName(),float(jaw_angles[min(max(index-1,0),len(poses)-1)]))
                expected = original_m @ pose[:3, :3].T + pose[:3, 3]
                matrix = np.asarray(UsdGeom.Xformable(target).ComputeLocalToWorldTransform(index))
                actual = (np.column_stack([np.asarray(UsdGeom.Mesh(target).GetPointsAttr().Get(index)), np.ones(len(local))]) @ matrix)[:, :3] * stage_units
                tool_error = max(tool_error, float(np.max(np.abs(actual-expected))))
        if part_count < 2 or tool_error > 1e-8:
            raise RuntimeError('Full CAD reference alignment mismatch')
    sources = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in [('scene', scene), ('parent', parent), ('layer', layer),
                                 ('trajectory', trajectory), ('approach', approach)]}
    if artery:
        sources['artery'] = {'sha256': hashlib.sha256(artery.read_bytes()).hexdigest()}
    if cad:
        sources['cad'] = {'sha256': hashlib.sha256(cad.read_bytes()).hexdigest()}
    report = {'status': 'USD replay exported and tissue samples reopened',
              'samples': len(points), 'duration_seconds': (len(points)-1)*dt,
              'dt': dt, 'max_saved_tissue_error_m': error, 'sources': sources,
              'max_saved_jaw_error_m': jaw_error, 'max_saved_fat_error_m': fat_error,
              'scope': 'CPU reference tool/tissue replay; time codes are solver steps; force integration is documented by the source run',
              'RL': False, 'Isaac_physics': False, 'render_verified': False,
              'material_calibrated': False, 'artery_is_visual_only': bool(artery)}
    report['full_distal_CAD_displayed'] = bool(cad)
    report['CAD_mesh_parts'] = part_count
    report['CAD_world_alignment_error_m'] = tool_error
    report['additional_CAD_parts_contact_validated'] = False
    report['jaw_articulation'] = 'recorded Rosota-convention half angles' if 'jaw_half_angles' in motion else 'source CAD rest pose'
    out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'parent', 'layer', 'trajectory', 'approach', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--artery', type=Path)
    parser.add_argument('--cad', type=Path, help='Display original complete distal CAD; added parts are visual only')
    export(**vars(parser.parse_args()))
