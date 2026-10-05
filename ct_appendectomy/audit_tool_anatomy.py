"""Sample full CAD hulls against static source anatomy; not continuous collision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from author_attachment_curve import mesh_at
from cad_penalty_contact import load_jaw,part_pose
from independent_containment import contains


def audit(scene, cad, approach, out, all_poses=False):
    stage = Usd.Stage.Open(str(scene.resolve()))
    tool = Usd.Stage.Open(str(cad.resolve()))
    motion = np.load(approach); poses = motion['poses']
    angles=motion['jaw_half_angles'] if 'jaw_half_angles' in motion else np.zeros(len(poses))
    if angles.shape!=(len(poses),):raise ValueError('Jaw trajectory length mismatch')
    names = [p.GetName() for p in tool.GetPrimAtPath('/Tool').GetChildren()
             if UsdGeom.Mesh(tool.GetPrimAtPath(str(p.GetPath())+'/mesh'))]
    anatomy = {}
    skipped = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith('/SurgicalRegion/'):
            continue
        mesh = mesh_at(stage, str(prim.GetPath()))
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        mesh.vertices = (np.column_stack([mesh.vertices, np.ones(len(mesh.vertices))]) @ matrix)[:, :3]
        if not mesh.is_watertight:
            skipped.append(str(prim.GetPath()))
        else:
            anatomy[str(prim.GetPath())] = mesh
    # Approach is straight insertion in this fixture. Check both endpoints and
    # maximum insertion; sample spacing below is geometry, not temporal CCD.
    indices = sorted({0, len(poses)-1, int(np.argmax(np.linalg.norm(poses[:, :3, 3]-motion['port'], axis=1)))})
    if 'jaw_half_angles' in motion:indices=sorted(set(indices)|{int(np.argmin(angles)),int(np.argmax(angles))})
    if all_poses:indices=list(range(len(poses)))
    hits = []
    for name in names:
        hull = load_jaw(cad, name)
        vertices, faces = trimesh.remesh.subdivide_to_size(hull.vertices, hull.faces, max_edge=.005, max_iter=10)
        samples = np.vstack([vertices, vertices[faces].mean(axis=1)])
        cache={}
        for index in indices:
            pose = part_pose(poses[index],name,float(angles[index]))
            key=pose.tobytes()
            if key in cache:
                hits.extend([dict(hit,pose_index=index) for hit in cache[key]])
                continue
            start=len(hits)
            world = samples @ pose[:3, :3].T + pose[:3, 3]
            for path, organ in anatomy.items():
                if (world.max(axis=0)<organ.bounds[0]).any() or (world.min(axis=0)>organ.bounds[1]).any():
                    continue
                inside = contains(organ, world)
                if not inside.any():
                    continue
                selected = world[inside]
                depth = np.concatenate([trimesh.proximity.closest_point_naive(organ, chunk)[1]
                                        for chunk in np.array_split(selected, max(1, len(selected)//16+1))])
                deep = depth>1e-4
                if deep.any():
                    hits.append({'part': name, 'pose_index': index, 'organ': path,
                                 'samples_over_0_1mm': int(deep.sum()),
                                 'max_depth_mm': float(depth.max()*1000)})
            cache[key]=hits[start:]
    report = {'status': 'anatomy penetration found' if hits else 'no penetration at inspected samples',
              'surgical_approach_accepted': False,
              'all_recorded_poses_checked':all_poses,'jaw_angles_applied':True,
              'poses_checked': indices, 'parts_checked': names, 'closed_organs_checked': len(anatomy),
              'open_meshes_skipped': skipped, 'penetrations': hits,
              'scope': 'Static source anatomy against subdivided convex CAD hull samples at selected poses',
              'continuous_collision': False, 'clinical_port_calibration': False,
              'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name,path in [('scene',scene),('cad',cad),('approach',approach)]}}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','cad','approach','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--all-poses',action='store_true')
    audit(**vars(parser.parse_args()))
