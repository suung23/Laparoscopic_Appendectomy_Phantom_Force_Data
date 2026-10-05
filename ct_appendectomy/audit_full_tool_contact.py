"""Audit omitted CAD-part contact in a recorded single-jaw tissue trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd, UsdGeom
from cad_penalty_contact import load_jaw, penalty, part_pose
from build_fat_layer import evaluate


def audit(cad, layer, trajectory, approach, out):
    source = Usd.Stage.Open(str(cad.resolve()))
    names = [p.GetName() for p in source.GetPrimAtPath('/Tool').GetChildren()
             if UsdGeom.Mesh(source.GetPrimAtPath(str(p.GetPath())+'/mesh'))]
    hulls = {name: load_jaw(cad, name) for name in names}
    data = np.load(layer); states = np.load(trajectory); motion = np.load(approach)
    mesh = trimesh.Trimesh(data['points'], data['faces'], process=False)
    area = np.zeros(len(mesh.vertices))
    np.add.at(area, mesh.faces.ravel(), np.repeat(mesh.area_faces/3, 3))
    records = []
    for index, state in enumerate(states['points']):
        points = evaluate(state, data['nodes'], data['weights'], data['frame_offsets'])
        pose = motion['poses'][min(max(index-1, 0), len(motion['poses'])-1)]
        angle=float(motion['jaw_half_angles'][min(max(index-1,0),len(motion['poses'])-1)]) if 'jaw_half_angles' in motion else 0.
        contacts = {}
        for name, local in hulls.items():
            hull = local.copy()
            transform=part_pose(pose,name,angle)
            hull.vertices = local.vertices @ transform[:3, :3].T + transform[:3, 3]
            try:
                _, contacts[name] = penalty(hull, points, area, 1e5)
            except ValueError as exc:
                contacts[name] = {'error': str(exc), 'active_nodes': None}
        records.append({'state': index, 'seconds': index*float(states['dt']), 'parts': contacts})
    omitted = [name for name in names if name != 'gripper_tip_left'
               and any(row['parts'][name]['active_nodes'] for row in records)]
    failures = sorted({name for row in records for name, result in row['parts'].items() if 'error' in result})
    report = {'status': 'incomplete collision audit' if failures else ('omitted part contact detected' if omitted else 'no omitted contact at sampled vertices'),
              'failed_parts': failures,
              'omitted_contact_parts': omitted, 'parts': names, 'samples': records,
              'scope': 'Replay audit only; convex part hulls against authored fat vertices',
              'all_organ_collision_validated': False, 'continuous_collision_checked': False,
              'measured_contact_force': False,
              'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in [('cad', cad), ('layer', layer), ('trajectory', trajectory), ('approach', approach)]}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cad', 'layer', 'trajectory', 'approach', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    audit(**vars(parser.parse_args()))
