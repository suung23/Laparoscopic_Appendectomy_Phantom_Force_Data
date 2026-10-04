"""Search collision-free internal camera poses without modifying anatomy.

This is an initialization search, not an anatomically validated trocar plan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom

from anatomy_audit import contains, load_meshes
from audit_view import first_hits


def plan(scene, output):
    meshes = load_meshes(scene)
    target = meshes['/World/Atlas/appendix/render']
    aim = target.centroid
    index = np.arange(80)+.5
    z = 1-2*index/80
    phi = index*np.pi*(3-np.sqrt(5))
    directions = np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
    # Anterior viewing hemisphere; no posterior-camera shortcut.
    directions = directions[directions[:,1] > .15]
    candidates = np.concatenate([aim+radius*directions for radius in (.045,.065,.085,.105)])
    inside, uncertain = contains(meshes['/World/BodyEnvelope/render'],candidates)
    valid = inside & ~uncertain
    for name, mesh in meshes.items():
        if name != '/World/BodyEnvelope/render':
            inside, uncertain = contains(mesh,candidates)
            valid &= ~inside & ~uncertain
    candidates = candidates[valid]
    if not len(candidates):
        raise RuntimeError('No candidate is inside the envelope and outside all organs')
    all_meshes = list(meshes.values())
    triangles = trimesh.util.concatenate(all_meshes).triangles
    owners = np.concatenate([np.full(len(mesh.faces), i) for i,mesh in enumerate(all_meshes)])
    target_id = list(meshes).index('/World/Atlas/appendix/render')
    sample = target.vertices[np.linspace(0,len(target.vertices)-1,24,dtype=int)]
    results = []
    for eye in candidates:
        rays = sample-eye
        distances = np.linalg.norm(rays,axis=1)
        rays /= distances[:,None]
        hits = first_hits(triangles,np.tile(eye,(len(rays),1)),rays,distances+.001)
        visible = (hits>=0) & (owners[np.maximum(hits,0)]==target_id)
        results.append({'eye_m':eye.tolist(),'target_ray_visible_fraction':float(visible.mean()),
                        'distance_to_target_m':float(np.linalg.norm(eye-aim))})
    # Prefer visibility, then working distance near 65 mm; deterministic tie break.
    best = max(results,key=lambda r:(r['target_ray_visible_fraction'],-abs(r['distance_to_target_m']-.065)))
    report = {'method':'Anterior hemisphere search; opaque first-hit rays to appendix vertices',
              'camera_eye_m':best['eye_m'],'camera_aim_m':aim.tolist(),'selected':best,
              'candidate_count':len(results),'candidates':results,
              'port_calibrated':False,'cavity_validated':False,
              'limitations':['Camera centre clearance only; shaft/trocar path not validated',
                             'Visual body envelope is not a segmented peritoneum']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if best['target_ray_visible_fraction'] == 0:
        raise RuntimeError('All candidates occluded; exposure/anatomy work required')
    stage = Usd.Stage.Open(str(scene))
    camera = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope'))
    eye = Gf.Vec3d(*best['eye_m'])
    pose = Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*aim),Gf.Vec3d(0,0,1)).GetInverse()
    camera.GetOrderedXformOps()[0].Set(pose)
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/ScopeLight')).GetOrderedXformOps()[0].Set(eye)
    stage.GetRootLayer().Save()
    manifest_path = scene.parent/'scene_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['camera_eye_m'] = best['eye_m']
    manifest['camera_aim_m'] = aim.tolist()
    manifest['camera_placement'] = report['method']+'; not port calibrated'
    manifest['camera_search'] = {key:value for key,value in report.items() if key!='candidates'}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='candidates'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plan(args.scene,args.out)
