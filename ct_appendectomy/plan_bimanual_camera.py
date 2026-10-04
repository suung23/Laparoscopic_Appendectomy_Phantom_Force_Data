"""Find a fixed endoscopic view of BOTH settled grasp sites, preserving anatomy."""
from pathlib import Path
import json
import shutil
import argparse
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf
from anatomy_audit import load_meshes
from audit_view import first_hits


def plan(scene,run,out):
    meshes=load_meshes(scene)
    attached=Usd.Stage.Open(str(run/'attached.usda'))
    live=np.load(run/'settled_geometry.npz')
    coll=attached.GetPrimAtPath('/World/Atlas/appendix/coll')
    target=trimesh.Trimesh(live['collision'],np.asarray(coll.GetAttribute('surfaceFaceVertexIndices').Get()).reshape(-1,3),process=False)
    meshes['/World/Atlas/appendix/render']=target
    centers=[];samples=[]
    for index in (1,5):
        ids=np.asarray(attached.GetPrimAtPath('/World/Bimanual/Grip'+str(index)+'Attachment').GetAttribute('omniphysics:vtxIndicesSrc0').Get(),int)
        center=live['simulation'][ids].mean(0);centers.append(center)
        nearest=np.argsort(np.linalg.norm(target.vertices-center,axis=1))[:100]
        samples.append(target.vertices[nearest[np.linspace(0,len(nearest)-1,12,dtype=int)]])
    aim=np.mean(centers,axis=0)
    i=np.arange(160)+.5;z=1-2*i/len(i);phi=i*np.pi*(3-np.sqrt(5))
    directions=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
    directions=directions[directions[:,1]>.1]
    candidates=np.concatenate([aim+r*directions for r in (.045,.065,.085,.105)])
    valid=np.ones(len(candidates),bool);clearance=np.full(len(candidates),np.inf)
    for name,m in meshes.items():
        inside=m.contains(candidates);valid &= inside if name=='/World/BodyEnvelope/render' else ~inside
        clearance=np.minimum(clearance,trimesh.proximity.closest_point(m,candidates)[1])
    valid &= clearance>.004
    all_meshes=list(meshes.values());triangles=trimesh.util.concatenate(all_meshes).triangles
    owners=np.concatenate([np.full(len(m.faces),i) for i,m in enumerate(all_meshes)])
    target_id=list(meshes).index('/World/Atlas/appendix/render')
    results=[];samples=np.vstack(samples)
    for idx in np.flatnonzero(valid):
        eye=candidates[idx];rays=samples-eye;distance=np.linalg.norm(rays,axis=1);rays/=distance[:,None]
        hits=first_hits(triangles,np.tile(eye,(len(rays),1)),rays,distance+.001)
        visible=((hits>=0)&(owners[np.maximum(hits,0)]==target_id)).reshape(2,12).mean(1)
        results.append({'eye':eye.tolist(),'clearance_m':float(clearance[idx]),'visible_fraction':visible.tolist()})
    if not results:raise RuntimeError('No free internal camera position')
    best=max(results,key=lambda r:(min(r['visible_fraction']),sum(r['visible_fraction']),r['clearance_m']))
    if min(best['visible_fraction'])<.25:raise RuntimeError('Both grasp sites not sufficiently visible: '+json.dumps(best))
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.Open(str(scene));eye=Gf.Vec3d(*best['eye'])
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*aim),Gf.Vec3d(0,1,0)).GetInverse())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/ScopeLight')).GetOrderedXformOps()[0].Set(eye)
    stage.GetRootLayer().Export(str(out/'ct_appendectomy.usda'))
    shutil.copyfile(scene.parent/'rosota_distal.usda',out/'rosota_distal.usda')
    manifest=json.loads((scene.parent/'scene_manifest.json').read_text())
    manifest['camera_eye_m']=best['eye'];manifest['camera_aim_m']=aim.tolist()
    manifest['cavity_exploration']['camera_clearance_m']=best['clearance_m']
    manifest['cavity_exploration']['camera_travel_radius_m']=min(.008,best['clearance_m']-.003)
    manifest['bimanual_camera_search']={'selected':best,'candidate_count':len(results),'based_on_settled_surface':True}
    (out/'scene_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['bimanual_camera_search']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan(a.scene,a.run,a.out)
