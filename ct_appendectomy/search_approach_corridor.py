"""Coarse geometric shaft corridor search; no anatomical port or robot IK claim."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from author_attachment_curve import mesh_at
from independent_containment import contains

def search(scene,layer,out):
    d=np.load(layer);tissue=trimesh.Trimesh(d['points'],d['faces'],process=False)
    target=tissue.vertices[710];normal=tissue.vertex_normals[710]
    helper=np.eye(3)[np.argmin(np.abs(normal))];x=np.cross(helper,normal);x/=np.linalg.norm(x);y=np.cross(normal,x)
    directions=[normal]
    for tilt in (20,40,60):
        for azimuth in np.arange(8)*np.pi/4:
            angle=np.deg2rad(tilt)
            directions.append(normal*np.cos(angle)+(x*np.cos(azimuth)+y*np.sin(azimuth))*np.sin(angle))
    samples=[]
    for direction in directions:
        helper=np.eye(3)[np.argmin(np.abs(direction))];u=np.cross(helper,direction);u/=np.linalg.norm(u);v=np.cross(direction,u)
        ring=np.array([np.zeros(3)]+[.003*(u*np.cos(a)+v*np.sin(a)) for a in np.arange(12)*np.pi/6])
        samples.append((target+np.arange(.025,.410,.002)[:,None,None]*direction+ring).reshape(-1,3))
    shape=np.array(samples).shape;points=np.concatenate(samples);counts=np.zeros(len(points),int);organs={}
    stage=Usd.Stage.Open(str(scene.resolve()))
    skipped=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith('/SurgicalRegion/'):continue
        mesh=mesh_at(stage,str(prim.GetPath()))
        matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        mesh.vertices=(np.column_stack([mesh.vertices,np.ones(len(mesh.vertices))])@matrix)[:,:3]
        if not mesh.is_watertight:skipped.append(str(prim.GetPath()));continue
        mask=contains(mesh,points);counts+=mask
        if mask.any():organs[str(prim.GetPath())]=mask.reshape(shape[:2]).sum(axis=1).tolist()
    scores=counts.reshape(shape[:2]).sum(axis=1);order=np.argsort(scores,kind='stable')
    candidates=[{'index':int(i),'forward':(-directions[i]).tolist(),'inside_sample_count':int(scores[i])} for i in order]
    report={'status':'coarse candidates only','candidates':candidates,'organ_sample_counts':organs,'skipped_open_organs':skipped,
            'target_vertex':710,'shaft_envelope_radius_m':.003,'axial_sampling_m':.002,
            'full_CAD_validated':False,'port_anatomically_calibrated':False,'continuous_collision':False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(candidates[:5],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','layer','out'):p.add_argument('--'+name,type=Path,required=True)
    search(**vars(p.parse_args()))
