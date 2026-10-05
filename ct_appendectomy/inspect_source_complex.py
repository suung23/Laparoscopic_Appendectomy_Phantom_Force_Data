"""Author a reproducible, unobstructed review view of the whole native assembly.

The inspection camera is not a calibrated or trocar-constrained surgical camera.
All source structures remain visible and in their original relative positions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdLux,Gf


def build(source,out):
    original=Usd.Stage.Open(str(source.resolve()))
    meshes={}
    for prim in original.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim)
            meshes[prim.GetName()]=trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
    names=list(meshes)
    selected=['appendix_FJ2565','mesoappendix_FJ3397','ascending_colon_FJ2566']
    aim=meshes[selected[0]].centroid
    cloud=[]
    for name in selected:
        vertices=meshes[name].vertices
        if name==selected[2]:vertices=vertices[np.argsort(np.linalg.norm(vertices-aim,axis=1))[:150]]
        cloud.append(vertices[np.linspace(0,len(vertices)-1,24,dtype=int)])
    cloud=np.vstack(cloud)
    i=np.arange(96)+.5;z=1-2*i/96;angle=i*np.pi*(3-np.sqrt(5))
    directions=np.c_[np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z]
    eyes=np.vstack([aim+radius*directions for radius in (.06,.09)])
    combined=trimesh.util.concatenate(list(meshes.values()))
    owners=np.concatenate([np.full(len(m.faces),index) for index,m in enumerate(meshes.values())])
    _,clearance,_=trimesh.proximity.closest_point(combined,eyes)
    valid=clearance>.004
    for mesh in meshes.values():
        if mesh.is_watertight:valid &= ~mesh.contains(eyes)
    records=[]
    for index in np.flatnonzero(valid):
        rays=cloud-eyes[index];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        hits=combined.ray.intersects_first(np.tile(eyes[index],(len(rays),1)),rays)
        labels=np.where(hits>=0,owners[np.maximum(hits,0)],-1)
        visibility=[float(np.mean(labels[k*24:(k+1)*24]==names.index(name))) for k,name in enumerate(selected)]
        records.append(dict(eye_m=eyes[index].tolist(),clearance_m=float(clearance[index]),visibility=visibility))
    if not records:raise RuntimeError('No source-clear inspection camera')
    best=max(records,key=lambda r:(min(r['visibility'][:2]),sum(r['visibility'])))
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(source.resolve(),out.resolve()).replace('\\','/'))
    UsdGeom.Xform.Define(stage,'/World')
    camera=UsdGeom.Camera.Define(stage,'/World/Endoscope')
    camera.CreateFocalLengthAttr(14.);camera.CreateHorizontalApertureAttr(24.)
    camera.CreateVerticalApertureAttr(13.5);camera.CreateClippingRangeAttr(Gf.Vec2f(.001,2.))
    eye=Gf.Vec3d(*best['eye_m'])
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*aim),Gf.Vec3d(0,0,1)).GetInverse())
    light=UsdLux.SphereLight.Define(stage,'/World/ScopeLight')
    light.CreateIntensityAttr(100.);light.CreateNormalizeAttr(True);light.CreateRadiusAttr(.004)
    UsdGeom.Xformable(light).AddTranslateOp().Set(eye)
    UsdLux.DomeLight.Define(stage,'/World/Fill').CreateIntensityAttr(3.)
    stage.GetRootLayer().Save()
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selected=best,
        targets=selected,aim_m=aim.tolist(),candidate_count=len(eyes),valid_candidates=len(records),
        geometry_changed=False,hidden_structures=False,
        camera_scope='Atlas inspection camera; no trocar, patient or optical calibration',
        ray_visibility_scope='Fraction of selected target rays first hitting the named structure, not screen area')
    (out/'view.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('out/ct_appendectomy/native_source_complex/surgical_region.usda'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/native_source_review'))
    a=p.parse_args();build(a.source,a.out)
