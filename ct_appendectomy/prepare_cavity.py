"""Prepare private CAD and a posterior support approximation for local inspection.

CT organs retain their source poses. This support is NOT segmented peritoneum.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf
from build_scene import ROOT, build, mesh_prim


def prepare():
    output=ROOT/'out/ct_appendectomy/cavity'
    build(ROOT.parent/'appendectomy_lab/assets', output,
          placement=ROOT/'ct_appendectomy/profiles/s0042_hybrid.json')
    source=json.loads((ROOT/'ct_appendectomy/rosota_distal_manifest.json').read_text())
    rotation=Rotation.from_euler('xyz',[3.141593,0,2.356194]).as_matrix()
    # Right-handed frame: closing X, hinge Y, insertion Z.
    basis=np.array([[2**-.5,2**-.5,0],[-2**-.5,2**-.5,0],[0,0,1.]])
    cad=Usd.Stage.CreateNew(str(output/'rosota_distal.usda'))
    cad.SetDefaultPrim(UsdGeom.Xform.Define(cad,'/Tool').GetPrim())
    UsdGeom.SetStageMetersPerUnit(cad,1.)
    for item in source['files']:
        file=ROOT/'assets/rosota_distal'/Path(item['path']).name
        raw=file.read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['sha']:
            raise ValueError('Upstream CAD hash mismatch: '+file.name)
        mesh=trimesh.load(file)
        mesh.vertices=(mesh.vertices*.001@rotation.T+np.array([0,0,.162154])-
                       np.array([.000093,-.000085,.494949]))@basis
        UsdGeom.Xform.Define(cad,'/Tool/'+file.stem)
        mesh_prim(cad,'/Tool/'+file.stem+'/mesh',mesh,(.55,.58,.61),.2)
        UsdShade.Shader(cad.GetPrimAtPath('/Tool/'+file.stem+'/mesh_material/surface')).GetInput('metallic').Set(.85)
    cad.GetRootLayer().Save()
    stage=Usd.Stage.Open(str(output/'ct_appendectomy.usda'))
    appendix=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath('/World/Atlas/appendix/render')).GetPointsAttr().Get())
    floor=float(appendix[:,1].min()-.0008)
    # Broad local support under the ileocaecal region. Keep CT registration fixed.
    center=appendix.mean(0)
    bed=trimesh.creation.box(extents=[.20,.012,.20])
    bed.apply_translation([center[0]-.025,floor-.006,center[2]])
    obj=mesh_prim(stage,'/World/PosteriorSupport/render',bed,(.42,.19,.17),.55)
    obj.GetPrim().SetCustomDataByKey('source','Engineering posterior support approximation; not CT peritoneum')
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr('convexHull')
    stage.GetRootLayer().Save()
    manifest=json.loads((output/'scene_manifest.json').read_text())
    # A conservative free ball certifies ALL camera translations in the UI range.
    eye=np.asarray(manifest['camera_eye_m'])
    from anatomy_audit import load_meshes
    meshes=load_meshes(output/'ct_appendectomy.usda')
    # Find a nearby working position with room for actual camera translation.
    offsets=np.array(np.meshgrid(*([np.linspace(-.02,.02,7)]*3))).reshape(3,-1).T
    candidates=eye+offsets
    valid=np.ones(len(candidates),dtype=bool)
    distances=np.full(len(candidates),np.inf)
    for name,m in meshes.items():
        inside=m.contains(candidates)
        valid &= inside if name=='/World/BodyEnvelope/render' else ~inside
        distances=np.minimum(distances,trimesh.proximity.closest_point(m,candidates)[1])
    from audit_view import first_hits
    target=meshes['/World/Atlas/appendix/render']
    all_meshes=list(meshes.values())
    triangles=trimesh.util.concatenate(all_meshes).triangles
    owners=np.concatenate([np.full(len(m.faces),i) for i,m in enumerate(all_meshes)])
    target_id=list(meshes).index('/World/Atlas/appendix/render')
    sample=target.vertices[np.linspace(0,len(target.vertices)-1,24,dtype=int)]
    selected=None
    for idx in np.argsort(-distances):
        if not valid[idx] or distances[idx]<.004:continue
        rays=sample-candidates[idx];length=np.linalg.norm(rays,axis=1);rays/=length[:,None]
        hits=first_hits(triangles,np.tile(candidates[idx],(len(rays),1)),rays,length+.001)
        visible=((hits>=0)&(owners[np.maximum(hits,0)]==target_id)).mean()
        if visible>=.25:selected=int(idx);break
    if selected is None:raise RuntimeError('No visible camera position with 4mm clearance')
    eye=candidates[selected];clearance=float(distances[selected])
    manifest['camera_eye_m']=eye.tolist()
    pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*manifest['camera_aim_m']),Gf.Vec3d(0,1,0)).GetInverse()
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0].Set(pose)
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/ScopeLight')).GetOrderedXformOps()[0].Set(Gf.Vec3d(*eye))
    UsdLux.SphereLight(stage.GetPrimAtPath('/World/ScopeLight')).GetIntensityAttr().Set(20.)
    stage.GetRootLayer().Save()
    travel=min(.008,max(0.,clearance-.003))
    manifest['cavity_exploration']={'gravity_direction':[0,-1,0],'gravity_m_s2':9.81,
        'posterior_support_y_m':floor,'support_is_segmented_peritoneum':False,
        'static_ct_organs':True,'camera_clearance_m':clearance,'camera_travel_radius_m':travel,
        'cad_commit':source['commit'],'cad_file':'rosota_distal.usda'}
    (output/'scene_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['cavity_exploration']))


if __name__=='__main__': prepare()
