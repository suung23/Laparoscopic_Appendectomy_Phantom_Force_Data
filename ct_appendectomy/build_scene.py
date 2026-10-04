"""Export a provenance-tracked CT + atlas hybrid scene in metres, RAS axes.

Appendix placement is a heuristic initialization, not patient registration.
No physics or surgical-success claim is made by this asset compiler.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, Vt

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_prim(stage, path, mesh, color, roughness=.32):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    obj.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    obj.CreateSubdivisionSchemeAttr('none')
    obj.CreateDoubleSidedAttr(True)
    obj.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(mesh.bounds.astype(np.float32)))
    obj.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
    obj.SetNormalsInterpolation('vertex')
    material = UsdShade.Material.Define(stage, path + '_material')
    shader = UsdShade.Shader.Define(stage, path + '_material/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(material)
    return obj


def build(source, output, anchor=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ctmeta = json.loads((source/'ct.json').read_text())
    atlasmeta = json.loads((source/'anatomy.json').read_text(encoding='utf-8'))
    ct = trimesh.load(source/'ct.glb', force='scene')
    atlas = trimesh.load(source/'anatomy.glb', force='scene')
    ct_transform = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float)
    atlas_transform = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], float)
    for scene, transform in ((ct,ct_transform),(atlas,atlas_transform)):
        for mesh in scene.geometry.values():
            mesh.apply_transform(transform)
            mesh.apply_scale(.001)
    appendix = atlas.geometry['appendix_FJ2565']
    ascending = atlas.geometry['ascending_colon_FJ2566']
    distance, _ = cKDTree(ascending.vertices).query(appendix.vertices)
    atlas_base = appendix.vertices[int(np.argmin(distance))]
    colon = ct.geometry['colon']
    if anchor is None:
        # RAS: patient right is +X; inferior colon selection is provisional.
        right = colon.vertices[colon.vertices[:,0] > np.percentile(colon.vertices[:,0],70)]
        lower = right[right[:,2] <= np.percentile(right[:,2], 15)]
        desired = np.median(lower,axis=0)
        target = lower[np.argmin(np.linalg.norm(lower-desired,axis=1))]
    else:
        target = np.asarray(anchor,float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('Anchor must contain three finite coordinates in metres')
        if np.linalg.norm(target-colon.centroid)>1.:
            raise ValueError('Anchor is over one metre from the colon; check coordinate units')
    offset = target - atlas_base
    stage = Usd.Stage.CreateNew(str(output/'ct_appendectomy.usda'))
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,'Z')
    structures=[]
    for name, mesh in ct.geometry.items():
        color = (.42,.12,.09) if name=='liver' else (.66,.36,.29)
        if 'artery' in name: color=(.55,.055,.04)
        if 'vena' in name: color=(.14,.16,.32)
        obj=mesh_prim(stage,'/World/CT/'+name,mesh,color)
        obj.GetPrim().SetCustomDataByKey('source','TotalSegmentator s0042 segmentation')
        structures.append({'name':name,'source':'CT','triangles':len(mesh.faces)})
    for key in ('appendix_FJ2565','mesoappendix_FJ3397','appendicular_artery_FJ3410'):
        mesh=atlas.geometry[key].copy()
        mesh.apply_translation(offset)
        name=key.split('_FJ')[0]
        UsdGeom.Xform.Define(stage,'/World/Atlas/'+name)
        color=(.72,.3,.25) if name=='appendix' else (.72,.52,.27)
        if name=='appendicular_artery': color=(.6,.03,.025)
        obj=mesh_prim(stage,'/World/Atlas/'+name+'/render',mesh,color)
        obj.GetPrim().SetCustomDataByKey('source','BodyParts3D atlas; heuristic placement, not CT segmentation')
        mesh.export(output/(name+'.ply'))
        structures.append({'name':name,'source':'atlas','triangles':len(mesh.faces)})
    camera=UsdGeom.Camera.Define(stage,'/World/Endoscope')
    aim=appendix.centroid+offset
    eye=aim+np.array([0,.11,.06])
    view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*aim),Gf.Vec3d(0,0,1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    camera.CreateFocalLengthAttr(18)
    camera.CreateHorizontalApertureAttr(24)
    camera.CreateVerticalApertureAttr(13.5)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001,2))
    light=UsdLux.SphereLight.Define(stage,'/World/ScopeLight')
    light.CreateIntensityAttr(100)
    light.CreateNormalizeAttr(True)
    light.CreateExposureAttr(0)
    light.CreateRadiusAttr(.004)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*eye))
    dome=UsdLux.DomeLight.Define(stage,'/World/Fill')
    dome.CreateIntensityAttr(30)
    report={'schema_version':1,'units':'m','axes':'RAS: +X right, +Y anterior, +Z superior',
        'ct':ctmeta,'atlas_source':atlasmeta['source'],'atlas_license':atlasmeta['license'],
        'inputs':{f:digest(source/f) for f in ('ct.glb','ct.json','anatomy.glb','anatomy.json')},
        'appendix_is_patient_specific':False,'registration_validated':False,
        'placement_method':'manual anchor' if anchor is not None else 'heuristic right inferior colon',
        'atlas_base_m':atlas_base.tolist(),'ct_anchor_m':target.tolist(),'atlas_translation_m':offset.tolist(),
        'camera_eye_m':eye.tolist(),'camera_aim_m':aim.tolist(),'structures':structures,
        'appearance':'procedural material, not measured tissue colour',
        'limitations':['No insufflation model','No validated anatomy registration','No cutting or bleeding in exported asset scene']}
    stage.GetRootLayer().customLayerData={'scene_type':'CT + atlas hybrid; registration unvalidated'}
    stage.GetRootLayer().Save()
    (output/'scene_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'stage':str(output/'ct_appendectomy.usda'),'structures':len(structures),'anchor_m':target.tolist()}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT.parent/'appendectomy_lab'/'assets')
    p.add_argument('--out',type=Path,default=ROOT/'out'/'ct_appendectomy')
    p.add_argument('--anchor',type=float,nargs=3,help='Reviewed attachment location in centred RAS metres')
    a=p.parse_args()
    build(a.source,a.out,a.anchor)
