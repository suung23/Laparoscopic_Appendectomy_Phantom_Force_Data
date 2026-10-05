"""Transfer reviewed appearance onto matching rest geometry, without scene replacement."""
import hashlib
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
from reference_appearance import require_nondegenerate_uv


def apply_appearance(target, source_path):
    source_path=Path(source_path).resolve()
    source=Usd.Stage.Open(str(source_path))
    material_root='/World/Appendix300Look'
    if not source.GetPrimAtPath(material_root):
        raise ValueError('Missing appearance material library')
    if target.GetPrimAtPath(material_root):
        raise ValueError('Appearance already installed; reset to original stage first')
    meshes=[]; assets=[]
    # Validate every target and texture before changing the live stage.
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material or not str(material.GetPath()).startswith(material_root+'/'):continue
        live=target.GetPrimAtPath(prim.GetPath())
        if not live or not live.IsA(UsdGeom.Mesh):raise ValueError('Missing target mesh: '+str(prim.GetPath()))
        for name in ('points','faceVertexCounts','faceVertexIndices'):
            if not np.array_equal(prim.GetAttribute(name).Get(),live.GetAttribute(name).Get()):
                raise ValueError('Rest geometry correspondence mismatch: '+str(prim.GetPath())+'/'+name)
        uv=UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        if uv.GetInterpolation()!=UsdGeom.Tokens.faceVarying:raise ValueError('Face-varying UVs required')
        coords=np.asarray(uv.ComputeFlattened())
        require_nondegenerate_uv(coords.reshape(-1,3,2))
        meshes.append((str(prim.GetPath()),str(material.GetPath()),coords))
    if not meshes:raise ValueError('No transferable appearance meshes')
    for prim in Usd.PrimRange(source.GetPrimAtPath(material_root)):
        for attr in prim.GetAttributes():
            if attr.GetTypeName()==Sdf.ValueTypeNames.Asset and attr.Get():
                path=Path(attr.Get().resolvedPath)
                if not path.is_file():raise ValueError('Unresolved appearance asset: '+str(attr.GetPath()))
                assets.append((str(attr.GetPath()),path))
    Sdf.CopySpec(source.Flatten(),material_root,target.GetEditTarget().GetLayer(),material_root)
    for name,path in assets:target.GetAttributeAtPath(name).Set(Sdf.AssetPath(path.as_posix()))
    for mesh,material,coords in meshes:
        prim=target.GetPrimAtPath(mesh)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(coords.astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(target.GetPrimAtPath(material)))
    return dict(source_scene_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        meshes=[m[0] for m in meshes],texture_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for _,p in assets},
        scope='Material bindings and UVs only; no geometry, camera, lighting or physics copied',
        clinical_appearance_validated=False)
