"""Numerical seam comparison and geometric invariants for surface texture baking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from pxr import Usd, UsdGeom, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from surface_texture_bake import bake, mirrored_sample


def analytic_checks():
    gradient = np.stack(np.meshgrid(np.linspace(0,1,11), np.linspace(0,1,11)), axis=-1)
    points = np.array([[0,0,0],[.04,0,0],[0,.04,0]])
    normals = np.tile([0,0,1.], (3,1))
    raster, _ = bake(points, np.array([[0,1,2]]), normals,
        np.array([[[0,0],[1,0],[0,1]]]), gradient, size=64)
    np.testing.assert_allclose(raster[50,5], [5.5/64,1-50.5/64], atol=1e-7)
    # Mirrored sampling agrees on either side of each tile boundary.
    left = mirrored_sample(gradient, np.array([[1-1e-8,.3],[2-1e-8,.3]]))
    right = mirrored_sample(gradient, np.array([[1+1e-8,.3],[2+1e-8,.3]]))
    np.testing.assert_allclose(left, right, atol=1e-7)


def sample(image, uv):
    xy = np.stack([(1-uv[:,1])*image.shape[0]-.5, uv[:,0]*image.shape[1]-.5])
    return np.stack([map_coordinates(image[...,i],xy,order=1,mode='nearest')
        for i in range(image.shape[2])],axis=1)


def seam_error(mesh, image):
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').ComputeFlattened()).reshape(-1,3,2)
    du,dv = uv[:,1]-uv[:,0],uv[:,2]-uv[:,0]
    collapsed = np.abs(du[:,0]*dv[:,1]-du[:,1]*dv[:,0]) <= 1e-12
    points = np.asarray(mesh.GetPointsAttr().Get())
    triangle = points[faces]
    area = np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)/2
    edges, pairs = {}, []
    for f, st in zip(faces, uv):
        for a,b in ((0,1),(1,2),(2,0)):
            key = tuple(sorted((int(f[a]),int(f[b]))))
            pair = st[[a,b]] if f[a]<f[b] else st[[b,a]]
            if key in edges:
                other = edges[key]
                if np.max(np.linalg.norm(pair-other,axis=1))>1e-5:
                    pairs.append((pair.mean(axis=0),other.mean(axis=0)))
            else:
                edges[key] = pair
    pairs = np.asarray(pairs)
    errors = np.abs(sample(image,pairs[:,0])-sample(image,pairs[:,1])).mean(axis=1)
    return dict(collapsed_uv_triangles=int(collapsed.sum()),
        collapsed_uv_surface_area_fraction=float(area[collapsed].sum()/area.sum()),
        seam_edges=len(errors), mean_rgb_absolute_jump=float(errors.mean()),
        p95_rgb_absolute_jump=float(np.quantile(errors,.95)),max_rgb_absolute_jump=float(errors.max()))


def compare(before, after):
    a,b = Usd.Stage.Open(str(before)), Usd.Stage.Open(str(after))
    records=[]
    for prim in a.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        old,new=UsdGeom.Mesh(prim),UsdGeom.Mesh(b.GetPrimAtPath(prim.GetPath()))
        for attr in ('points','faceVertexCounts','faceVertexIndices','normals'):
            np.testing.assert_array_equal(prim.GetAttribute(attr).Get(),new.GetPrim().GetAttribute(attr).Get())
        mat,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not mat or not str(mat.GetPath()).startswith('/World/Appendix300Look/'): continue
        path=str(mat.GetPath())+'/Color'
        data=[]
        for stage,mesh in ((a,old),(b,new)):
            asset=stage.GetPrimAtPath(path).GetAttribute('inputs:file').Get()
            image=np.asarray(Image.open(asset.resolvedPath).convert('RGB'),dtype=float)/255
            data.append(seam_error(mesh,image))
        records.append(dict(mesh=str(prim.GetPath()),before=data[0],after=data[1]))
    for prim in a.Traverse():
        if prim.GetPath().name in ('Endoscope','Fill','ScopeLight'):
            for attr in prim.GetAttributes():
                old=attr.Get(); new=b.GetPrimAtPath(prim.GetPath()).GetAttribute(attr.GetName()).Get()
                if old is not None: np.testing.assert_array_equal(old,new)
    return dict(analytic_checks_passed=True, geometry_and_normals_unchanged=True,
        camera_and_named_lights_unchanged=True,
        before_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
        after_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),meshes=records,
        metric='Mean channel absolute difference in sRGB [0,1] at paired UV seam edge midpoints; bilinear sampling',
        limitations='Measures texture continuity only, not realism, BRDF, anatomy or mechanics')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    analytic_checks();report=compare(args.before,args.after)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
