"""Bake a rest-surface triplanar field into existing face-varying UVs.

This repairs unrelated image samples at atlas seams. It does not reconstruct
clinical tissue, vessels, reflectance or anatomy. Surface positions are only
used during baking, so textures subsequently follow the deforming mesh UVs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
from reference_appearance import atlas_coordinates, require_nondegenerate_uv


def mirrored_sample(image, coordinates):
    folded = np.mod(coordinates, 2.)
    folded = np.minimum(folded, 2. - folded)
    xy = np.stack([folded[:, 1] * (image.shape[0]-1),
                   folded[:, 0] * (image.shape[1]-1)])
    return np.stack([map_coordinates(image[..., c], xy, order=1, mode='nearest')
                     for c in range(image.shape[2])], axis=1)


def surface_field(points, normals, image, period):
    weights = np.abs(normals) ** 4
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-15)
    result = np.zeros((len(points), image.shape[2]))
    for axis, coordinates in enumerate(((1, 2), (0, 2), (0, 1))):
        result += weights[:, axis, None] * mirrored_sample(image, points[:, coordinates] / period)
    return result


def bake(points, faces, normals, uv, image, size=2048, period=.04):
    if size < 64 or period <= 0 or not np.isfinite(period):
        raise ValueError('Positive finite surface period and atlas size >= 64 required')
    pixels = np.stack([uv[..., 0]*size-.5, (1.-uv[..., 1])*size-.5], axis=-1)
    result = np.zeros((size, size, image.shape[2]), dtype=np.float32)
    covered = np.zeros((size, size), dtype=bool)
    for face, triangle in zip(faces, pixels):
        lo = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(triangle.max(axis=0)).astype(int), size-1)
        if np.any(hi < lo):
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        q = np.stack([x.ravel(), y.ravel()], axis=1)
        a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
        determinant = a[0]*b[1]-a[1]*b[0]
        if abs(determinant) < 1e-12:
            continue
        d = q-triangle[0]
        w1 = (d[:,0]*b[1]-d[:,1]*b[0])/determinant
        w2 = (a[0]*d[:,1]-a[1]*d[:,0])/determinant
        w = np.stack([1-w1-w2, w1, w2], axis=1)
        inside = np.all(w >= -1e-7, axis=1)
        q, w = q[inside], w[inside]
        if not len(q):
            continue
        value = surface_field(w @ points[face], w @ normals[face], image, period)
        result[q[:,1], q[:,0]] = value
        covered[q[:,1], q[:,0]] = True
    if not covered.any():
        raise ValueError('UV rasterization produced no covered pixels')
    # Nearest covered texels pad atlas charts for bilinear filtering. This does
    # not remove finite-resolution seam error; that must be measured/rendered.
    nearest = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    result[~covered] = result[nearest[0][~covered], nearest[1][~covered]]
    return result, float(covered.mean())


def build(source, out, size):
    import xatlas
    out.mkdir(parents=True, exist_ok=True)
    source_stage = Usd.Stage.Open(str(source.resolve()))
    stage = Usd.Stage.CreateNew(str((out/'appearance.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(source.resolve(), out.resolve()).replace('\\', '/'))
    records = []
    for prim in source_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material or not str(material.GetPath()).startswith('/World/Appendix300Look/'):
            continue
        material_path = str(material.GetPath())
        color_asset = source_stage.GetPrimAtPath(material_path+'/Color').GetAttribute('inputs:file').Get()
        rough_asset = source_stage.GetPrimAtPath(material_path+'/Roughness').GetAttribute('inputs:file').Get()
        color_path, rough_path = Path(color_asset.resolvedPath), Path(rough_asset.resolvedPath)
        color = np.asarray(Image.open(color_path).convert('RGB'), dtype=float)/255
        rough = np.asarray(Image.open(rough_path).convert('L'), dtype=float)/255
        field_image = np.dstack([color, rough])
        points = np.asarray(mesh.GetPointsAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        normals = np.asarray(mesh.GetNormalsAttr().Get())
        atlas = xatlas.Atlas()
        atlas.add_mesh(atlas_coordinates(points), faces.astype(np.uint32))
        packing = xatlas.PackOptions()
        packing.padding = 8
        packing.resolution = size
        atlas.generate(pack_options=packing)
        mapping, indices, coords = atlas[0]
        if not np.array_equal(mapping[indices], faces):
            raise ValueError('Atlas changed triangle correspondence')
        uv = coords[indices]
        require_nondegenerate_uv(uv)
        target = UsdGeom.Mesh(stage.GetPrimAtPath(prim.GetPath()))
        UsdGeom.PrimvarsAPI(target).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(uv.reshape(-1,2).astype(np.float32)))
        if len(normals) != len(points) or mesh.GetNormalsInterpolation() != UsdGeom.Tokens.vertex:
            raise ValueError('Requires smooth vertex normals')
        raster, occupancy = bake(points, faces, normals, uv, field_image, size)
        stem = material_path.rsplit('/',1)[1]
        for channel, pixels in [('Color', raster[...,:3]), ('Roughness', raster[...,3])]:
            filename = stem+'_'+channel.lower()+'.png'
            Image.fromarray(np.uint8(np.clip(pixels,0,1)*255)).save(out/filename)
            stage.GetPrimAtPath(material_path+'/'+channel).GetAttribute('inputs:file').Set(Sdf.AssetPath(filename))
        records.append(dict(mesh=str(prim.GetPath()), atlas_occupancy=occupancy,
            packer_dimensions=[atlas.width,atlas.height], collapsed_uv_triangles=0,
            source_color_sha256=hashlib.sha256(color_path.read_bytes()).hexdigest(),
            source_roughness_sha256=hashlib.sha256(rough_path.read_bytes()).hexdigest()))
        print('Baked '+str(prim.GetPath()), flush=True)
    stage.GetRootLayer().Save()
    report = dict(source_scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        atlas_size=size, requested_packer_padding_pixels=8, surface_period_m=.04, geometry_unchanged=True, camera_unchanged=True,
        lighting_unchanged=True, physics_unchanged=True, meshes=records,
        status='unreviewed seam-repair candidate', clinical_reconstruction=False,
        limitations=['Same synthetic source textures; no recovered albedo or vessels',
            'Triplanar blending can blur and duplicate motifs',
            'Finite atlas resolution leaves residual filtering error',
            'Anatomical registration and tissue mechanics remain unresolved'])
    (out/'bake.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--size', type=int, default=2048)
    a=p.parse_args()
    build(a.source,a.out,a.size)
