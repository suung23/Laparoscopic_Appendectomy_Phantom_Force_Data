"""Image-derived liver colour experiment, not recovered albedo or tissue mechanics.

UVs are authored on unchanged surface topology. No fake normals, displacement,
or material stiffness is inferred from the image. Only liver is affected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt, Gf


def author_texture(stage, mesh_path, texture_path, repeat=8., roughness=.32):
    import xatlas
    if not np.isfinite(repeat) or repeat <= 0 or not np.isfinite(roughness) or not 0 < roughness <= 1:
        raise ValueError('Finite positive repeat and roughness in (0, 1] required')
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    if not mesh:
        raise ValueError('Target mesh is missing')
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    points = np.asarray(mesh.GetPointsAttr().Get(), np.float32)
    if not np.all(counts == 3):
        raise ValueError('Only triangular surfaces supported')
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.uint32).reshape(-1, 3)
    mapping, indices, uv = xatlas.parametrize(points, faces)
    if not np.array_equal(mapping[indices], faces):
        raise RuntimeError('UV unwrap changed triangle correspondence')
    # Face-varying UVs preserve original vertices, FEM skinning and collisions.
    st = uv[indices].reshape(-1, 2) * float(repeat)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
    path = '/World/ReferenceLiverMaterial'
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path+'/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    surface.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.38)
    uvreader = UsdShade.Shader.Define(stage, path+'/UV')
    uvreader.CreateIdAttr('UsdPrimvarReader_float2')
    uvreader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    uvreader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    image = UsdShade.Shader.Define(stage, path+'/Color')
    image.CreateIdAttr('UsdUVTexture')
    image.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    image.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    image.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
    image.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    image.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uvreader.ConnectableAPI(), 'result')
    image.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), 'rgb')
    surface.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return {'mesh':mesh_path, 'vertices':len(points), 'triangles':len(faces),
            'uv_interpolation':'faceVarying', 'geometry_unchanged':True,
            'roughness':roughness, 'roughness_calibrated':False, 'uv_repeat':repeat}


def build(source, reference, out, ffmpeg):
    source, reference, out = Path(source).resolve(), Path(reference).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((reference/'provenance.json').read_text())
    video = next(reference.glob('videos/**/*.mp4'))
    record = next(x for x in provenance['files'] if x['path'].endswith('.mp4'))
    if hashlib.sha256(video.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Clinical reference hash mismatch')
    # Manually inspected liver ROI at t=12s, in the original 1920x1080 image.
    # Avoid instrument, HUD, visible white highlights and gallbladder.
    frame = out/'reference_frame.png'
    subprocess.run([str(ffmpeg), '-hide_banner','-loglevel','error','-ss','12','-i',str(video),
        '-frames:v','1','-y',str(frame)], check=True)
    im = Image.open(frame).convert('RGB')
    if im.size != (1920,1080):
        raise ValueError('ROI was reviewed at 1920x1080 only')
    roi = (1400,690,1800,870)
    patch = np.asarray(im.crop(roi))
    # Mirrored boundary construction is explicit and reversible; it is not
    # new tissue generation and does not claim to remove baked illumination.
    tile = np.concatenate([patch, patch[:, ::-1]], axis=1)
    tile = np.concatenate([tile, tile[::-1]], axis=0)
    Image.fromarray(tile).save(out/'liver_observed_color.png')
    stage = Usd.Stage.CreateNew(str(out/'appearance.usda'))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(source, out).replace('\\','/'))
    report = author_texture(stage, '/World/CT/liver', 'liver_observed_color.png')
    report.update(source_revision=provenance['revision'], source_video_sha256=record['sha256'],
        source_scene_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_frame_time_s=12., source_roi_xyxy=list(roi),
        attribution=provenance['attribution'], license=provenance['license'],
        measured_albedo=False, measured_brdf=False, physical_scale_calibrated=False,
        limitations=['Baked lighting remains in colour texture','Mirrored tiling can repeat visible features',
                    'No subsurface or wet-film calibration','Clinical donor differs from CT subject',
                    'Liver-only appearance experiment; not a patient-specific reconstruction'])
    stage.GetRootLayer().Save()
    (out/'appearance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=Path('.cache/bin/ffmpeg.exe'))
    a=p.parse_args();build(a.scene,a.reference,a.out,a.ffmpeg)
