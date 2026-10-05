"""Reference-informed authored tissue appearance, not reconstruction or albedo fitting.

Synthetic vascular motifs are appearance detail, never segmented vessels or
collision geometry. Keeps mesh topology and physics untouched for comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pxr import Usd, UsdShade, UsdLux, Sdf
from reference_appearance import author_texture


def texture(folder, name, color, seed, vessels=True):
    rng=np.random.default_rng(seed); size=1024
    noise=np.zeros((size,size),float)
    for resolution, amplitude in ((8,.08),(32,.035),(128,.012)):
        small=Image.fromarray(np.uint8(rng.uniform(0,255,(resolution,resolution))))
        noise+=amplitude*(np.asarray(small.resize((size,size),Image.Resampling.BICUBIC))/255-.5)
    rgb=np.clip(np.asarray(color)[None,None,:]+noise[:,:,None],0,1)
    mask=Image.new('L',(size,size)); draw=ImageDraw.Draw(mask)
    if vessels:
        def branch(x,y,angle,length,width,depth):
            points=[(x,y)]
            for _ in range(8):
                angle+=rng.normal(0,.11); x+=length/8*np.cos(angle); y+=length/8*np.sin(angle)
                points.append((x,y))
            draw.line(points,fill=int(95+depth*14),width=max(1,int(width)))
            if depth:
                for turn in (-.55,.6): branch(x,y,angle+turn,length*.62,width*.60,depth-1)
        for _ in range(6): branch(rng.uniform(100,924),rng.uniform(100,924),rng.uniform(0,6.28),rng.uniform(110,200),3,3)
    vascular=np.asarray(mask.filter(ImageFilter.GaussianBlur(.8)))/255
    vein=np.array([.43,.13,.16])
    rgb=rgb*(1-vascular[:,:,None]*.65)+vein*vascular[:,:,None]*.65
    Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(folder/(name+'_color.png'))
    rough=np.clip(.25+noise*.7,.16,.34)
    Image.fromarray(np.uint8(rough*255)).save(folder/(name+'_roughness.png'))


def build(source,out,reference):
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads(reference.read_text(encoding='utf-8'))
    reference_root=reference.parent
    for item in provenance['files']:
        if hashlib.sha256((reference_root/item['local_name']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Clinical reference changed')
    stage=Usd.Stage.CreateNew(str((out/'appearance.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(source.resolve(),out.resolve()).replace('\\','/'))
    groups=[('bowel',(.68,.43,.40),True,['/World/CT/colon','/World/CT/small_bowel']),
            ('appendix',(.70,.40,.39),True,['/World/Atlas/appendix/render']),
            ('mesentery',(.72,.56,.32),True,['/World/Atlas/mesoappendix/render']),
            ('lining',(.60,.37,.34),True,['/World/BodyEnvelope/render'])]
    records=[]
    for index,(name,color,vessels,paths) in enumerate(groups):
        texture(out,name,color,200+index,vessels)
        for n,path in enumerate(paths):
            if not stage.GetPrimAtPath(path): continue
            material=f'/World/Appendix300Look/{name}_{n}'
            records.append(author_texture(stage,path,name+'_color.png',repeat=1.,roughness=.25,material_path=material))
            shader=UsdShade.Shader(stage.GetPrimAtPath(material+'/Surface'))
            rough=UsdShade.Shader.Define(stage,material+'/Roughness')
            rough.CreateIdAttr('UsdUVTexture')
            rough.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(name+'_roughness.png'))
            rough.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
            rough.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(UsdShade.Shader(stage.GetPrimAtPath(material+'/UV')).ConnectableAPI(),'result')
            rough.CreateOutput('r',Sdf.ValueTypeNames.Float)
            shader.GetInput('roughness').ConnectToSource(rough.ConnectableAPI(),'r')
    fill=UsdLux.DomeLight(stage.GetPrimAtPath('/World/Fill'))
    if fill: fill.GetIntensityAttr().Set(3.)
    stage.GetRootLayer().Save()
    report=dict(reference=provenance,source_scene=str(source),meshes=records,
        status='unreviewed_visual_candidate',measured_albedo=False,measured_brdf=False,
        segmented_vessels=False,geometry_reconstructed=False,
        limitations=['Authored colours and vascular motifs, informed by inspected references, not recovered patient anatomy',
            'UV chart seams and vessel continuity require rendered review','Geometry and existing registration problems unchanged',
            'No new tool mechanics or FEM integration in this appearance layer'])
    (out/'appearance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    build(a.source,a.out,a.reference)
