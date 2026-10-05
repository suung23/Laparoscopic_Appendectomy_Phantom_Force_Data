"""Extract a candidate midsurface from a closed atlas envelope.

PCA-axis paired intersections are an authoring approximation, not measured
histological layers. Rays with multiple intervals are rejected, not bridged.
The output has no assigned thickness, density, force law or fat reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from build_scene import mesh_prim


def extract(mesh, spacing=.00075):
    if not np.isfinite(spacing) or spacing<=0:raise ValueError('Positive finite spacing required')
    if not mesh.is_watertight:raise ValueError('Closed envelope required')
    center=mesh.vertices.mean(axis=0)
    _,_,axes=np.linalg.svd(mesh.vertices-center,full_matrices=False)
    # Deterministic axis signs, with a right-handed frame.
    for k in (0,1):
        if axes[k,np.argmax(np.abs(axes[k]))]<0:axes[k]*=-1
    axes[2]=np.cross(axes[0],axes[1])
    local=(mesh.vertices-center)@axes.T
    lo,hi=local.min(axis=0),local.max(axis=0)
    x=np.arange(lo[0]+spacing/2,hi[0],spacing)
    y=np.arange(lo[1]+spacing/2,hi[1],spacing)
    if len(x)*len(y)>250000:raise ValueError('Grid exceeds authoring memory limit')
    if min(len(x),len(y))<2:raise ValueError('Spacing too coarse for surface')
    xx,yy=np.meshgrid(x,y)
    origins=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,lo[2]-.001)]@axes+center
    locations,rays,_=mesh.ray.intersects_location(origins,np.tile(axes[2],(len(origins),1)),multiple_hits=True)
    depth=(locations-center)@axes[2]
    order=np.lexsort((depth,rays));rays,depth=rays[order],depth[order]
    groups={}
    for ray,value in zip(rays,depth):
        values=groups.setdefault(int(ray),[])
        if not values or value-values[-1]>1e-8:values.append(float(value))
    points=np.full((len(origins),3),np.nan);chord=np.full(len(origins),np.nan)
    multi=[];odd=[]
    for ray,values in groups.items():
        if len(values)!=2:
            (multi if len(values)>2 else odd).append(ray);continue
        if values[1]-values[0]<=1e-8:continue
        z=(values[0]+values[1])/2
        points[ray]=np.array([xx.ravel()[ray],yy.ravel()[ray],z])@axes+center
        chord[ray]=values[1]-values[0]
    valid=np.isfinite(chord)
    paired_indices=np.flatnonzero(valid)
    outside=paired_indices[~mesh.contains(points[paired_indices])]
    valid[outside]=False
    faces=[]
    for row in range(len(y)-1):
        for col in range(len(x)-1):
            a=row*len(x)+col;b=a+1;c=a+len(x);d=c+1
            for triangle in ((a,b,d),(a,d,c)):
                if valid[list(triangle)].all():faces.append(triangle)
    if not faces:raise ValueError('No unambiguous midsurface faces')
    faces=np.asarray(faces,dtype=int);used=np.unique(faces)
    remap=np.full(len(points),-1);remap[used]=np.arange(len(used))
    surface=trimesh.Trimesh(points[used],remap[faces],process=False)
    data=dict(grid_shape=[len(y),len(x)],grid_spacing_m=spacing,
        rays=len(origins),intersected_rays=len(groups),paired_rays=int(valid.sum()),multiple_intervals=len(multi),odd_intersections=len(odd),
        accepted_fraction_of_intersected_rays=float(valid.sum()/max(len(groups),1)),
        rejected_outside_midpoints=len(outside),
        supported_projected_area_fraction=float(valid.mean()),
        axis_chord_mm_p05_p50_p95=(np.quantile(chord[used],[.05,.5,.95])*1000).tolist(),
        vertices=len(surface.vertices),triangles=len(surface.faces),
        connected_surface_components=len(surface.split(only_watertight=False)),
        boundary_edges=int(np.sum(np.bincount(surface.edges_unique_inverse)==1)),
        frame_axes=axes.tolist(),frame_center_m=center.tolist())
    arrays=dict(points=surface.vertices,faces=surface.faces,source_grid_indices=used,
        source_axis_chord_m=chord[used],rejected_multiple_interval_rays=np.asarray(multi,dtype=int),
        rejected_outside_midpoint_rays=outside,
        rejected_odd_intersection_rays=np.asarray(odd,dtype=int))
    return surface,arrays,data


def build(source,out,spacing):
    stage=Usd.Stage.Open(str(source))
    path='/SurgicalRegion/mesoappendix_FJ3397'
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
    source_mesh=trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
    surface,arrays,report=extract(source_mesh,spacing)
    out.mkdir(parents=True,exist_ok=True)
    result=Usd.Stage.CreateNew(str((out/'meso_midsurface.usda').resolve()))
    root=UsdGeom.Xform.Define(result,'/MesoappendixCandidate');result.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(result,1.);UsdGeom.SetStageUpAxis(result,'Z')
    mesh_prim(result,'/MesoappendixCandidate/surface',surface,(.72,.56,.42))
    result.GetRootLayer().Save();np.savez_compressed(out/'surface.npz',**arrays)
    report.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_prim=path,
        status='authoring candidate; not anatomical or physical validation',
        shell_thickness_m=None,calibrated=False,fat_reconstructed=False,physics_integrated=False,
        limitations=['PCA projection can lose folds; rejected rays are retained in NPZ',
            'Grid sampling loses narrow features and trims boundaries',
            'Envelope chord includes all atlas volume, not anatomical membrane thickness',
            'Midsurface has open boundaries and no material thickness or contact model',
            'No replacement of the production simulator geometry'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('out/ct_appendectomy/native_source_complex/surgical_region.usda'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/meso_midsurface_v1'))
    p.add_argument('--spacing',type=float,default=.00075)
    a=p.parse_args();build(a.source,a.out,a.spacing)
