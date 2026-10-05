"""Inspect midsurface boundaries and replace only the mesoappendix in a review layer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from build_scene import mesh_prim


def build(scene,surface_file,out):
    source=Usd.Stage.Open(str(scene.resolve()))
    data=np.load(surface_file)
    surface=trimesh.Trimesh(data['points'],data['faces'],process=False)
    counts=np.bincount(surface.edges_unique_inverse)
    edges=surface.edges_unique[counts==1]
    degree=np.bincount(edges.ravel(),minlength=len(surface.vertices))
    groups=trimesh.graph.connected_components(edges,min_len=2)
    records=[]
    for group in groups:
        own_edges=edges[np.isin(edges[:,0],group)]
        length=np.linalg.norm(surface.vertices[own_edges[:,1]]-surface.vertices[own_edges[:,0]],axis=1).sum()
        records.append(dict(vertices=len(group),closed_cycle=bool(np.all(degree[group]==2)),
            perimeter_mm=float(length*1000),vertex_indices=group.tolist()))
    proximity={}
    boundary=np.unique(edges)
    for name in ('appendix_FJ2565','ascending_colon_FJ2566','appendicular_artery_FJ3410'):
        usd=UsdGeom.Mesh(source.GetPrimAtPath('/SurgicalRegion/'+name))
        mesh=trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()),
            np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
        _,distance,_=trimesh.proximity.closest_point(mesh,surface.vertices[boundary])
        proximity[name]=dict(distance_mm_min_p05_p50_p95=(np.quantile(distance,[0,.05,.5,.95])*1000).tolist(),
            boundary_vertices_within_0_75_mm=int(np.sum(distance<.00075)),
            interpretation='Geometric proximity only; no tissue attachment inferred')
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    mesh_prim(stage,'/SurgicalRegion/mesoappendix_FJ3397',surface,(.72,.56,.32))
    stage.GetRootLayer().Save()
    report=dict(source_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        surface_sha256=hashlib.sha256(surface_file.read_bytes()).hexdigest(),
        boundary_components=records,boundary_proximity=proximity,
        nonmanifold_edges=int(np.sum(counts>2)),branched_boundary_vertices=int(np.sum(degree>2)),
        camera_and_other_anatomy_unchanged=True,
        scope='Only the mesoappendix envelope is replaced with a zero-thickness authoring surface',
        thickness_calibrated=False,fat_reconstructed=False,physics_integrated=False,
        acceptance='Requires visual boundary and anatomy review; proximity is not anatomical validation')
    (out/'boundary_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='boundary_components'},indent=2))
    print('Boundary components:',[(r['vertices'],r['closed_cycle'],r['perimeter_mm']) for r in records])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('out/ct_appendectomy/native_source_review/review.usda'))
    p.add_argument('--surface',type=Path,default=Path('out/ct_appendectomy/meso_midsurface_v1/surface.npz'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/meso_midsurface_review'))
    a=p.parse_args();build(a.scene,a.surface,a.out)
