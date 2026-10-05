"""Continuous authored adipose shell; geometric prototype, not measured tissue."""
import json,os,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pxr import Usd
from build_scene import mesh_prim
from boundary_surface import project_boundary
from build_fat_layer import frames,evaluate

def build(scene,surface,out):
    data=np.load(surface);parent=trimesh.Trimesh(data['points'],data['faces'],process=False)
    refined=parent.subdivide().subdivide();points=refined.vertices
    edges=refined.edges_sorted;unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=unique[counts==1]
    distance=cKDTree(points[np.unique(boundary)]).query(points)[0]
    taper=1-np.exp(-(distance/.0015)**2)
    rng=np.random.default_rng(1729)
    field=np.zeros(len(points))
    for _ in range(85):
        face=rng.choice(len(parent.faces),p=parent.area_faces/parent.area)
        center=rng.dirichlet([1,1,1])@parent.triangles[face]
        width=rng.uniform(.0015,.0035)
        field+=rng.uniform(.0003,.0008)*np.exp(-np.sum((points-center)**2,axis=1)/(2*width**2))
    # Saturation joins overlapping lobules without accumulating tall spikes.
    height=.00002+taper*(.00015+.0012*(1-np.exp(-field/.0009)))
    top=points+refined.vertex_normals*height[:,None]
    bottom=points-refined.vertex_normals*.00002
    n=len(points);faces=[refined.faces,refined.faces[:,::-1]+n]
    # Recover directed boundary edges from face winding for coherent side walls.
    directed=refined.edges[np.isin(refined.edges_unique_inverse,np.flatnonzero(np.bincount(refined.edges_unique_inverse)==1))]
    sides=[]
    for a,b in directed:sides.extend([[b,a,a+n],[b,a+n,b+n]])
    shell=trimesh.Trimesh(np.vstack([top,bottom]),np.vstack(faces+[np.array(sides)]),process=False)
    if shell.volume<0:shell.invert()
    assert shell.is_watertight and shell.is_winding_consistent
    binding,_=project_boundary(parent,shell.vertices,.01)
    base=binding.positions(parent.vertices)
    offset=np.einsum('ikj,ik->ij',frames(parent.vertices,binding.nodes),shell.vertices-base)
    error=float(np.abs(evaluate(parent.vertices,binding.nodes,binding.weights,offset)-shell.vertices).max());assert error<1e-10
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    mesh_prim(stage,'/SurgicalRegion/authored_fat_layer',shell,(.68,.48,.23),roughness=.34)
    stage.GetRootLayer().Save()
    np.savez_compressed(out/'layer.npz',points=shell.vertices,faces=shell.faces,nodes=binding.nodes,weights=binding.weights,frame_offsets=offset)
    report={'status':'unreviewed continuous visual candidate','vertices':len(shell.vertices),'triangles':len(shell.faces),'components':len(shell.split()),'watertight':bool(shell.is_watertight),'winding_consistent':bool(shell.is_winding_consistent),'height_mm_min_median_max':(np.quantile(height,[0,.5,1])*1000).tolist(),'binding_error_m':error,'source_sha256':hashlib.sha256(surface.read_bytes()).hexdigest(),'provenance':'Procedurally authored over BodyParts3D-derived membrane; DBCLS CC BY 4.0','reference':'Appendix300 sampled visual observations; no reconstructed metric geometry','physics_integrated':False,'anatomy_validated':False,'collision_checked':False,'self_intersections_checked':False,'parameters':'seed 1729; Gaussian widths 1.5-3.5 mm; saturation height 1.2 mm; all authoring choices'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--surface',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.scene,a.surface,a.out)
