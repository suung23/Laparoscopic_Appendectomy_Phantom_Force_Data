"""Authored lobulated fat layer bound to a candidate membrane (visual only).

Inspired by Appendix300 appearance, not reconstructed patient geometry or
calibrated adipose mechanics. Deterministic dimensions are authoring choices.
"""
import json,os,hashlib
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd
from boundary_surface import project_boundary
from build_scene import mesh_prim

def frames(points,faces):
    t=points[faces];a=t[:,1]-t[:,0];a/=np.linalg.norm(a,axis=1)[:,None]
    n=np.cross(a,t[:,2]-t[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    return np.stack([a,np.cross(n,a),n],axis=2)

def evaluate(parent,nodes,weights,offset):
    center=np.einsum('ij,ijk->ik',weights,parent[nodes])
    return center+np.einsum('ijk,ik->ij',frames(parent,nodes),offset)

def build(scene,surface,out):
    data=np.load(surface);parent=trimesh.Trimesh(data['points'],data['faces'],process=False)
    rng=np.random.default_rng(1729);centers=[];meshes=[]
    probability=parent.area_faces/parent.area
    for _ in range(2000):
        face=int(rng.choice(len(parent.faces),p=probability));w=rng.dirichlet([1,1,1]);c=w@parent.triangles[face]
        if centers and np.linalg.norm(np.array(centers)-c,axis=1).min()<.004:continue
        centers.append(c)
        axes=np.array([rng.uniform(.0017,.003),rng.uniform(.0013,.0023),rng.uniform(.0006,.0012)])
        shape=trimesh.creation.icosphere(subdivisions=2)
        # Low amplitude shape variation, not a measured histological pattern.
        shape.vertices*=1+.07*np.sin(shape.vertices[:,0]*6+shape.vertices[:,1]*4)[:,None]
        frame=frames(parent.vertices,parent.faces[[face]])[0]
        shape.vertices=(shape.vertices*axes)@frame.T+c+frame[:,2]*axes[2]*.35
        meshes.append(shape)
        if len(meshes)==40:break
    layer=trimesh.util.concatenate(meshes)
    binding,_=project_boundary(parent,layer.vertices,.01)
    base=binding.positions(parent.vertices)
    offset=np.einsum('ikj,ik->ij',frames(parent.vertices,binding.nodes),layer.vertices-base)
    rebuilt=evaluate(parent.vertices,binding.nodes,binding.weights,offset)
    error=float(np.abs(rebuilt-layer.vertices).max());assert error<1e-10
    rotation=trimesh.transformations.rotation_matrix(.43,[.2,.7,.3])[:3,:3];shift=np.array([.01,-.02,.03])
    moved=evaluate(parent.vertices@rotation.T+shift,binding.nodes,binding.weights,offset)
    rigid_error=float(np.abs(moved-(layer.vertices@rotation.T+shift)).max());assert rigid_error<1e-10
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    mesh_prim(stage,'/SurgicalRegion/authored_fat_lobules',layer,(.75,.43,.12),roughness=.28)
    stage.GetRootLayer().Save()
    np.savez_compressed(out/'layer.npz',points=layer.vertices,faces=layer.faces,nodes=binding.nodes,weights=binding.weights,frame_offsets=offset)
    report={'status':'unreviewed visual authoring candidate','seed':1729,'lobules':len(meshes),'vertices':len(layer.vertices),'triangles':len(layer.faces),'roundtrip_error_m':error,'rigid_motion_error_m':rigid_error,'parent_surface_sha256':hashlib.sha256(surface.read_bytes()).hexdigest(),'appearance_reference':'validation/appendix300-backbone-review.json','provenance':'Procedural authored geometry; membrane derived from BodyParts3D / DBCLS, CC BY 4.0','dimensions_calibrated':False,'contact_integrated':False,'collision_checked':False,'limitations':['Local-frame surface following is not adipose mechanics','Overlaps and seam appearance require review','No reconstructed video geometry or patient-specific claim']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--surface',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.scene,a.surface,a.out)
