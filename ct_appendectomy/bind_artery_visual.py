"""Bind original artery to authored membrane for visual following only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd
from author_attachment_curve import mesh_at
from boundary_surface import project_boundary
from build_fat_layer import frames,evaluate

def build(scene,parent_file,out):
    stage=Usd.Stage.Open(str(scene.resolve()));vessel=mesh_at(stage,'/SurgicalRegion/appendicular_artery_FJ3410');d=np.load(parent_file);parent=trimesh.Trimesh(d['points'],d['faces'],process=False)
    binding,distance=project_boundary(parent,vessel.vertices,.1)
    offset=np.einsum('ikj,ik->ij',frames(parent.vertices,binding.nodes),vessel.vertices-binding.positions(parent.vertices))
    restored=evaluate(parent.vertices,binding.nodes,binding.weights,offset);error=float(np.abs(restored-vessel.vertices).max());assert error<1e-10
    follow=np.clip((.01-distance)/.005,0,1);follow=follow**2*(3-2*follow)
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'binding.npz',points=vessel.vertices,faces=vessel.faces,nodes=binding.nodes,weights=binding.weights,frame_offsets=offset,follow_weight=follow)
    report={'status':'unreviewed visual binding','vertices':len(vessel.vertices),'fixed_vertices':int((follow==0).sum()),'full_follow_vertices':int((follow==1).sum()),'blend_policy':'full following within 5 mm, fixed beyond 10 mm; heuristic smooth blend','max_parent_distance_mm':float(distance.max()*1000),'roundtrip_error_m':error,'parent_sha256':hashlib.sha256(parent_file.read_bytes()).hexdigest(),'source_scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'attribution':'BodyParts3D / DBCLS CC BY 4.0; authored membrane association','limitations':['Nearest-surface association is heuristic, not anatomical attachment validation','No vessel elasticity, blood flow, cutting or endpoint anchoring','Cross-section distortion under membrane deformation requires review']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.scene,a.parent,a.out)
