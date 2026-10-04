"""Offline quality-meshing experiment; not a simulator runtime dependency."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--mesher-path',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sys.path.append(str(a.mesher_path.resolve()))
import tetgen
a.out.mkdir(parents=True,exist_ok=True)
stage=Usd.Stage.Open(str(a.scene))
mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Atlas/appendix/render'))
points=np.asarray(mesh.GetPointsAttr().Get(),float)
matrix=np.asarray(mesh.ComputeLocalToWorldTransform(0))
points=(np.column_stack([points,np.ones(len(points))])@matrix)[:,:3]
if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get())==3):raise ValueError('Triangular surface required')
faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
source=trimesh.Trimesh(points,faces,process=False)
if not source.is_watertight or not source.is_winding_consistent:raise ValueError('Closed consistently oriented source required')
generator=tetgen.TetGen(points,faces)
options=dict(order=1,minratio=1.5,mindihedral=10.,steinerleft=50000,quality=True,
             nomergefacet=True,nomergevertex=True)
result=generator.tetrahedralize(**options)
nodes,cells=np.asarray(result[0]),np.asarray(result[1])
q=nodes[cells];vol=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6
negative=vol<0
cells[negative]=cells[negative][:,[1,0,2,3]]
edges=sum(np.sum((q[:,i]-q[:,j])**2,axis=1) for i in range(4) for j in range(i))
quality=12*(3*np.abs(vol))**(2/3)/edges
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from surface_target import TissueSurface
surface=TissueSurface(cells)
boundary_nodes=np.unique(surface.faces)
scale=float(np.ptp(points,axis=0).max());origin=points.mean(0)
normalized=trimesh.Trimesh((points-origin)/scale,faces,process=False)
_,dist,_=trimesh.proximity.closest_point(normalized,(nodes[boundary_nodes]-origin)/scale)
dist*=scale
boundary=trimesh.Trimesh((nodes-origin)/scale,surface.faces,process=False)
boundary.remove_unreferenced_vertices()
source_samples=np.concatenate([points,points[faces].mean(1)])
_,reverse_dist,_=trimesh.proximity.closest_point(boundary,(source_samples-origin)/scale)
reverse_dist*=scale
boundary_centers=nodes[surface.faces].mean(1)
_,center_dist,_=trimesh.proximity.closest_point(normalized,(boundary_centers-origin)/scale)
center_dist*=scale
boundary_closed=bool(boundary.is_watertight and boundary.is_winding_consistent)
duplicates=len(cells)-len(np.unique(np.sort(cells,axis=1),axis=0))
volume_error=abs(np.abs(vol).sum()-abs(source.volume))/abs(source.volume)
report=dict(status='geometric_candidate' if not duplicates and np.all(np.abs(vol)>1e-15) and max(dist.max(),reverse_dist.max(),center_dist.max())<1e-6 and volume_error<1e-5 and boundary_closed else 'failed',
    source_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),tool='tetgen',version=tetgen.__version__,
    license='Python wrapper MIT; underlying TetGen AGPLv3/commercial. Offline experiment, not linked into simulator.',
    license_reference='https://tetgen.pyvista.org/index.html',options=options,
    nodes=len(nodes),tetrahedra=len(cells),duplicate_tetrahedra=duplicates,
    boundary_node_to_source_max_m=float(dist.max()),relative_volume_error=float(volume_error),
    source_vertex_and_face_center_to_boundary_max_m=float(reverse_dist.max()),
    boundary_face_center_to_source_max_m=float(center_dist.max()),
    boundary_closed_and_consistent=boundary_closed,
    mean_ratio_min=float(quality.min()),mean_ratio_percentiles_1_5_50=np.percentile(quality,[1,5,50]).tolist(),
    cells_below_mean_ratio_0_1=int((quality<.1).sum()),
    limitations=['Bidirectional vertex and face-centroid samples, not a continuous Hausdorff proof.','No physics, material, attachment or force validation yet.'])
np.savez_compressed(a.out/'mesh.npz',points=nodes,tets=cells)
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
