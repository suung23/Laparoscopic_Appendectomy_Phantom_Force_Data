"""Propose an atlas-side attachment curve for review, never anatomical ground truth.

The path follows appendix mesh edges, biased toward the existing mesoappendix.
Endpoints and bias are engineering heuristics. Saved barycentric bindings avoid
re-projecting or jumping between surface triangles during later deformation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from pxr import Usd,UsdGeom,Gf,Vt
from boundary_surface import project_boundary


def mesh_at(stage,path):
    m=UsdGeom.Mesh(stage.GetPrimAtPath(path))
    return trimesh.Trimesh(np.asarray(m.GetPointsAttr().Get()),
        np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)


def edge_path(mesh,near_distance,start,end,bias_scale=.002):
    if not np.isfinite(bias_scale) or bias_scale<=0:raise ValueError('Positive finite bias scale required')
    edges=mesh.edges_unique
    length=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    weights=length*(1+(np.asarray(near_distance)[edges].mean(axis=1)/bias_scale)**2)
    graph=coo_matrix((np.r_[weights,weights],
        (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(mesh.vertices),)*2).tocsr()
    distance,previous=dijkstra(graph,indices=start,return_predecessors=True)
    if not np.isfinite(distance[end]):raise ValueError('Attachment endpoints are disconnected')
    path=[end]
    while path[-1]!=start:
        path.append(int(previous[path[-1]]))
        if path[-1]<0 or len(path)>len(mesh.vertices):raise ValueError('Invalid predecessor chain')
    return np.asarray(path[::-1],dtype=int)


def resample_path(points,count):
    if count<2:raise ValueError('At least two samples required')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<=0 or (np.diff(arc)<=0).any():raise ValueError('Nonzero path edges required')
    samples=np.linspace(0,arc[-1],count)
    return np.column_stack([np.interp(samples,arc,points[:,axis]) for axis in range(3)]),arc[-1]


def build(scene,out):
    stage=Usd.Stage.Open(str(scene.resolve()))
    path='/SurgicalRegion/appendix_FJ2565'
    appendix=mesh_at(stage,path)
    meso=mesh_at(stage,'/SurgicalRegion/mesoappendix_FJ3397')
    colon=mesh_at(stage,'/SurgicalRegion/ascending_colon_FJ2566')
    _,near_meso,_=trimesh.proximity.closest_point(meso,appendix.vertices)
    _,near_colon,_=trimesh.proximity.closest_point(colon,appendix.vertices)
    start=int(np.argmin(near_colon))
    end=int(np.argmax(np.linalg.norm(appendix.vertices-appendix.vertices[start],axis=1)))
    indices=edge_path(appendix,near_meso,start,end)
    samples,length=resample_path(appendix.vertices[indices],65)
    binding,projection=project_boundary(appendix,samples,1e-7)
    bound=binding.positions(appendix.vertices)
    # These are serialization/correspondence invariants, not tissue validation.
    np.testing.assert_allclose(bound,samples,rtol=0,atol=1e-9)
    _,meso_distance,_=trimesh.proximity.closest_point(meso,bound)
    out.mkdir(parents=True,exist_ok=True)
    report=dict(status='unreviewed heuristic attachment curve',source_prim=path,
        attribution='BodyParts3D, Database Center for Life Science; CC BY 4.0 per native assembly provenance',
        source_license_record='validation/native-source-complex.json',
        source_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        parent_vertices_sha256=hashlib.sha256(appendix.vertices.tobytes()).hexdigest(),
        parent_faces_sha256=hashlib.sha256(appendix.faces.tobytes()).hexdigest(),
        parent_vertex_count=len(appendix.vertices),nodes=binding.nodes.tolist(),weights=binding.weights.tolist(),
        rest_positions_m=bound.tolist(),source_edge_path=indices.tolist(),arc_length_mm=float(length*1000),
        meso_distance_mm_min_median_max=(np.quantile(meso_distance,[0,.5,1])*1000).tolist(),
        max_projection_error_m=float(projection.max()),
        heuristics={'base':'appendix vertex nearest native ascending-colon mesh',
            'distal':'appendix vertex farthest in Euclidean distance from selected base',
            'edge_cost':'edge length times (1 + (mean meso distance / 2 mm)^2)'},
        clinical_attachment_validated=False,force_coupling_integrated=False,
        limitations=['Endpoint and curve location need anatomical review',
            'Parent vertex/triangle ordering must be preserved; binds to this atlas mesh only',
            'Not a tissue junction, constraint, or surgical success result'])
    (out/'attachment_curve.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    result=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    result.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    curve=UsdGeom.BasisCurves.Define(result,'/World/AttachmentReview/Curve')
    curve.CreateTypeAttr('linear');curve.CreateWrapAttr('nonperiodic')
    curve.CreateCurveVertexCountsAttr([len(bound)]);curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(bound.astype(np.float32)))
    curve.CreateWidthsAttr([.0006]);curve.SetWidthsInterpolation('constant')
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.,1.,1.)])
    result.GetRootLayer().Save()
    print(json.dumps({k:report[k] for k in ('arc_length_mm','meso_distance_mm_min_median_max','max_projection_error_m','status')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('out/ct_appendectomy/native_source_review/review.usda'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/attachment_curve_v1'))
    a=p.parse_args();build(a.scene,a.out)
