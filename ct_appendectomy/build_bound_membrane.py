"""Build a boundary-driven mesoappendix authoring candidate in native anatomy.

Two heuristic atlas curves are bound to their respective parent surfaces.
This does not establish the true attachment footprint or an accurate membrane.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from author_attachment_curve import mesh_at,edge_path,resample_path
from boundary_surface import SurfaceBinding,project_boundary,coons_surface
from build_scene import mesh_prim


def build(scene,profile,out):
    stage=Usd.Stage.Open(str(scene.resolve()))
    data=json.loads(profile.read_text())
    appendix=mesh_at(stage,data['source_prim'])
    for key,array in [('parent_vertices_sha256',appendix.vertices),('parent_faces_sha256',appendix.faces)]:
        if hashlib.sha256(array.tobytes()).hexdigest()!=data[key]:raise ValueError('Attachment source ordering mismatch')
    first=SurfaceBinding(data['nodes'],data['weights'],data['parent_vertex_count'])
    bottom=first.positions(appendix.vertices)
    mesentery_path='/SurgicalRegion/mesentery_FJ3396'
    parent=mesh_at(stage,mesentery_path)
    envelope=mesh_at(stage,'/SurgicalRegion/mesoappendix_FJ3397')
    # Refine only the overlapping source patch for search: interpolation leaves
    # the original planar triangles unchanged. Final bindings use original IDs.
    tri=parent.triangles
    keep=((tri.max(axis=1)>=envelope.bounds[0]-.003)&
          (tri.min(axis=1)<=envelope.bounds[1]+.003)).all(axis=1)
    local=parent.submesh([np.flatnonzero(keep)],append=True)
    vertices,faces=trimesh.remesh.subdivide_to_size(local.vertices,local.faces,max_edge=.001,max_iter=8)
    search=trimesh.Trimesh(vertices,faces,process=True)
    _,near,_=trimesh.proximity.closest_point(envelope,search.vertices)
    # Explicit authoring corridor, not an anatomical tolerance. Reject paths
    # outside the native envelope neighborhood or within adjacent ileum.
    allowed=near<=.003
    ileum=mesh_at(stage,'/SurgicalRegion/ileum_FJ2599')
    allowed &= ~ileum.contains(search.vertices)
    ids=np.flatnonzero(allowed)
    if not len(ids):raise ValueError('No feasible mesenteric attachment region')
    endpoints=[int(ids[np.argmin(np.linalg.norm(search.vertices[ids]-point,axis=1))])
               for point in bottom[[0,-1]]]
    if endpoints[0]==endpoints[1]:
        raise ValueError("Both endpoints collapse to one attachment vertex; author a root footprint instead of forcing a strip")
    path=edge_path(search,near,*endpoints,allowed_vertices=allowed)
    landmarks,length=resample_path(search.vertices[path],len(bottom))
    original_root_length=float(length)
    accepted_smoothing_steps=0
    for _ in range(16):
        proposal=landmarks.copy()
        proposal[1:-1]=.5*landmarks[1:-1]+.25*(landmarks[:-2]+landmarks[2:])
        proposal,_,_=trimesh.proximity.closest_point(parent,proposal)
        proposal[[0,-1]]=landmarks[[0,-1]]
        # Sample segment interiors too: endpoint feasibility alone is insufficient.
        probes=np.concatenate([proposal]+[(1-t)*proposal[:-1]+t*proposal[1:] for t in (.25,.5,.75)])
        _,dist,_=trimesh.proximity.closest_point(envelope,probes)
        if (dist>.003).any() or ileum.contains(probes).any():break
        landmarks=proposal;accepted_smoothing_steps+=1
    length=float(np.linalg.norm(np.diff(landmarks,axis=0),axis=1).sum())
    second,projection_error=project_boundary(parent,landmarks,1e-7)
    top=second.positions(parent.vertices)
    v=np.linspace(0,1,25)[:,None]
    # Author a positive outward departure at the appendix boundary. This is
    # a geometric Hermite tangent condition, not a tissue constitutive law.
    normals=np.einsum('ij,ijk->ik',first.weights,appendix.vertex_normals[first.nodes])
    norms=np.linalg.norm(normals,axis=1)
    if (norms<1e-10).any():raise ValueError('Undefined attachment normal')
    normals/=norms[:,None]
    span=top-bottom
    target=.1*np.linalg.norm(span,axis=1)
    correction=np.maximum(0,target-np.einsum('ij,ij->i',span,normals))[:,None]*normals
    curve=(1-v[:,:,None])*bottom[None]+v[:,:,None]*top[None]+(v*(1-v)**2)[:,:,None]*correction[None]
    left=curve[:,0];right=curve[:,-1]
    surface=coons_surface(bottom,top,left,right)
    # Coons captures edge corrections; supply the remaining interior offset.
    offset=curve-surface.vertices.reshape(curve.shape)
    offset[[0,-1]]=0;offset[:,[0,-1]]=0
    surface=coons_surface(bottom,top,left,right,interior_offset=offset)
    positions=surface.vertices.reshape(len(v),len(bottom),3)
    np.testing.assert_allclose(positions[0],bottom,rtol=0,atol=1e-10)
    np.testing.assert_allclose(positions[-1],top,rtol=0,atol=1e-10)
    collisions={};uncheckable=[]
    # Centroid containment is only a coarse penetration diagnostic. Edge/face
    # intersection and open-surface contact remain separate requirements.
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName()=='mesoappendix_FJ3397':continue
        other=mesh_at(stage,str(prim.GetPath()))
        if not other.is_watertight:
            uncheckable.append(str(prim.GetPath()));continue
        inside=other.contains(surface.triangles_center)
        if inside.any():
            _,distance,_=trimesh.proximity.closest_point(other,surface.triangles_center[inside])
            deep=distance>1e-4
            if deep.any():collisions[str(prim.GetPath())]=dict(centroids_inside_over_0_1mm=int(deep.sum()),max_depth_mm=float(distance.max()*1000))
    out.mkdir(parents=True,exist_ok=True)
    result=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    result.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    mesh_prim(result,'/SurgicalRegion/mesoappendix_FJ3397',surface,(.72,.56,.32))
    result.GetRootLayer().Save()
    np.savez_compressed(out/'surface.npz',points=surface.vertices,faces=surface.faces)
    report=dict(status=('rejected: organ penetration detected' if collisions else 'unreviewed boundary-driven membrane candidate'),
        source_scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        appendix_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        second_parent=mesentery_path,second_parent_vertex_count=len(parent.vertices),
        second_parent_vertices_sha256=hashlib.sha256(parent.vertices.tobytes()).hexdigest(),
        second_parent_faces_sha256=hashlib.sha256(parent.faces.tobytes()).hexdigest(),
        second_binding_nodes=second.nodes.tolist(),second_binding_weights=second.weights.tolist(),
        second_rest_curve_m=top.tolist(),search_path_vertex_indices=path.tolist(),search_vertex_count=len(search.vertices),
        search_max_edge_m=.001,search_allowed_vertices=int(allowed.sum()),
        second_curve_length_mm=float(length*1000),
        root_length_before_smoothing_mm=original_root_length*1000,root_smoothing_steps=accepted_smoothing_steps,max_projection_error_m=float(projection_error.max()),
        vertices=len(surface.vertices),triangles=len(surface.faces),
        connected_components=len(surface.split(only_watertight=False)),winding_consistent=bool(surface.is_winding_consistent),
        outward_departure_ratio=.1,
        min_departure_normal_dot_m=float(np.einsum('ij,ij->i',span+correction,normals).min()),
        boundary_position_error_m=float(max(np.abs(positions[0]-bottom).max(),np.abs(positions[-1]-top).max())),
        penetrations=collisions,open_parent_surfaces_not_checked=uncheckable,
        anatomy_accepted=False,thickness_calibrated=False,physics_integrated=False,self_intersections_tested=False,
        assumptions=['Mesenteric endpoints selected within 3 mm of native meso envelope, outside ileum; unvalidated authoring corridor',
            'Mesenteric path edge cost biased toward native mesoappendix envelope',
            'Free edges and interior use outward Hermite departure; ratio 0.1 is an authoring choice, not measured tissue shape'],
        limitations=['Candidate may cross organs despite being topologically connected',
            'Centroid containment is not exhaustive collision detection',
            'Both attachment footprints require anatomical review'],
        attribution='BodyParts3D, Database Center for Life Science; CC BY 4.0 per native assembly provenance')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['vertices','triangles','connected_components','boundary_position_error_m','second_curve_length_mm','penetrations','open_parent_surfaces_not_checked']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=Path('out/ct_appendectomy/native_source_review/review.usda'))
    p.add_argument('--profile',type=Path,default=Path('ct_appendectomy/profiles/native_appendix_attachment_candidate.json'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/bound_membrane_v1'))
    a=p.parse_args();build(a.scene,a.profile,a.out)
