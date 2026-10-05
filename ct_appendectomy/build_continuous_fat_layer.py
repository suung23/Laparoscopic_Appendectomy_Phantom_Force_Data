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

def oriented_offset(mesh,heights):
    """Reduce offsets locally until projected face area stays positive.

    Geometric authoring guard only; does not rule out nonlocal intersections.
    """
    heights=np.array(heights,float,copy=True)
    base=np.cross(mesh.triangles[:,1]-mesh.triangles[:,0],mesh.triangles[:,2]-mesh.triangles[:,0])
    target=np.sum(base*base,axis=1)*1e-4
    initial_bad=None
    for iteration in range(41):
        points=mesh.vertices+mesh.vertex_normals*heights[:,None]
        t=points[mesh.faces];area=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
        bad=np.einsum('ij,ij->i',area,base)<=target
        if initial_bad is None:initial_bad=int(bad.sum())
        if not bad.any():return points,heights,{'iterations':iteration,'initial_bad_faces':initial_bad,'remaining_bad_faces':0,'min_absolute_offset_m':float(np.abs(heights).min())}
        heights[np.unique(mesh.faces[bad])]*=.5
    raise ValueError('Offset surface orientation could not be preserved')

def build(scene,surface,out,clip_neighbors=False):
    data=np.load(surface);parent=trimesh.Trimesh(data['points'],data['faces'],process=False)
    refined=parent.subdivide().subdivide();points=refined.vertices
    edges=refined.edges_sorted;unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=unique[counts==1]
    distance=cKDTree(points[np.unique(boundary)]).query(points)[0]
    taper=1-np.exp(-(distance/.0015)**2)
    rng=np.random.default_rng(1729)
    field=np.zeros(len(points));centers=[]
    # Smooth maximum (power mean without normalization) preserves valleys,
    # unlike a saturated sum that made the entire layer nearly constant.
    for _ in range(1500):
        face=rng.choice(len(parent.faces),p=parent.area_faces/parent.area)
        center=rng.dirichlet([1,1,1])@parent.triangles[face]
        if centers and np.linalg.norm(np.array(centers)-center,axis=1).min()<.0028:continue
        centers.append(center)
        width=rng.uniform(.0009,.0018)
        bump=rng.uniform(.0005,.0014)*np.exp(-np.sum((points-center)**2,axis=1)/(2*width**2))
        field+=bump**6
        if len(centers)>=65:break
    field=field**(1/6)
    height=.00002+taper*(.00012+field)
    clipping={}
    if clip_neighbors:
        from independent_containment import contains
        from author_attachment_curve import mesh_at
        source=Usd.Stage.Open(str(scene.resolve()))
        for name in ['appendix_FJ2565','mesentery_FJ3396']:
            obstacle=mesh_at(source,'/SurgicalRegion/'+name)
            baseline_inside=contains(obstacle,points)
            first=np.full(len(points),np.inf)
            for fraction in np.linspace(.125,1,8):
                hit=contains(obstacle,points+refined.vertex_normals*(height*fraction)[:,None])
                first[(first==np.inf)&hit]=fraction
            candidates=np.flatnonzero(np.isfinite(first)&~baseline_inside)
            low=np.maximum(0,first[candidates]-.125);high=first[candidates]
            for _ in range(10):
                middle=(low+high)/2
                hit=contains(obstacle,points[candidates]+refined.vertex_normals[candidates]*(height[candidates]*middle)[:,None])
                high=np.where(hit,middle,high);low=np.where(hit,low,middle)
            before=height.copy()
            height[candidates]=np.maximum(.00002,height[candidates]*low*.95)
            clipping[name]={'changed_vertices':int(np.sum(height<before)),
                'baseline_inside_vertices':int(baseline_inside.sum()),
                'max_reduction_mm':float((before-height).max()*1000)}
    if clip_neighbors:
        # Monotone lower envelope: limit edgewise thickness gradients without
        # increasing any collision-constrained height (authoring slope 0.5).
        import heapq
        adjacency=[[] for _ in points]
        for a,b in refined.edges_unique:
            cost=.5*np.linalg.norm(points[a]-points[b])
            adjacency[a].append((b,cost));adjacency[b].append((a,cost))
        queue=[(float(h),i) for i,h in enumerate(height)];heapq.heapify(queue)
        while queue:
            value,i=heapq.heappop(queue)
            if value>height[i]:continue
            for j,cost in adjacency[i]:
                if height[j]>value+cost:
                    height[j]=value+cost;heapq.heappush(queue,(height[j],j))
    top,height,top_guard=oriented_offset(refined,height)
    bottom,bottom_height,bottom_guard=oriented_offset(refined,np.full(len(points),-.00002))
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
    np.savez_compressed(out/'layer.npz',points=shell.vertices,faces=shell.faces,nodes=binding.nodes,weights=binding.weights,frame_offsets=offset,height=height,lobe_centers=np.array(centers))
    report={'status':'unreviewed continuous visual candidate','vertices':len(shell.vertices),'triangles':len(shell.faces),'components':len(shell.split()),'watertight':bool(shell.is_watertight),'winding_consistent':bool(shell.is_winding_consistent),'height_mm_min_median_max':(np.quantile(height,[0,.5,1])*1000).tolist(),'binding_error_m':error,'source_sha256':hashlib.sha256(surface.read_bytes()).hexdigest(),'provenance':'Procedurally authored over BodyParts3D-derived membrane; DBCLS CC BY 4.0','reference':'Appendix300 sampled visual observations; no reconstructed metric geometry','physics_integrated':False,'anatomy_validated':False,'collision_checked':False,'self_intersections_checked':False,'parameters':'seed 1729; 2.8 mm seed spacing; Gaussian widths 0.9-1.8 mm; sixth-power smooth maximum; all authoring choices','lobe_centers':len(centers),'neighbor_clipping':clipping,'top_offset_guard':top_guard,'bottom_offset_guard':bottom_guard,'clipping_scope':'Eight normal-line samples and ten bisections; not continuous collision detection; pre-existing membrane penetrations remain'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--surface',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--clip-neighbors',action='store_true')
    a=p.parse_args();build(a.scene,a.surface,a.out,a.clip_neighbors)
