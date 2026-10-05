"""Explicit surface boundaries and work-conjugate parent-mesh bindings.

An authoring/coupling primitive, not an anatomical attachment detector or a
constitutive/contact model. Landmark selection requires separate review.
"""
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class SurfaceBinding:
    nodes: np.ndarray
    weights: np.ndarray
    parent_vertex_count: int

    def __post_init__(self):
        raw=np.asarray(self.nodes)
        if raw.dtype.kind not in 'iu':raise ValueError('Integer parent node indices required')
        if isinstance(self.parent_vertex_count,bool) or not isinstance(self.parent_vertex_count,(int,np.integer)) or self.parent_vertex_count<1:
            raise ValueError('Positive integer parent vertex count required')
        self.nodes=np.array(self.nodes,dtype=int,copy=True)
        self.weights=np.array(self.weights,dtype=float,copy=True)
        if self.nodes.ndim!=2 or self.nodes.shape[1]!=3 or self.weights.shape!=self.nodes.shape:
            raise ValueError('Expected three nodes and weights per boundary point')
        if (self.nodes<0).any() or (self.nodes>=self.parent_vertex_count).any():
            raise ValueError('Parent node out of range')
        if (not np.isfinite(self.weights).all() or (self.weights < -1e-9).any()
                or not np.allclose(self.weights.sum(axis=1),1.,rtol=0,atol=1e-9)):
            raise ValueError('Convex barycentric coordinates required')

    def positions(self,parent):
        parent=np.asarray(parent,float)
        if parent.shape!=(self.parent_vertex_count,3) or not np.isfinite(parent).all():
            raise ValueError('Finite parent coordinates with unchanged vertex ordering required')
        return np.einsum('ij,ijk->ik',self.weights,parent[self.nodes])

    def parent_forces(self,boundary_forces):
        force=np.asarray(boundary_forces,float)
        if force.shape!=(len(self.nodes),3) or not np.isfinite(force).all():
            raise ValueError('One finite force vector per boundary point required')
        result=np.zeros((self.parent_vertex_count,3))
        np.add.at(result,self.nodes.ravel(),(self.weights[:,:,None]*force[:,None,:]).reshape(-1,3))
        return result


def project_boundary(mesh,landmarks,max_distance):
    if not np.isfinite(max_distance) or max_distance<=0:raise ValueError('Finite positive projection limit required')
    landmarks=np.asarray(landmarks,float)
    if landmarks.ndim!=2 or landmarks.shape[1]!=3 or not np.isfinite(landmarks).all():
        raise ValueError('Finite 3D landmarks required')
    # Trimesh's absolute tolerances can misclassify small triangles in metres.
    # Normalize only the query copy, retaining original face/node ordering.
    origin=mesh.bounds.mean(axis=0)
    scale=1./float(np.ptp(mesh.vertices,axis=0).max())
    if not np.isfinite(scale):raise ValueError('Nonzero parent extent required')
    query=trimesh.Trimesh((mesh.vertices-origin)*scale,mesh.faces,process=False)
    local,distance,faces=trimesh.proximity.closest_point(query,(landmarks-origin)*scale)
    projected=local/scale+origin;distance=distance/scale
    if np.any(distance>max_distance):raise ValueError('Landmark exceeds projection limit; review attachment location')
    weights=trimesh.triangles.points_to_barycentric(query.triangles[faces],local)
    binding=SurfaceBinding(mesh.faces[faces],weights,len(mesh.vertices))
    np.testing.assert_allclose(binding.positions(mesh.vertices),projected,rtol=0,atol=1e-10)
    return binding,distance


def coons_surface(bottom,top,left,right,interior_offset=None):
    """Transfinite interpolation that preserves all four supplied boundaries.

    Curves must already have ordered, corresponding samples. This function
    neither discovers attachment curves nor guarantees absence of intersections.
    """
    bottom,top,left,right=[np.asarray(x,float) for x in (bottom,top,left,right)]
    if any(x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or not np.isfinite(x).all() for x in (bottom,top,left,right)):
        raise ValueError('Finite 3D boundary curves with at least two samples required')
    if bottom.shape!=top.shape or left.shape!=right.shape:raise ValueError('Opposite boundaries need matching sample counts')
    if not all(np.allclose(a,b,rtol=0,atol=1e-10) for a,b in
        ((bottom[0],left[0]),(bottom[-1],right[0]),(top[0],left[-1]),(top[-1],right[-1]))):
        raise ValueError('Boundary corners do not agree')
    nu,nv=len(bottom),len(left)
    u=np.linspace(0,1,nu)[None,:,None];v=np.linspace(0,1,nv)[:,None,None]
    bilinear=(1-u)*(1-v)*bottom[0]+u*(1-v)*bottom[-1]+(1-u)*v*top[0]+u*v*top[-1]
    points=(1-v)*bottom[None]+v*top[None]+(1-u)*left[:,None]+u*right[:,None]-bilinear
    if interior_offset is not None:
        offset=np.asarray(interior_offset,float)
        if offset.shape!=points.shape or not np.isfinite(offset).all():raise ValueError('Invalid interior offset')
        if not all(np.allclose(edge,0,rtol=0,atol=1e-12) for edge in (offset[0],offset[-1],offset[:,0],offset[:,-1])):
            raise ValueError('Interior fold offsets may not move attachment boundaries')
        points+=offset
    faces=[]
    for row in range(nv-1):
        for col in range(nu-1):
            a=row*nu+col;faces.extend(((a,a+1,a+nu+1),(a,a+nu+1,a+nu)))
    result=trimesh.Trimesh(points.reshape(-1,3),np.asarray(faces),process=False)
    if np.any(result.area_faces<1e-14):raise ValueError('Degenerate surface triangles; review boundary correspondence')
    return result
