"""Conforming planar tetrahedral split for an isolated topology benchmark.

Produces duplicated interface vertices and conserves solid volume. This is not
a finite scissor blade, a damage law, a hollow-organ model or an Isaac live cut.
"""
import numpy as np


def signed_volumes(points,tetrahedra):
    q=np.asarray(points)[np.asarray(tetrahedra)]
    return np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6


def cut_plane(points,tetrahedra,normal,offset,velocities=None):
    points=np.asarray(points,float)
    tetrahedra=np.asarray(tetrahedra,int).reshape(-1,4)
    normal=np.asarray(normal,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite Nx3 points required')
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-12 or not np.isfinite(offset):raise ValueError('Invalid plane')
    if tetrahedra.min()<0 or tetrahedra.max()>=len(points):raise ValueError('Invalid tetrahedron indices')
    normal_length=np.linalg.norm(normal)
    normal=normal/normal_length
    offset=float(offset)/normal_length
    velocity=np.zeros_like(points) if velocities is None else np.asarray(velocities,float)
    if velocity.shape!=points.shape or not np.isfinite(velocity).all():raise ValueError('Invalid velocities')
    distance=points@normal-offset
    # Exact node/plane coincidences require a separately designed degeneracy rule.
    if np.any(np.abs(distance)<1e-10):raise ValueError('Plane crosses a node; perturb the benchmark plane explicitly')
    if np.any(np.abs(signed_volumes(points,tetrahedra))<1e-15):raise ValueError('Degenerate input tetrahedra')
    vertices,velocities_out,cells,sides=[],[],[],[]
    lookup={}
    def vertex(side,index):
        key=(side,'v',int(index))
        if key not in lookup:
            lookup[key]=len(vertices);vertices.append(points[index]);velocities_out.append(velocity[index])
        return lookup[key]
    def intersection(side,i,j):
        i,j=sorted((int(i),int(j)))
        key=(side,'e',i,j)
        if key not in lookup:
            t=distance[i]/(distance[i]-distance[j])
            lookup[key]=len(vertices)
            vertices.append(points[i]+t*(points[j]-points[i]))
            velocities_out.append(velocity[i]+t*(velocity[j]-velocity[i]))
        return lookup[key]
    faces=((0,2,1),(0,1,3),(0,3,2),(1,2,3))
    helper=np.eye(3)[np.argmin(np.abs(normal))]
    u=np.cross(normal,helper);u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    for tetra in tetrahedra:
        if np.all(distance[tetra]>0) or np.all(distance[tetra]<0):
            side=1 if distance[tetra[0]]>0 else -1
            cells.append([vertex(side,i) for i in tetra]);sides.append(side)
            continue
        for side in (-1,1):
            polygons=[]
            cut_vertices=set()
            for face in faces:
                triangle=tetra[list(face)]
                polygon=[]
                for i,j in zip(triangle,np.roll(triangle,-1)):
                    inside_i=distance[i]*side>0
                    inside_j=distance[j]*side>0
                    if inside_i:polygon.append(vertex(side,i))
                    if inside_i!=inside_j:
                        identity=intersection(side,i,j)
                        polygon.append(identity);cut_vertices.add(identity)
                if len(polygon)>=3:polygons.append(polygon)
            ids=sorted(cut_vertices)
            coords=np.asarray([vertices[i] for i in ids])
            centered=coords-coords.mean(axis=0)
            order=np.argsort(np.arctan2(centered@v,centered@u))
            polygons.append([ids[i] for i in order])
            all_ids=sorted(set(i for face in polygons for i in face))
            centroid=len(vertices)
            vertices.append(np.mean([vertices[i] for i in all_ids],axis=0))
            velocities_out.append(np.mean([velocities_out[i] for i in all_ids],axis=0))
            for polygon in polygons:
                # Canonical diagonal yields matching triangulation at shared faces.
                start=polygon.index(min(polygon))
                polygon=polygon[start:]+polygon[:start]
                for k in range(1,len(polygon)-1):
                    cells.append([centroid,polygon[0],polygon[k],polygon[k+1]])
                    sides.append(side)
    vertices=np.asarray(vertices)
    cells=np.asarray(cells,dtype=np.int32)
    volume=signed_volumes(vertices,cells)
    if np.any(np.abs(volume)<1e-15):raise ValueError('Cut creates near-degenerate tetrahedra; refine or reposition plane')
    inverted=volume<0
    temporary=cells[inverted,0].copy();cells[inverted,0]=cells[inverted,1];cells[inverted,1]=temporary
    return {'points':vertices,'tetrahedra':cells,'velocities':np.asarray(velocities_out),'side':np.asarray(sides)}


def boundary_faces(points,tetrahedra):
    tetrahedra=np.asarray(tetrahedra)
    if np.any(signed_volumes(points,tetrahedra)<=0):raise ValueError('Positive tetrahedra required')
    faces=np.concatenate([tetrahedra[:,face] for face in ((0,2,1),(0,1,3),(0,3,2),(1,2,3))])
    _,inverse,count=np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError('Non-manifold tetrahedral mesh')
    return faces[count[inverse]==1]
