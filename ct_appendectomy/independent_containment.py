"""Independent sampled containment diagnostic using NumPy and SciPy, no Rtree.

Closed surfaces only; boundary and tangency cases require distance checks.
Not triangle/triangle intersection detection or a runtime contact solver.
"""
import numpy as np
from scipy.spatial import cKDTree

def contains(mesh,points):
    if not mesh.is_watertight:raise ValueError('Closed mesh required')
    points=np.asarray(points,float)
    origin=mesh.bounds.mean(axis=0);span=np.ptp(mesh.vertices,axis=0).max()
    if span<=0:raise ValueError('Nonzero mesh extent required')
    p=(points-origin)/span;v=(mesh.vertices-origin)/span
    active=np.flatnonzero(((p>=v.min(axis=0))&(p<=v.max(axis=0))).all(axis=1))
    result=np.zeros(len(p),bool)
    if not len(active):return result
    direction=np.array([1.,.371,.529]);direction/=np.linalg.norm(direction)
    b=np.cross(direction,[0.,0.,1.]);b/=np.linalg.norm(b);c=np.cross(direction,b)
    rotation=np.stack([direction,b,c],axis=1)
    q=p[active]@rotation;t=(v@rotation)[mesh.faces]
    tree=cKDTree(q[:,1:]);hit_ids=[];hit_t=[]
    for tri in t:
        uv=tri[:,1:];center=uv.mean(axis=0);radius=np.linalg.norm(uv-center,axis=1).max()+1e-12
        ids=np.asarray(tree.query_ball_point(center,radius),dtype=int)
        if not len(ids):continue
        e=uv[1]-uv[0];f=uv[2]-uv[0];det=e[0]*f[1]-e[1]*f[0]
        if abs(det)<1e-14:continue
        w=q[ids,1:]-uv[0]
        u=(w[:,0]*f[1]-w[:,1]*f[0])/det;z=(e[0]*w[:,1]-e[1]*w[:,0])/det
        mask=(u>=-1e-12)&(z>=-1e-12)&(u+z<=1+1e-12)
        ray_t=tri[0,0]+u*(tri[1,0]-tri[0,0])+z*(tri[2,0]-tri[0,0])-q[ids,0]
        mask &= ray_t>1e-12
        hit_ids.extend(ids[mask]);hit_t.extend(ray_t[mask])
    if not hit_ids:return result
    ids=np.asarray(hit_ids);dist=np.asarray(hit_t);order=np.lexsort((dist,ids));ids=ids[order];dist=dist[order]
    unique=np.r_[True,(np.diff(ids)!=0)|(np.abs(np.diff(dist))>1e-10)]
    count=np.bincount(ids[unique],minlength=len(active));result[active]=count%2==1
    return result
