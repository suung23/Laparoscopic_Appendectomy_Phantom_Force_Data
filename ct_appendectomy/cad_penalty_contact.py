"""Area-weighted frictionless CAD penetration penalty reference.
No calibrated stiffness, continuous collision detection, friction, or tool control.
"""
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from independent_containment import contains

def load_jaw(path,part='gripper_tip_left'):
    stage=Usd.Stage.Open(str(path.resolve()));mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Tool/'+part+'/mesh'))
    if not mesh:raise ValueError('CAD jaw mesh missing')
    points=np.asarray(mesh.GetPointsAttr().Get(),float);matrix=np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0))
    points=(np.column_stack([points,np.ones(len(points))])@matrix)[:,:3]*UsdGeom.GetStageMetersPerUnit(stage)
    # Explicit convex collision approximation, matching the earlier Isaac path.
    return trimesh.Trimesh(points,np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False).convex_hull

def penalty(mesh,points,area,stiffness):
    points=np.asarray(points,float);area=np.asarray(area,float)
    if area.shape!=(len(points),) or (area<0).any() or not np.isfinite(area).all() or not np.isfinite(stiffness) or stiffness<=0:raise ValueError('Invalid area or stiffness')
    origin=mesh.bounds.mean(axis=0);scale=1/np.ptp(mesh.vertices,axis=0).max();query=trimesh.Trimesh((mesh.vertices-origin)*scale,mesh.faces,process=False)
    local=(points-origin)*scale;inside=contains(query,local);force=np.zeros_like(points);depth=np.zeros(len(points))
    ids=np.flatnonzero(inside)
    for chunk in np.array_split(ids,max(1,len(ids)//32+1)):
        if not len(chunk):continue
        closest,distance,_=trimesh.proximity.closest_point_naive(query,local[chunk]);delta=(closest-local[chunk])/scale
        force[chunk]=stiffness*area[chunk,None]*delta;depth[chunk]=distance/scale
    return force,{'active_nodes':int((depth>1e-10).sum()),'max_penetration_m':float(depth.max(initial=0)),'energy_J':float(.5*stiffness*np.sum(area*depth**2)),'reaction_N':(-force.sum(axis=0)).tolist()}
