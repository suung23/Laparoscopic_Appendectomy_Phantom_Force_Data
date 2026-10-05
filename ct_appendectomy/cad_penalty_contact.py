"""Area-weighted frictionless CAD penetration penalty reference.
No calibrated stiffness, continuous collision detection, friction, or tool control.
"""
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from independent_containment import contains

def part_pose(pose,name,half_angle):
    """Rosota rest-CAD hinge convention: left +Y rotation, right -Y."""
    if not np.isfinite(half_angle) or not 0<=half_angle<=np.deg2rad(37):raise ValueError('Jaw half-angle outside 0-37 degrees')
    result=np.array(pose,float,copy=True)
    if result.shape!=(4,4) or not np.isfinite(result).all():raise ValueError('Invalid tool pose')
    sign={'gripper_tip_left':1,'gripper_tip_right':-1}.get(name,0)
    c,s=np.cos(sign*half_angle),np.sin(sign*half_angle)
    hinge=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    result[:3,:3]=result[:3,:3]@hinge
    return result

def load_jaw(path,part='gripper_tip_left'):
    stage=Usd.Stage.Open(str(path.resolve()));mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Tool/'+part+'/mesh'))
    if not mesh:raise ValueError('CAD jaw mesh missing')
    points=np.asarray(mesh.GetPointsAttr().Get(),float);matrix=np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0))
    points=(np.column_stack([points,np.ones(len(points))])@matrix)[:,:3]*UsdGeom.GetStageMetersPerUnit(stage)
    # Explicit convex collision approximation, matching the earlier Isaac path.
    hull=trimesh.Trimesh(points,np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False).convex_hull
    if not hull.is_watertight:
        # Long, thin metric CAD can lose hull facets to absolute tolerances.
        # Recompute from original points in normalized coordinates, then restore metres.
        center=(points.min(axis=0)+points.max(axis=0))/2
        span=np.ptp(points,axis=0).max()
        hull=trimesh.convex.convex_hull((points-center)/span)
        hull.vertices=hull.vertices*span+center
    if not hull.is_watertight:raise ValueError('CAD convex hull is not closed: '+part)
    return hull

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

def compound_penalty(parts,points,area,stiffness):
    """Add per-part potentials; overlapping hulls can over-stiffen contact."""
    total=np.zeros_like(points,dtype=float);records={};active=np.zeros(len(points),bool)
    for name,mesh in parts.items():
        force,record=penalty(mesh,points,area,stiffness)
        total+=force;records[name]=record;active|=np.linalg.norm(force,axis=1)>0
    return total,{'active_nodes':int(active.sum()),'max_penetration_m':max((r['max_penetration_m'] for r in records.values()),default=0.),
                  'energy_J':sum(r['energy_J'] for r in records.values()),'reaction_N':(-total.sum(axis=0)).tolist(),'parts':records}
