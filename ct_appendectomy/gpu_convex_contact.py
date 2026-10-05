"""GPU nodal penalty against convex CAD planes; no CCD, friction or tissue calibration."""
import numpy as np
import torch
from scipy.spatial import ConvexHull

class ConvexContact:
    def __init__(self,mesh,device='cuda'):
        points=np.asarray(mesh.vertices,float);origin=points.mean(axis=0);span=np.ptp(points,axis=0).max()
        if not np.isfinite(points).all() or span<=0:raise ValueError('Invalid convex points')
        equations=ConvexHull((points-origin)/span).equations
        normals=equations[:,:3];offset=equations[:,3]*span-normals@origin
        self.normals=torch.tensor(normals,dtype=torch.float64,device=device)
        self.offset=torch.tensor(offset,dtype=torch.float64,device=device)

    def penalty(self,points,poses,area,stiffness=1e5,chunk_size=4096):
        if points.ndim!=3 or points.shape[-1]!=3 or poses.shape!=(len(points),4,4):raise ValueError('Invalid batch shapes')
        if points.dtype!=torch.float64 or poses.dtype!=torch.float64 or area.dtype!=torch.float64:raise ValueError('Float64 reference required')
        if points.device!=self.normals.device or poses.device!=points.device or area.device!=points.device:raise ValueError('Device mismatch')
        if area.shape!=(points.shape[1],) or not torch.isfinite(area).all() or torch.any(area<0) or not np.isfinite(stiffness) or stiffness<=0 or chunk_size<1:raise ValueError('Invalid penalty parameters')
        if not torch.isfinite(points).all() or not torch.isfinite(poses).all():raise ValueError('Nonfinite coordinates')
        rotation=poses[:,:3,:3]
        if not torch.allclose(rotation.transpose(1,2)@rotation,torch.eye(3,device=points.device,dtype=points.dtype),atol=1e-9,rtol=0):raise ValueError('Rigid rotations required')
        force=torch.zeros_like(points);depth=torch.zeros(points.shape[:2],device=points.device,dtype=points.dtype)
        for start in range(0,points.shape[1],chunk_size):
            end=min(start+chunk_size,points.shape[1])
            local=(points[:,start:end]-poses[:,None,:3,3])@rotation
            signed=local@self.normals.T+self.offset
            closest,index=signed.max(dim=-1)
            penetration=(-closest).clamp_min(0)
            normal=self.normals[index]@rotation.transpose(1,2)
            force[:,start:end]=stiffness*area[None,start:end,None]*penetration[:,:,None]*normal
            depth[:,start:end]=penetration
        return force,{'depth_m':depth,'energy_J':.5*stiffness*(area*depth**2).sum(1),'reaction_N':-force.sum(1)}
