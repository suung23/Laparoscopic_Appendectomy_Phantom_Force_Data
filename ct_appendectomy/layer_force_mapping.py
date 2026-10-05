"""Work-conjugate pullback for the normal-offset visual-layer map.

This transfers supplied nodal loads. It is not a contact detector, calibrated
fat material, or integration with an Isaac solver. CPU float64 reference.
"""
import numpy as np
from boundary_surface import SurfaceBinding

def pullback_forces(parent,nodes,weights,offset,forces):
    import torch
    parent=np.asarray(parent,float);offset=np.asarray(offset,float);forces=np.asarray(forces,float)
    binding=SurfaceBinding(nodes,weights,len(parent));binding.positions(parent)
    if offset.shape!=(len(binding.nodes),3) or forces.shape!=offset.shape or not np.isfinite(offset).all() or not np.isfinite(forces).all():
        raise ValueError('Finite per-layer-point offsets and forces required')
    p=torch.tensor(parent,dtype=torch.float64,requires_grad=True)
    ids=torch.tensor(binding.nodes,dtype=torch.int64);w=torch.tensor(binding.weights,dtype=torch.float64)
    delta=torch.tensor(offset,dtype=torch.float64);load=torch.tensor(forces,dtype=torch.float64)
    triangles=p[ids];edge=triangles[:,1]-triangles[:,0];length=torch.linalg.vector_norm(edge,dim=1)
    if torch.any(length<1e-12):raise ValueError('Degenerate attachment edge')
    tangent=edge/length[:,None]
    normal=torch.linalg.cross(tangent,triangles[:,2]-triangles[:,0]);norm=torch.linalg.vector_norm(normal,dim=1)
    if torch.any(norm<1e-12):raise ValueError('Degenerate attachment face')
    normal=normal/norm[:,None];bitangent=torch.linalg.cross(normal,tangent)
    position=(w[:,:,None]*triangles).sum(dim=1)+delta[:,0,None]*tangent+delta[:,1,None]*bitangent+delta[:,2,None]*normal
    # J^T f includes rotation of the local frame. Barycentric weights alone
    # omit the torque contributed by nonzero normal offsets.
    result=torch.autograd.grad((position*load).sum(),p)[0]
    return result.detach().numpy()
