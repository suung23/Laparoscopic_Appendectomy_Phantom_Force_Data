"""Rest-angle hinge bending reference inspired by Discrete Shells (2003).
https://multires.caltech.edu/pubs/ds.pdf
Stiffness is an explicit uncalibrated N.m parameter, not fitted tissue data.
"""
import numpy as np
import torch
import trimesh

class HingeBending:
    def __init__(self,points,faces,stiffness):
        if not np.isfinite(stiffness) or stiffness<0:raise ValueError('Nonnegative finite bending stiffness required')
        mesh=trimesh.Trimesh(points,faces,process=False)
        if not mesh.is_winding_consistent:raise ValueError('Consistent winding required')
        self.faces=torch.tensor(np.asarray(faces),dtype=torch.int64)
        self.pairs=torch.tensor(mesh.face_adjacency,dtype=torch.int64)
        self.edges=torch.tensor(mesh.face_adjacency_edges,dtype=torch.int64)
        lengths=np.linalg.norm(mesh.vertices[mesh.face_adjacency_edges[:,1]]-mesh.vertices[mesh.face_adjacency_edges[:,0]],axis=1)
        areas=mesh.area_faces[mesh.face_adjacency].sum(axis=1)
        if (areas<=1e-14).any():raise ValueError('Degenerate rest hinge')
        self.weight=torch.tensor(3*lengths**2/areas,dtype=torch.float64)
        self.stiffness=stiffness
        self.rest_cos,self.rest_sin=self.angles(torch.tensor(points,dtype=torch.float64))
    def angles(self,p):
        t=p[self.faces];cross=torch.linalg.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=torch.linalg.vector_norm(cross,dim=1)
        if torch.any(length<=1e-14):raise ValueError('Degenerate current face')
        normal=cross/length[:,None];a=normal[self.pairs[:,0]];b=normal[self.pairs[:,1]]
        edge=p[self.edges[:,1]]-p[self.edges[:,0]];edge=edge/torch.linalg.vector_norm(edge,dim=1)[:,None]
        return (a*b).sum(dim=1),(torch.linalg.cross(a,b)*edge).sum(dim=1)
    def energy(self,p):
        c,s=self.angles(p)
        delta=torch.atan2(s*self.rest_cos-c*self.rest_sin,c*self.rest_cos+s*self.rest_sin)
        return .5*self.stiffness*(self.weight*delta**2).sum()
