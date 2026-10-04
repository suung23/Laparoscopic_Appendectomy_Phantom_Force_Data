"""Objective tetrahedral deformation measures, not material calibration."""
import numpy as np


class DeformationMeasure:
    def __init__(self, reference, tetrahedra):
        self.reference=np.asarray(reference,float).copy()
        self.tets=np.asarray(tetrahedra,int).reshape(-1,4)
        if self.reference.ndim!=2 or self.reference.shape[1]!=3 or not np.isfinite(self.reference).all():
            raise ValueError('Finite reference nodes required')
        if self.tets.min()<0 or self.tets.max()>=len(self.reference):raise ValueError('Invalid tetrahedron index')
        D=self.edges(self.reference)
        if np.any(np.abs(np.linalg.det(D))<6e-15):raise ValueError('Degenerate reference tetrahedron')
        self.inverse=np.linalg.inv(D)

    def edges(self,p):
        q=p[self.tets]
        return np.stack([q[:,i]-q[:,0] for i in (1,2,3)],axis=-1)

    def measure(self,current):
        q=np.asarray(current,float)
        if q.shape!=self.reference.shape or not np.isfinite(q).all():raise ValueError('Invalid tissue state')
        F=self.edges(q)@self.inverse
        stretch=np.linalg.svd(F,compute_uv=False)
        jacobian=np.linalg.det(F)
        # Best-fit rigid transform removes translation and rotation. A moved
        # rigid mesh therefore cannot pass a non-rigid-deformation gate.
        a=self.reference-self.reference.mean(0);b=q-q.mean(0)
        u,_,vt=np.linalg.svd(a.T@b)
        if np.linalg.det(u@vt)<0:u[:,-1]*=-1
        residual=np.linalg.norm(a@u@vt-b,axis=1)
        displacement=np.linalg.norm(q-self.reference,axis=1)
        return dict(nonrigid_rms_m=float(np.sqrt(np.mean(residual**2))),
            nonrigid_max_m=float(residual.max()),max_displacement_m=float(displacement.max()),
            stretch_min=float(stretch.min()),stretch_max=float(stretch.max()),
            p99_abs_principal_strain=float(np.percentile(np.abs(stretch-1),99)),
            minimum_jacobian=float(jacobian.min()),inverted_tetrahedra=int(np.sum(jacobian<=0)))
