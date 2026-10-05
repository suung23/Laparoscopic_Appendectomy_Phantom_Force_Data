"""CPU float64 triangular Neo-Hookean membrane reference.

Plane-stress linearized Lame constants, finite-strain surface metric energy.
No bending, contact, cutting, material calibration or Isaac integration.
"""
import numpy as np
import torch

class MembraneReference:
    def __init__(self,points,faces,young,poisson,thickness):
        if not (np.isfinite([young,poisson,thickness]).all() and young>0 and thickness>0 and -1<poisson<.5):raise ValueError('Invalid material parameters')
        self.faces=torch.tensor(np.asarray(faces),dtype=torch.int64)
        rest=torch.tensor(np.asarray(points),dtype=torch.float64)
        tri=rest[self.faces];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0]
        length=torch.linalg.vector_norm(a,dim=1);cross=torch.linalg.vector_norm(torch.linalg.cross(a,b),dim=1)
        if torch.any(length<=1e-12) or torch.any(cross<=1e-14):raise ValueError('Degenerate rest triangles')
        tangent=a/length[:,None];x=(b*tangent).sum(dim=1);y=cross/length
        dm=torch.zeros((len(tri),2,2),dtype=torch.float64);dm[:,0,0]=length;dm[:,0,1]=x;dm[:,1,1]=y
        self.inverse=torch.linalg.inv(dm);self.area=cross/2;self.thickness=thickness
        self.mu=young/(2*(1+poisson));self.lam=young*poisson/(1-poisson**2)
        self.shape=rest.shape
    def energy(self,points):
        tri=points[self.faces];ds=torch.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],dim=2)
        f=ds@self.inverse;c=f.transpose(1,2)@f
        det=torch.linalg.det(c)
        if torch.any(det<=1e-16):raise ValueError('Collapsed current membrane triangle')
        logj=.5*torch.log(det);trace=c[:,0,0]+c[:,1,1]
        density=.5*self.mu*(trace-2-2*logj)+.5*self.lam*logj**2
        return (density*self.area*self.thickness).sum()
    def energy_forces(self,points):
        p=torch.tensor(np.asarray(points),dtype=torch.float64,requires_grad=True)
        if p.shape!=self.shape or not torch.isfinite(p).all():raise ValueError('Invalid current points')
        energy=self.energy(p);force=-torch.autograd.grad(energy,p)[0]
        return float(energy.detach()),force.detach().numpy()
