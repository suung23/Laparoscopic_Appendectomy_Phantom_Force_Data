"""Implicit-Euler CPU membrane reference; no contact or Isaac integration."""
import numpy as np
from scipy.optimize import minimize

class MembraneDynamics:
    def __init__(self,model,rest,density,fixed):
        if not np.isfinite(density) or density<=0:raise ValueError('Positive density required')
        self.model=model;self.points=np.array(rest,float,copy=True);self.velocity=np.zeros_like(self.points)
        self.fixed=np.asarray(fixed,dtype=int);self.free=np.setdiff1d(np.arange(len(rest)),self.fixed)
        self.mass=np.zeros(len(rest));tri_mass=model.area.cpu().numpy()*model.thickness*density
        np.add.at(self.mass,model.faces.cpu().numpy().ravel(),np.repeat(tri_mass/3,3))
        if (self.mass<=0).any():raise ValueError('Every vertex needs positive mass')
        inverse=model.inverse.cpu().numpy()
        gradients=np.stack([-inverse.sum(axis=1),inverse[:,0],inverse[:,1]],axis=1)
        local=(2*model.mu+model.lam)*model.thickness*model.area.cpu().numpy()[:,None]*np.sum(gradients**2,axis=2)
        self.stiffness_diagonal=np.zeros(len(rest))
        np.add.at(self.stiffness_diagonal,model.faces.cpu().numpy().ravel(),local.ravel())
    def step(self,dt,forces,maxiter=1000,tolerance=1e-6,defect_tolerance=None):
        forces=np.asarray(forces,float)
        if not np.isfinite(dt) or dt<=0 or forces.shape!=self.points.shape or not np.isfinite(forces).all():raise ValueError('Invalid step inputs')
        old=self.points.copy();prediction=old+dt*self.velocity;free=self.free
        if defect_tolerance is not None and (not np.isfinite(defect_tolerance) or defect_tolerance<=0):raise ValueError('Positive inertial defect tolerance required')
        energy_scale=1e-6
        scale=np.sqrt(energy_scale/(self.mass[free]/dt**2+self.stiffness_diagonal[free]))[:,None]
        def objective(z):
            points=old.copy();points[free]+=z.reshape(-1,3)*scale
            energy,internal=self.model.energy_forces(points)
            delta=points-prediction
            value=energy+.5*np.sum(self.mass[:,None]*delta**2)/dt**2-np.sum(forces*(points-old))
            gradient=self.mass[:,None]*delta/dt**2-internal-forces
            return value/energy_scale,(gradient[free]*scale/energy_scale).ravel()
        initial=(dt*self.velocity[free]+dt**2*forces[free]/self.mass[free,None])/scale
        force_limit=np.full((len(free),1),tolerance)
        if defect_tolerance is not None:force_limit=np.minimum(force_limit,defect_tolerance*self.mass[free,None]/dt**2)
        result=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'gtol':float(np.min(force_limit*scale))/energy_scale,'ftol':0. if defect_tolerance is not None else 1e-15,'maxls':30})
        _,gradient=objective(result.x);physical_residual=gradient.reshape(-1,3)*energy_scale/scale
        residual=float(np.max(np.abs(physical_residual)))
        defect=float(np.max(np.abs(physical_residual)*dt**2/self.mass[free,None]))
        report={'optimizer_success':bool(result.success),'iterations':int(result.nit),'residual_N':residual,'tolerance_N':tolerance,'message':str(result.message)}
        report.update(inertial_residual_defect_m=defect,defect_tolerance_m=defect_tolerance,defect_is_position_error_bound=False)
        if not np.isfinite(residual) or residual>tolerance or (defect_tolerance is not None and defect>defect_tolerance):
            raise RuntimeError('Unconverged membrane step: '+str(report))
        updated=old.copy();updated[free]+=result.x.reshape(-1,3)*scale
        self.velocity=(updated-old)/dt;self.points=updated
        report['status']='converged';return report
