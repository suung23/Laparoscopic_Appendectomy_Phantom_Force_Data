"""Device-resident batched implicit membrane reference, without contact or RL."""
import numpy as np
import torch
from membrane_dynamics import MembraneDynamics

class BatchedMembraneDynamics:
    def __init__(self,model,rest,density,fixed,batch_size):
        if batch_size<1:raise ValueError('Positive batch size required')
        setup=MembraneDynamics(model,rest,density,fixed)
        self.model=model;device=model.device
        self.points=torch.tensor(rest,dtype=torch.float64,device=device).repeat(batch_size,1,1)
        self.velocity=torch.zeros_like(self.points)
        self.mass=torch.tensor(setup.mass,dtype=torch.float64,device=device)
        self.stiffness=torch.tensor(setup.stiffness_diagonal,dtype=torch.float64,device=device)
        self.free=torch.tensor(setup.free,device=device);self.fixed=torch.tensor(setup.fixed,device=device)

    def step(self,dt,forces,maxiter=1000,tolerance=1e-7,defect_tolerance=1e-6,history_size=10,independent=False):
        if not np.isfinite([dt,tolerance,defect_tolerance]).all() or min(dt,tolerance,defect_tolerance)<=0:raise ValueError('Positive finite step controls required')
        if forces.shape!=self.points.shape or forces.device!=self.points.device or forces.dtype!=self.points.dtype or not torch.isfinite(forces).all():raise ValueError('Invalid batched forces')
        old=self.points;prediction=old+dt*self.velocity;free=self.free;energy_scale=1e-6
        scale=torch.sqrt(energy_scale/(self.mass[free]/dt**2+self.stiffness[free]))[:,None]
        initial=(dt*self.velocity[:,free]+dt**2*forces[:,free]/self.mass[free,None])/scale
        z=initial.detach().clone().requires_grad_(True)
        force_limit=torch.minimum(torch.full_like(self.mass[free],tolerance),defect_tolerance*self.mass[free]/dt**2)
        if maxiter<1 or history_size<1:raise ValueError('Positive optimizer limits required')
        optimizer=torch.optim.LBFGS([z],max_iter=maxiter,max_eval=5*maxiter,tolerance_grad=float((force_limit[:,None]*scale).min()/energy_scale),tolerance_change=0.,line_search_fn='strong_wolfe',history_size=history_size)
        calls=0
        def evaluate(candidate=None):
            candidate=z if candidate is None else candidate
            points=old.clone();points[:,free]=old[:,free]+candidate.detach()*scale
            energy,internal=self.model.batch_energy_forces(points)
            delta=points-prediction
            value=energy+.5*(self.mass[None,:,None]*delta**2).sum(dim=(1,2))/dt**2-(forces*(points-old)).sum(dim=(1,2))
            residual=self.mass[None,:,None]*delta/dt**2-internal-forces
            return points,value,residual[:,free]
        def closure():
            nonlocal calls
            calls+=1;_,value,residual=evaluate()
            z.grad=(residual*scale/energy_scale).detach()
            return value.sum()/energy_scale
        independent_report=None
        if independent:
            from independent_batch_lbfgs import minimize
            def objective(candidate):
                _,value,residual=evaluate(candidate)
                return value/energy_scale,residual*scale/energy_scale
            solution,independent_report=minimize(objective,z,float((force_limit[:,None]*scale).min()/energy_scale),maxiter,history_size)
            with torch.no_grad():z.copy_(solution)
            calls=independent_report['objective_calls']
        else:optimizer.step(closure)
        updated,_,residual=evaluate()
        maximum=residual.abs().amax(dim=(1,2));defect=(residual.abs()*dt**2/self.mass[free,None]).amax(dim=(1,2))
        report={'force_residual_N':maximum.cpu().tolist(),'inertial_defect_m':defect.cpu().tolist(),'objective_calls':calls,'batch_size':len(old),
                'optimizer_iterations':int(optimizer.state[z].get('n_iter',0)),'max_iterations':maxiter,'max_evaluations':5*maxiter,'history_size':history_size}
        self.last_report=report
        report['independent_optimizer']=independent_report
        if not torch.isfinite(residual).all() or torch.any(maximum>tolerance) or torch.any(defect>defect_tolerance):
            raise RuntimeError('Batched convergence failed; state unchanged: '+str(report))
        self.velocity=(updated-old)/dt;self.points=updated
        return report
