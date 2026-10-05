"""Implicit Euler reference dynamics; not realtime or calibrated tissue.

Solves elastic + contact + inertia - external work. Failed solves never advance
the caller's state. Backward Euler has numerical damping, not physical viscosity.
"""
import numpy as np
from scipy.sparse import diags


class ImplicitConvergenceError(RuntimeError):
    """Carries an unaccepted iterate for diagnosis, never an advanced state."""
    def __init__(self, result):
        super().__init__('Implicit step failed: '+result['status'])
        self.result = result


class InertialPotential:
    def __init__(self, prediction, mass, dt, contact=None):
        self.prediction = np.asarray(prediction, float)
        mass = np.asarray(mass, float)
        if (self.prediction.ndim != 2 or self.prediction.shape[1] != 3
            or mass.shape != (len(self.prediction),) or not np.isfinite(mass).all()
            or np.any(mass <= 0) or not np.isfinite(self.prediction).all()
            or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid inertial inputs')
        self.diagonal = mass/dt**2
        self.contact = contact

    def energy_gradient(self, x):
        delta = x-self.prediction
        gradient = self.diagonal[:,None]*delta
        energy = .5*np.sum(delta*gradient)
        if self.contact is not None:
            e, g = self.contact.energy_gradient(x)
            energy += e; gradient += g
        return float(energy), gradient

    def hessian(self, x):
        matrix = diags(np.repeat(self.diagonal,3)).tocsc()
        if self.contact is not None:
            matrix += self.contact.hessian(x)
        return matrix

    def feasible_step(self, x, direction):
        return self.contact.feasible_step(x,direction) if self.contact is not None else 1.


def implicit_step(elastic, positions, velocity, mass, loads, dt, contact=None,
                  max_iterations=120, tolerance=1e-11):
    positions, velocity = np.asarray(positions,float), np.asarray(velocity,float)
    if positions.shape != elastic.rest.shape or velocity.shape != positions.shape:
        raise ValueError('State shape mismatch')
    potential = InertialPotential(positions+dt*velocity,mass,dt,contact)
    # Start from previous feasible position, not a predictor inside obstacles.
    result = elastic.solve(positions, loads, np.zeros_like(positions,bool),
        method='lbfgs_rotated', contact=potential,
        max_iterations=max_iterations, tolerance=tolerance)
    if result['status'] != 'converged':
        # Local refinement only; the 1e-8 eligibility threshold is NOT an
        # acceptance tolerance. Final force tolerance remains caller-specified.
        if result['residual_n'] <= 1e-8:
            from .newton_direction import newton_direction
            from .determinant_step import positive_volume_step
            refinements = []
            for _ in range(3):
                x = result['positions']
                e,g = elastic.energy_gradient(x,loads)
                pe,pg = potential.energy_gradient(x); energy,gradient = e+pe,g+pg
                try:
                    direction,diagnostic = newton_direction(elastic,x,gradient,potential)
                except RuntimeError:
                    break
                alpha = min(positive_volume_step(elastic.rest,x,direction,elastic.tets)['alpha'],
                            potential.feasible_step(x,direction))
                candidate = x+alpha*direction
                te,tg = elastic.energy_gradient(candidate,loads)
                tpe,tpg = potential.energy_gradient(candidate)
                residual = float(np.max(np.abs(tg+tpg)))
                # Explicit roundoff allowance, with >=10x force contraction.
                allowance = 64*np.finfo(float).eps*max(abs(e)+abs(pe),abs(te)+abs(tpe),1e-30)
                if residual >= .1*result['residual_n'] or te+tpe > energy+allowance:
                    break
                refinements.append(dict(**diagnostic,alpha=alpha,residual_n=residual,
                    energy_change_j=te+tpe-energy,energy_roundoff_allowance_j=allowance))
                result.update(positions=candidate,residual_n=residual)
                result['history'].append(dict(energy_j=te+tpe,residual_n=residual))
                if residual <= tolerance:
                    result['status']='converged'
                    break
            result['newton_refinements']=refinements
        if result['status'] != 'converged':
            raise ImplicitConvergenceError(result)
    result['velocity'] = (result['positions']-positions)/dt
    return result
