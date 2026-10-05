"""Reference quasistatic compressible Neo-Hookean solve in SI units.

CPU correctness prototype, not calibrated tissue or a contact/dynamics solver.
Uses a log(J) constitutive domain and feasible Armijo steps, never repairs a
completed trajectory. Prescribed nodes must initially satisfy their constraints.
"""
import numpy as np
from .determinant_step import positive_volume_step


class ElasticReference:
    def __init__(self, rest, tets, young, poisson):
        self.rest = np.asarray(rest, dtype=float)
        self.tets = np.asarray(tets)
        if not np.isfinite(young) or young <= 0 or not 0 <= poisson < .5:
            raise ValueError('Invalid elastic parameters')
        positive_volume_step(self.rest, self.rest, np.zeros_like(self.rest), self.tets)
        q = self.rest[self.tets]
        dm = np.stack([q[:, i] - q[:, 0] for i in (1, 2, 3)], axis=2)
        self.inverse = np.linalg.inv(dm)
        self.volume = np.linalg.det(dm) / 6
        self.mu = young / (2 * (1 + poisson))
        self.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

    def energy_gradient(self, x, loads):
        x, loads = np.asarray(x, float), np.asarray(loads, float)
        if x.shape != self.rest.shape or loads.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(loads).all():
            raise ValueError('Invalid positions or nodal loads')
        q = x[self.tets]
        ds = np.stack([q[:, i] - q[:, 0] for i in (1, 2, 3)], axis=2)
        f = ds @ self.inverse
        j = np.linalg.det(f)
        if np.any(j <= 0):
            raise ValueError('Non-positive determinant')
        logj = np.log(j)
        # Equivalent spectral energy avoids subtracting O(1) quantities when
        # strain energy is O(strain**2), notably near rigid motion.
        stretch = np.linalg.svd(f, compute_uv=False)
        t = stretch-1
        remainder = t-np.log1p(t)
        small = np.abs(t)<1e-3
        ts = t[small]
        remainder[small] = sum(((-1.)**n)*ts**n/n for n in range(2,9))
        spectral_logj = np.log1p(t).sum(axis=1)
        density = .5*self.mu*np.sum(t*t+2*remainder,axis=1)+.5*self.lam*spectral_logj**2
        p = self.mu*f + (self.lam*logj-self.mu)[:, None, None]*np.linalg.inv(f).transpose(0, 2, 1)
        h = self.volume[:, None, None] * (p @ self.inverse.transpose(0, 2, 1))
        gradient = -loads.copy()
        for i in range(3):
            np.add.at(gradient, self.tets[:, i+1], h[:, :, i])
        np.add.at(gradient, self.tets[:, 0], -h.sum(axis=2))
        return float(self.volume @ density - np.sum(loads*(x-self.rest))), gradient

    def hessian_vector(self, x, direction):
        """Exact constitutive tangent action; can be indefinite when deformed."""
        x, direction = np.asarray(x,float), np.asarray(direction,float)
        if x.shape != self.rest.shape or direction.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(direction).all():
            raise ValueError('Invalid tangent inputs')
        q, dq = x[self.tets], direction[self.tets]
        ds = np.stack([q[:,i]-q[:,0] for i in (1,2,3)],axis=2)
        dds = np.stack([dq[:,i]-dq[:,0] for i in (1,2,3)],axis=2)
        f, df = ds@self.inverse, dds@self.inverse
        j = np.linalg.det(f)
        if np.any(j <= 0): raise ValueError('Non-positive determinant')
        inverse = np.linalg.inv(f); inverse_t = inverse.transpose(0,2,1)
        dlogj = np.trace(inverse@df,axis1=1,axis2=2)
        dp = (self.mu*df + self.lam*dlogj[:,None,None]*inverse_t -
              (self.lam*np.log(j)-self.mu)[:,None,None]*(inverse_t@df.transpose(0,2,1)@inverse_t))
        h = self.volume[:,None,None]*(dp@self.inverse.transpose(0,2,1))
        result = np.zeros_like(x)
        for i in range(3): np.add.at(result,self.tets[:,i+1],h[:,:,i])
        np.add.at(result,self.tets[:,0],-h.sum(axis=2))
        return result

    def solve(self, initial, loads, fixed, max_iterations=2000, tolerance=1e-7, floor=.01, method='gradient', contact=None, refinement_threshold=None):
        x = np.asarray(initial, float).copy()
        fixed = np.asarray(fixed, bool)
        if fixed.shape != x.shape or max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Invalid solve constraints')
        history = []
        if refinement_threshold is not None and (not np.isfinite(refinement_threshold) or refinement_threshold <= tolerance):
            raise ValueError('Refinement threshold must exceed final tolerance')
        if method not in ('gradient', 'lbfgs', 'rest_preconditioned', 'lbfgs_preconditioned', 'lbfgs_rotated'):
            raise ValueError('Unknown search method')
        pairs = []
        previous_x = previous_free = None
        # Diagonal length scale approximation; Armijo decides accepted steps.
        length = np.cbrt(self.volume.sum())
        stiffness = max(self.mu * length, 1e-12)
        status = 'iteration_limit'
        precondition = None
        if method in ('rest_preconditioned', 'lbfgs_preconditioned') and not fixed.all():
            from .elastic_preconditioner import make_preconditioner
            precondition = make_preconditioner(self, x, fixed, contact)
        def objective(positions):
            energy, gradient = self.energy_gradient(positions, loads)
            if contact is not None:
                contact_energy, contact_gradient = contact.energy_gradient(positions)
                energy += contact_energy; gradient += contact_gradient
            return energy, gradient
        for iteration in range(max_iterations):
            energy, gradient = objective(x)
            free = gradient.copy(); free[fixed] = 0
            residual = float(np.max(np.abs(free)))
            history.append(dict(energy_j=energy, residual_n=residual))
            if residual <= tolerance:
                status = 'converged'; break
            if refinement_threshold is not None and residual <= refinement_threshold:
                status = 'refinement_requested'; break
            if method == 'lbfgs_rotated' and iteration % 10 == 0:
                from .elastic_preconditioner import make_preconditioner
                precondition = make_preconditioner(self, x, fixed, contact, rotate=True)
                pairs.clear(); previous_x = previous_free = None
            direction = -free / stiffness
            if precondition is not None:
                direction = precondition(free)
            if method in ('lbfgs', 'lbfgs_preconditioned', 'lbfgs_rotated'):
                if previous_x is not None:
                    s, y = x-previous_x, free-previous_free
                    sy = float(np.sum(s*y))
                    if sy > 1e-12*np.linalg.norm(s)*np.linalg.norm(y) and sy > 0:
                        pairs.append((s, y, 1/sy))
                        pairs = pairs[-10:]
                q = free.copy(); coefficients = []
                for s, y, rho in reversed(pairs):
                    coefficient = rho*float(np.sum(s*q))
                    coefficients.append(coefficient); q -= coefficient*y
                scale = 1/stiffness
                if pairs:
                    s, y, rho = pairs[-1]
                    scale = float(np.sum(s*y)/np.sum(y*y))
                r = -precondition(q) if precondition is not None else scale*q
                for (s,y,rho), coefficient in zip(pairs, reversed(coefficients)):
                    r += s*(coefficient-rho*float(np.sum(y*r)))
                direction = -r
                direction[fixed] = 0
                if np.sum(free*direction) >= 0:
                    pairs.clear(); direction = -free/stiffness
                previous_x, previous_free = x.copy(), free.copy()
            alpha = positive_volume_step(self.rest, x, direction, self.tets, floor=floor)['alpha']
            if contact is not None:
                alpha = min(alpha, contact.feasible_step(x, direction))
            slope = float(np.sum(gradient*direction))
            for _ in range(60):
                candidate = x + alpha*direction
                trial, _ = objective(candidate)
                if trial <= energy + 1e-4*alpha*slope:
                    x = candidate; break
                alpha *= .5
            else:
                status = 'line_search_failed'; break
        energy, gradient = objective(x)
        residual_gradient = gradient.copy(); residual_gradient[fixed] = 0
        residual = float(np.max(np.abs(residual_gradient)))
        if residual <= tolerance:
            status = 'converged'
        return dict(positions=x, status=status, residual_n=residual,
                    constraint_reaction_n=np.where(fixed, gradient, 0), history=history)
