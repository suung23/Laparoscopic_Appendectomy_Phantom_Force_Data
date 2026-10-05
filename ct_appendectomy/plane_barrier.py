"""Frictionless plane barrier for the CPU reference optimizer.

Not calibrated contact: finite activation distance introduces a positive gap.
Per-node weights must be supplied explicitly to control mesh dependence.
"""
import numpy as np


class PlaneBarrier:
    def __init__(self, normal, offset, activation, stiffness, weights):
        self.normal = np.asarray(normal, float)
        self.weights = np.asarray(weights, float)
        if (self.normal.shape != (3,) or not np.isfinite(self.normal).all()
            or not np.isclose(np.linalg.norm(self.normal), 1., atol=1e-12, rtol=0)
            or not np.isfinite([offset, activation, stiffness]).all()
            or activation <= 0 or stiffness <= 0 or self.weights.ndim != 1
            or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
            or not np.any(self.weights > 0)):
            raise ValueError('Invalid plane barrier parameters')
        self.offset, self.activation, self.stiffness = offset, activation, stiffness

    def gaps(self, x):
        x = np.asarray(x, float)
        if x.shape != (len(self.weights), 3) or not np.isfinite(x).all():
            raise ValueError('Invalid contact positions')
        d = x @ self.normal - self.offset
        if np.any(d[self.weights > 0] <= 0):
            raise ValueError('Contact state is not strictly feasible')
        return d

    def energy_gradient(self, x):
        d = self.gaps(x)
        active = (self.weights > 0) & (d < self.activation)
        gap = d[active]; delta = gap-self.activation
        log = np.log(gap/self.activation)
        k = self.stiffness*self.weights[active]
        energy = -np.sum(k*delta**2*log)
        derivative = -k*(2*delta*log+delta**2/gap)
        gradient = np.zeros_like(x, dtype=float)
        gradient[active] = derivative[:,None]*self.normal
        return float(energy), gradient

    def feasible_step(self, x, direction, safety=.9):
        d = self.gaps(x)
        direction = np.asarray(direction, float)
        if direction.shape != np.shape(x) or not np.isfinite(direction).all() or not 0 < safety < 1:
            raise ValueError('Invalid contact direction')
        speed = direction @ self.normal
        approaching = (self.weights > 0) & (speed < 0)
        if not approaching.any():
            return 1.
        hit = float(np.min(-d[approaching]/speed[approaching]))
        return safety*hit if hit <= 1 else 1.
