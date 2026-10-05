"""Exact-tangent preconditioned CG direction for unconstrained reference states.

Caller must check feasibility and globalization. CG success alone is not proof
of positive definiteness; reject non-descent directions explicitly.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from .elastic_preconditioner import make_preconditioner


def newton_direction(elastic, positions, gradient, potential):
    shape = positions.shape
    auxiliary = potential.hessian(positions)
    def action(vector):
        return elastic.hessian_vector(positions,vector.reshape(shape)).ravel()+auxiliary@vector
    precondition = make_preconditioner(elastic,positions,np.zeros(shape,bool),potential,rotate=True)
    operator = LinearOperator((positions.size,positions.size),matvec=action,dtype=float)
    metric = LinearOperator(operator.shape,matvec=lambda v:-precondition(v.reshape(shape)).ravel(),dtype=float)
    calls = [0]
    def count(_): calls[0] += 1
    direction,info = cg(operator,-gradient.ravel(),M=metric,rtol=1e-6,atol=0,maxiter=300,callback=count)
    residual = float(np.linalg.norm(action(direction)+gradient.ravel()))
    slope = float(gradient.ravel()@direction)
    if info != 0 or not np.isfinite(direction).all() or slope >= 0:
        raise RuntimeError(f'Newton direction rejected: cg={info}, slope={slope}')
    return direction.reshape(shape),dict(cg_iterations=calls[0],linear_residual_n=residual,slope_j=slope)
