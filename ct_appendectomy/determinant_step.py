"""Double-precision feasible-step bound for a future constrained optimizer.

Not a dynamics solver, contact constraint, or post-processing repair. The
determinant along a linear nodal update is cubic; endpoint tests alone are unsafe.
"""
import numpy as np


def positive_volume_step(rest, current, direction, tets, floor=.01, safety=.9):
    rest,current,direction=[np.asarray(x,float) for x in (rest,current,direction)]
    tets=np.asarray(tets)
    if (rest.ndim!=2 or rest.shape[1]!=3 or current.shape!=rest.shape or direction.shape!=rest.shape
        or not all(np.isfinite(x).all() for x in (rest,current,direction))
        or tets.ndim!=2 or tets.shape[1]!=4 or not len(tets) or tets.dtype.kind not in 'iu'
        or tets.min()<0 or tets.max()>=len(rest) or not 0<floor<1 or not 0<safety<1):
        raise ValueError('Invalid feasible-step inputs')
    def edges(x):
        q=x[tets];return [q[:,i]-q[:,0] for i in (1,2,3)]
    def det(a,b,c):return np.einsum('ij,ij->i',np.cross(a,b),c)
    r=det(*edges(rest))
    if np.any(r<=0):raise ValueError('Rest cells must have positive orientation')
    a,b,c=edges(current);u,v,w=edges(direction)
    coeff=np.column_stack((det(a,b,c)/r-floor,
        (det(u,b,c)+det(a,v,c)+det(a,b,w))/r,
        (det(u,v,c)+det(u,b,w)+det(a,v,w))/r,det(u,v,w)/r))
    if not np.isfinite(coeff).all() or np.any(coeff[:,0]<=0):
        raise ValueError('Initial state is not strictly feasible')
    earliest=1.;limiting=None
    # Bernstein coefficients bound the polynomial on [0,1] by their convex
    # hull. Strictly positive coefficients need no individual root solve.
    bernstein=np.column_stack((coeff[:,0],coeff[:,0]+coeff[:,1]/3,
        coeff[:,0]+2*coeff[:,1]/3+coeff[:,2]/3,coeff.sum(axis=1)))
    margin=64*np.finfo(float).eps*np.sum(np.abs(coeff),axis=1)
    certified=np.all(bernstein>margin[:,None],axis=1)
    unresolved=np.flatnonzero(~certified)
    for index in unresolved:
        co=coeff[index]
        # Keep nonzero coefficients; tiny high-order terms may still carry roots.
        degree=3
        while degree>0 and co[degree]==0:degree-=1
        if degree==0:continue
        roots=np.roots(co[:degree+1][::-1])
        for root in roots:
            if abs(root.imag)<=1e-8*max(1.,abs(root.real)) and 0<root.real<=earliest:
                earliest=float(root.real);limiting=index
    alpha=safety*earliest if limiting is not None else 1.
    # Check polynomial minima on the entire accepted interval, not just endpoints.
    for co in coeff[unresolved]:
        candidates=[0.,alpha]
        for root in np.roots([3*co[3],2*co[2],co[1]]):
            if abs(root.imag)<1e-10 and 0<root.real<alpha:candidates.append(float(root.real))
        if min(np.polynomial.polynomial.polyval(x,co) for x in candidates)<=0:
            raise RuntimeError('Numerical root bound could not be verified')
    return dict(alpha=alpha,limiting_element=limiting,volume_ratio_floor=floor,
                scope='Floating-point linear-path feasibility only; no energy or contact guarantee')
