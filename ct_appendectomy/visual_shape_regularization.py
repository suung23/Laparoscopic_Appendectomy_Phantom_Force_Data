"""Local rotation shape regularization for visual attachments, not vessel physics."""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import factorized

def regularize(rest,edges,target,fixed,tracking=.1,iterations=30):
    rest=np.asarray(rest,float);target=np.asarray(target,float);edges=np.asarray(edges,int)
    if target.shape!=rest.shape or not np.isfinite(target).all() or tracking<=0:raise ValueError('Invalid target or tracking')
    n=len(rest);i,j=edges.T;degree=np.bincount(edges.ravel(),minlength=n).astype(float)
    adjacency=coo_matrix((np.ones(2*len(edges)),(np.r_[i,j],np.r_[j,i])),shape=(n,n)).tocsr()
    matrix=diags(degree*(1+tracking))-adjacency
    fixed=np.asarray(fixed,int);free=np.setdiff1d(np.arange(n),fixed)
    solve=factorized(matrix[free][:,free].tocsc());x=target.copy();reference=rest[i]-rest[j]
    for _ in range(iterations):
        delta=x[i]-x[j];outer=delta[:,:,None]*reference[:,None,:]
        covariance=np.zeros((n,3,3));np.add.at(covariance,i,outer);np.add.at(covariance,j,outer)
        u,_,vt=np.linalg.svd(covariance);sign=np.linalg.det(u@vt);u[:,:,-1]*=sign[:,None];rotation=u@vt
        desired=np.einsum('nij,nj->ni',.5*(rotation[i]+rotation[j]),reference)
        rhs=tracking*degree[:,None]*target;np.add.at(rhs,i,desired);np.add.at(rhs,j,-desired)
        rhs=rhs[free]-matrix[free][:,fixed]@target[fixed]
        x[free]=np.column_stack([solve(rhs[:,axis]) for axis in range(3)])
    return x
