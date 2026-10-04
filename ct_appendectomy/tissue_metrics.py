"""Geometry-based FEM health metrics, independent of any solver API."""
import numpy as np


def tetra_health(rest, current, indices):
    rest,current=np.asarray(rest,float),np.asarray(current,float)
    indices=np.asarray(indices,int).reshape(-1,4)
    if rest.shape!=current.shape or not np.isfinite(current).all():
        raise ValueError('Invalid tissue state')
    def volume(points):
        tetra=points[indices]
        return np.einsum('ij,ij->i',np.cross(tetra[:,1]-tetra[:,0],tetra[:,2]-tetra[:,0]),tetra[:,3]-tetra[:,0])/6
    v0,v1=volume(rest),volume(current)
    valid=np.abs(v0)>1e-15
    if not valid.any(): raise ValueError('No non-degenerate rest tetrahedra')
    ratios=v1[valid]/v0[valid]
    return {'tetrahedra':int(len(indices)),'degenerate_rest_tetrahedra':int((~valid).sum()),
            'inverted_tetrahedra':int(np.sum(ratios<=0)),
            'minimum_signed_volume_ratio':float(ratios.min()),
            'median_volume_ratio':float(np.median(ratios)),
            'total_absolute_volume_ratio':float(np.abs(v1[valid]).sum()/np.abs(v0[valid]).sum()),
            'p99_absolute_volume_change':float(np.percentile(np.abs(ratios-1),99))}
