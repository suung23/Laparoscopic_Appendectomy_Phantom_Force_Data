"""Geometry-based FEM health metrics, independent of any solver API."""
import numpy as np


def step_velocity_comparison(previous,current,reported,dt):
    """Compare interval-mean displacement velocity with reported endpoint velocity.

    Their difference can include real acceleration; it is not a sensor error bound.
    """
    previous,current,reported=[np.asarray(x,float) for x in (previous,current,reported)]
    if (previous.ndim!=2 or previous.shape[1]!=3 or not len(previous) or
        current.shape!=previous.shape or reported.shape!=previous.shape or
        not all(np.isfinite(x).all() for x in (previous,current,reported)) or
        not np.isfinite(dt) or dt<=0):raise ValueError('Invalid velocity comparison sample')
    mean_velocity=(current-previous)/dt
    gap=float(np.sqrt(np.mean(np.sum((reported-mean_velocity)**2,axis=1))))
    return mean_velocity,gap


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
