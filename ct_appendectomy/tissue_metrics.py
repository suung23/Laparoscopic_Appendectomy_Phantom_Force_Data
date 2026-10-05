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


def target_patch_motion(reference,current,indices,tool_tip,tool_rotation_local_to_world,gravity_up_world):
    """Measure target-patch pose error and deformation without inferring force."""
    reference,current=np.asarray(reference,float),np.asarray(current,float)
    indices=np.asarray(indices,int).reshape(-1)
    tip,R,up=np.asarray(tool_tip,float),np.asarray(tool_rotation_local_to_world,float),np.asarray(gravity_up_world,float)
    if (reference.ndim!=2 or reference.shape[1]!=3 or current.shape!=reference.shape or
        not np.isfinite(reference).all() or not np.isfinite(current).all() or
        not len(indices) or indices.min()<0 or indices.max()>=len(current) or
        tip.shape!=(3,) or R.shape!=(3,3) or up.shape!=(3,) or
        not all(np.isfinite(x).all() for x in (tip,R,up))):raise ValueError('Invalid target-patch measurement inputs')
    if (not np.allclose(R.T@R,np.eye(3),atol=1e-5) or
        not np.isclose(np.linalg.det(R),1.,atol=1e-5) or not np.isclose(np.linalg.norm(up),1.,atol=1e-5)):
        raise ValueError('Tool rotation and gravity-up axis must be orthonormal')
    reference_center=reference[indices].mean(axis=0)
    current_center=current[indices].mean(axis=0)
    displacement=current_center-reference_center
    relative_local=R.T@(current_center-tip)
    return {'reference_center_world_m':reference_center.tolist(),
        'current_center_world_m':current_center.tolist(),
        'patch_displacement_world_m':displacement.tolist(),
        'patch_lift_m':float(displacement@up),
        'centre_in_shaft_frame_m':relative_local.tolist(),
        'distance_m':float(np.linalg.norm(relative_local))}
