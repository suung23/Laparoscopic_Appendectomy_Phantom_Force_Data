"""Independent revolute-joint virtual-work check for a known point force."""
import numpy as np


def point_jacobian(origins,axes,point):
    origins,axes,point=map(lambda x:np.asarray(x,float),(origins,axes,point))
    if origins.ndim!=2 or origins.shape[1]!=3 or axes.shape!=origins.shape or point.shape!=(3,):
        raise ValueError('Expected Nx3 joint origins/axes and one world point')
    if not all(np.isfinite(x).all() for x in (origins,axes,point)) or not np.allclose(np.linalg.norm(axes,axis=1),1.,atol=1e-6):
        raise ValueError('Finite geometry and unit world axes required')
    return np.cross(axes,point-origins).T


def assess_torque(measured,expected,release):
    measured,expected,release=map(lambda x:np.asarray(x,float),(measured,expected,release))
    if measured.ndim!=2 or measured.shape!=expected.shape or release.shape!=measured.shape or not all(np.isfinite(x).all() for x in (measured,expected,release)):
        raise ValueError('Matched finite torque arrays required')
    error=np.linalg.norm(measured-expected,axis=1)
    tolerance=np.maximum(.0001,.01*np.linalg.norm(expected,axis=1))
    release_error=np.linalg.norm(release,axis=1)
    return dict(status='passed' if np.all(error<tolerance) and np.all(release_error<tolerance) else 'failed',
        error_nm=error.tolist(),tolerance_nm=tolerance.tolist(),release_error_nm=release_error.tolist(),
        measured_nm=measured.tolist(),expected_nm=expected.tolist(),
        scope='Geometric virtual-work reference versus projected joint-torque output; not independent force sensing or tissue calibration')
