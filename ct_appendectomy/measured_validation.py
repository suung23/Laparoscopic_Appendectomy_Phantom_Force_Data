"""Errors against measured references in a pre-calibrated coordinate frame.

No per-frame ICP, scale fit, time warp or best-fit force scale is allowed here:
those can hide inaccurate deformation, drift or incorrect material stiffness.
"""
import numpy as np
from scipy.spatial import cKDTree


def surface_error(predicted_m, measured_m, tolerance_m=.001):
    a, b = np.asarray(predicted_m, float), np.asarray(measured_m, float)
    for points in (a,b):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('At least three finite XYZ points in metres required')
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Positive finite metric tolerance required')
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    precision, recall = np.mean(ab <= tolerance_m), np.mean(ba <= tolerance_m)
    return {'symmetric_rms_m':float(np.sqrt((np.mean(ab**2)+np.mean(ba**2))/2)),
            'prediction_to_measurement_p95_m':float(np.percentile(ab,95)),
            'measurement_to_prediction_p95_m':float(np.percentile(ba,95)),
            'precision':float(precision), 'recall':float(recall),
            'f_score':float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
            'tolerance_m':float(tolerance_m), 'alignment_fitted_during_evaluation':False}


def force_error(predicted_time_s, predicted_force_n, measured_time_s, measured_force_n):
    pt, mt = np.asarray(predicted_time_s,float), np.asarray(measured_time_s,float)
    pf, mf = np.asarray(predicted_force_n,float), np.asarray(measured_force_n,float)
    for time, force in ((pt,pf),(mt,mf)):
        if time.ndim != 1 or len(time)<2 or force.shape != (len(time),3):
            raise ValueError('Increasing timestamps and Nx3 force vectors required')
        if not np.isfinite(time).all() or not np.isfinite(force).all() or not np.all(np.diff(time)>0):
            raise ValueError('Finite values and strictly increasing timestamps required')
    if mt[0]<pt[0] or mt[-1]>pt[-1]:
        raise ValueError('Prediction must cover the full measured interval; no extrapolation')
    aligned = np.column_stack([np.interp(mt,pt,pf[:,i]) for i in range(3)])
    error = np.linalg.norm(aligned-mf,axis=1)
    return {'vector_rmse_n':float(np.sqrt(np.mean(error**2))),
            'p95_vector_error_n':float(np.percentile(error,95)),
            'max_vector_error_n':float(error.max()),
            'bias_xyz_n':np.mean(aligned-mf,axis=0).tolist(),
            'time_shift_fitted':False, 'force_scale_fitted':False,
            'sample_count':len(mt)}
