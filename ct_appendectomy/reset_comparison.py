"""Compare physical reset states using only exact rest-mesh correspondence."""
import numpy as np
from node_correspondence import node_permutation


def compare_states(first,second,tissue_tolerance_m=1e-5,joint_tolerance_rad=1e-5):
    for state in (first,second):
        for name in ['tissue','rest','joints']:
            if not np.isfinite(state[name]).all():raise ValueError('Nonfinite reset '+name)
        if np.shape(state['tissue'])!=np.shape(state['rest']):raise ValueError('Tissue/rest shape mismatch')
    if np.shape(first['joints'])!=np.shape(second['joints']):raise ValueError('Joint shape mismatch')
    permutation=node_permutation(first['rest'],first['tets'],second['rest'],second['tets'])
    distance=np.linalg.norm(second['tissue'][permutation]-first['tissue'],axis=1)
    joint=float(np.max(np.abs(second['joints']-first['joints'])))
    maximum=float(distance.max())
    return dict(status='matched' if maximum<=tissue_tolerance_m and joint<=joint_tolerance_rad else 'mismatched',
        tissue_max_m=maximum,tissue_rms_m=float(np.sqrt(np.mean(distance**2))),
        tissue_p99_m=float(np.percentile(distance,99)),joint_max_rad=joint,
        correspondence_changed_indices=int(np.sum(permutation!=np.arange(len(permutation)))),
        tissue_tolerance_m=tissue_tolerance_m,joint_tolerance_rad=joint_tolerance_rad,
        method='Exact rest geometry/connectivity correspondence; no alignment, scale fit or tolerance relaxation')
