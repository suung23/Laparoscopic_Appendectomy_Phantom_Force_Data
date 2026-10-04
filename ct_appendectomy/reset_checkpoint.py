"""Restore an explicit physical initial configuration while preserving FEM rest shape."""
import numpy as np
from node_correspondence import node_permutation
from tissue_metrics import tetra_health


def checkpoint_points(checkpoint,rest,tets):
    for name in ['rest','tissue']:
        if not np.isfinite(checkpoint[name]).all():raise ValueError('Nonfinite checkpoint')
    mapping=node_permutation(checkpoint['rest'],checkpoint['tets'],rest,tets)
    points=np.empty_like(np.asarray(rest,float))
    points[mapping]=checkpoint['tissue']
    health=tetra_health(rest,points,tets)
    if health['inverted_tetrahedra'] or health['degenerate_rest_tetrahedra']:
        raise ValueError('Checkpoint has invalid tetrahedra')
    return points,health
