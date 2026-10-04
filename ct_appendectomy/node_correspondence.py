"""Recover a pure node permutation only when exact rest geometry and topology agree."""
from collections import defaultdict
import numpy as np


def signatures(points,tets):
    coordinates=[tuple(p) for p in np.asarray(points)]
    incident=[[] for _ in coordinates]
    for tet in np.asarray(tets):
        key=tuple(sorted(coordinates[i] for i in tet))
        for i in tet:incident[i].append(key)
    return [(coordinates[i],tuple(sorted(incident[i]))) for i in range(len(coordinates))]


def node_permutation(old_points,old_tets,new_points,new_tets):
    if np.shape(old_points)!=np.shape(new_points):raise ValueError('Node count changed')
    old=signatures(old_points,old_tets);new=signatures(new_points,new_tets)
    groups=defaultdict(list)
    for i,key in enumerate(new):groups[key].append(i)
    mapping=[]
    for key in old:
        if not groups[key]:raise ValueError('Rest geometry or incident topology changed')
        mapping.append(groups[key].pop(0))
    mapping=np.asarray(mapping,int)
    canonical=lambda t:sorted(tuple(sorted(row)) for row in t)
    if canonical(mapping[np.asarray(old_tets)])!=canonical(np.asarray(new_tets)):
        raise ValueError('Ambiguous correspondence does not preserve tetrahedral connectivity')
    return mapping
