"""Offline interior smoothing with fixed anatomical boundary and connectivity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def quality(x):
    d = np.einsum('ij,ij->i', np.cross(x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]), x[:, 3]-x[:, 0])
    edges = sum(np.sum((x[:, i]-x[:, j])**2, axis=1) for i in range(4) for j in range(i))
    return np.where(d > 0, 12*(np.maximum(d, 0)/2)**(2/3)/edges, 0), d


def improve(points, tets, sweeps=8):
    points = np.asarray(points, float).copy()
    tets = np.asarray(tets, int)
    _, d = quality(points[tets])
    if np.any(d <= 0):
        raise ValueError('Positive oriented nondegenerate tetrahedra required')
    faces = np.concatenate([tets[:, f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    f, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if count.max() > 2:raise ValueError('Nonmanifold mesh')
    boundary = np.unique(f[count == 1]); original_boundary = points[boundary].copy()
    incident = [[] for _ in points]
    for i, cell in enumerate(tets):
        for node in cell:incident[node].append(i)
    interior = np.setdiff1d(np.arange(len(points)), boundary)
    trace = []
    for sweep in range(sweeps):
        accepted = 0
        for node in interior:
            cells = tets[incident[node]]
            neighbors = np.setdiff1d(np.unique(cells), [node])
            if not len(neighbors):continue
            old = points[node].copy(); target = points[neighbors].mean(0)
            best = float(quality(points[cells])[0].min()); chosen = old
            for alpha in (.125, .25, .5, 1.):
                points[node] = old + alpha*(target-old)
                q, det = quality(points[cells])
                if np.all(det > 0) and float(q.min()) > best + 1e-10:
                    best = float(q.min()); chosen = points[node].copy()
            points[node] = chosen
            accepted += int(not np.array_equal(old, chosen))
        q, _ = quality(points[tets])
        trace.append(dict(sweep=sweep+1, accepted=accepted, minimum=float(q.min()),
                          below_0_1=int((q < .1).sum())))
        if not accepted:break
    if not np.array_equal(points[boundary], original_boundary):
        raise RuntimeError('Anatomical boundary moved')
    return points, trace


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();z=np.load(a.mesh);points,trace=improve(z['points'],z['tets'])
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'mesh.npz',points=points,tets=z['tets'])
    report=dict(source_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),trace=trace,
                boundary_exactly_preserved=True,connectivity_unchanged=True,
                scope='Geometric candidate only; no force or deformation validation.')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
