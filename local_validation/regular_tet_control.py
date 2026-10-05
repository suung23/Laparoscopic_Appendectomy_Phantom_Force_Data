"""Reproducible numerical control mesh; never an anatomical asset replacement."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def build(n=5, spacing=.004):
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError('At least two integer grid points per axis required')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Positive finite spacing required')
    points = np.array(list(itertools.product(range(n), repeat=3)), float) * spacing
    tets = []
    for origin in itertools.product(range(n - 1), repeat=3):
        for order in itertools.permutations(range(3)):
            q = list(origin)
            cell = [q[0] * n * n + q[1] * n + q[2]]
            for axis in order:
                q[axis] += 1
                cell.append(q[0] * n * n + q[1] * n + q[2])
            tets.append(cell)
    tets = np.array(tets, dtype=np.int64)
    x = points[tets]
    d = np.einsum('ij,ij->i', np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
    tets[d < 0] = tets[d < 0][:, [1, 0, 2, 3]]
    faces = np.concatenate([tets[:, f] for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if counts.max() > 2 or not np.isclose(abs(d).sum() / 6, ((n - 1) * spacing)**3):
        raise RuntimeError('Invalid control mesh')
    return points, tets


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    points, tets = build()
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'mesh.npz', points=points, tets=tets)
    (a.out / 'provenance.json').write_text(json.dumps(dict(
        generator='regular_tet_control.build', n=5, spacing_m=.004,
        scope='Numerical control only; no clinical anatomy'), indent=2) + '\n')
