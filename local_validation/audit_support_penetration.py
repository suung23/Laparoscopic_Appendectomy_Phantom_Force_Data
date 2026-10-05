"""Audit live tissue snapshots against a known axis-aligned support box.

This is a geometric diagnostic, not contact force or clinical validation.
Bounds must describe the actual world-space collider, not its local mesh.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(points, tets, lower, upper, tolerance):
    points = np.asarray(points, float)
    raw_tets = np.asarray(tets)
    if raw_tets.dtype.kind not in 'iu' or raw_tets.ndim != 2 or raw_tets.shape[1] != 4:
        raise ValueError('Expected integer tetrahedron indices')
    tets = raw_tets
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or not len(tets) or
            not np.isfinite(points).all() or lower.shape != (3,) or
            upper.shape != (3,) or not np.isfinite(lower).all() or
            not np.isfinite(upper).all() or not np.all(upper > lower) or
            tets.min() < 0 or tets.max() >= len(points) or
            not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Invalid geometry or tolerance')
    faces = np.concatenate([tets[:, f] for f in
                            ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))])
    unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold tetrahedral mesh')
    boundary = np.unique(unique[counts == 1])
    # Distance to the nearest box face; positive only inside every face.
    depth = np.minimum(points - lower, upper - points).min(axis=1)
    inside = depth > tolerance
    return dict(nodes=len(points), boundary_nodes=len(boundary),
                penetrating_nodes=int(inside.sum()),
                penetrating_boundary_nodes=int(inside[boundary].sum()),
                maximum_interior_distance_m=float(max(0, depth.max())),
                scope='Node samples only; does not exclude element-face crossings between nodes.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--bounds', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=0.0001)
    a = p.parse_args()
    b = json.loads(a.bounds.read_text())
    phases = []
    for name in ['before_initialize', 'after_initialize', 'before_warmup',
                 'warmup_1', 'warmup_60', 'warmup_480',
                 'after_settlement', 'settlement_timeout']:
        file = a.run / f'tissue_{name}.npz'
        if not file.exists():
            continue
        metadata = json.loads(file.with_suffix('.json').read_text())
        source = metadata.get('observation_source')
        if source != 'physx_tensor' and not (name == 'before_initialize' and source == 'usd'):
            raise ValueError('Requires live PhysX tensor observations, not potentially stale USD points')
        with np.load(file) as z:
            row = audit(z['points'], z['tets'], b['support_min_m'],
                        b['support_max_m'], a.tolerance)
        phases.append(dict(phase=name, observation_source=source, **row))
    if not phases:
        raise ValueError('No tissue snapshots found')
    result = dict(status='penetration_detected' if any(x['penetrating_nodes'] for x in phases)
                  else 'no_sampled_penetration', run=a.run.as_posix(),
                  tolerance_m=a.tolerance, bounds=b, phases=phases,
                  limitations='Axis-aligned static box only; no force attribution or mechanical acceptance.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], phases=len(phases))))


if __name__ == '__main__':
    main()
