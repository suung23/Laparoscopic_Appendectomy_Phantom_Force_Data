"""Real-mesh constitutive patch audit, not a tissue calibration or contact test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference


def audit(mesh, relaxation_iterations=0):
    z = np.load(mesh)
    rest, tets = z['points'].astype(float), z['tets']
    rest -= rest.mean(axis=0)
    model = ElasticReference(rest, tets, 12000., .45)
    faces = np.concatenate([tets[:, f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    boundary = np.zeros(len(rest), bool)
    boundary[np.unique(unique[counts == 1])] = True
    if counts.max() > 2 or boundary.all():
        raise RuntimeError('Mesh lacks valid interior patch nodes')
    transform = np.array([[1.02,.02,0],[0,.98,.01],[0,0,.9]])
    x = rest @ transform
    zero = np.zeros_like(rest)
    energy, gradient = model.energy_gradient(x, zero)
    interior_residual = float(np.max(np.abs(gradient[~boundary])))
    rng = np.random.default_rng(4701)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(gradient*direction))
    derivative_checks = []
    for epsilon in (1e-7, 1e-8, 1e-9):
        difference = (model.energy_gradient(x+epsilon*direction, zero)[0] -
                      model.energy_gradient(x-epsilon*direction, zero)[0])/(2*epsilon)
        derivative_checks.append(dict(epsilon_m=epsilon, finite_difference_n=difference,
                                      absolute_error_n=abs(difference-analytic)))
    solution = model.solve(x, zero, np.repeat(boundary[:, None], 3, axis=1))
    force_balance = float(np.linalg.norm(gradient.sum(axis=0)))
    torque_balance = float(np.linalg.norm(np.cross(x, gradient).sum(axis=0)))
    passed = (interior_residual < 1e-9 and force_balance < 1e-9 and torque_balance < 1e-10
              and max(v['absolute_error_n'] for v in derivative_checks) < 1e-7
              and solution['status'] == 'converged')
    report = dict(status='passed' if passed else 'failed',
        scope='Anatomical tetrahedral constitutive patch consistency only; no contact, dynamics, fitted tissue, or visual validation',
        mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((root/'ct_appendectomy/constrained_elasticity.py').read_bytes()).hexdigest(),
        nodes=len(rest), elements=len(tets), boundary_nodes=int(boundary.sum()),
        parameters=dict(young_pa=12000., poisson=.45, provenance='engineering placeholders'),
        energy_j=energy, affine_volume_ratio=float(np.linalg.det(transform)),
        maximum_interior_residual_n=interior_residual, net_force_n=force_balance,
        net_torque_nm=torque_balance, directional_derivative_n=analytic,
        derivative_checks=derivative_checks, patch_solve_status=solution['status'])
    if relaxation_iterations:
        initial = x.copy()
        # Perturb all interior nodes smoothly; fixed surface retains affine data.
        initial[~boundary] += 2e-6*np.array([1.,-.5,.25])
        relaxed = model.solve(initial, zero, np.repeat(boundary[:,None],3,axis=1),
                              max_iterations=relaxation_iterations)
        report['interior_relaxation'] = dict(status=relaxed['status'],
            requested_iterations=relaxation_iterations,
            initial_residual_n=relaxed['history'][0]['residual_n'],
            final_residual_n=relaxed['residual_n'],
            maximum_position_error_m=float(np.linalg.norm(relaxed['positions']-x,axis=1).max()),
            history=relaxed['history'])
        if relaxed['status'] != 'converged':
            report['status'] = 'relaxation_incomplete'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--relaxation-iterations', type=int, default=0)
    args = parser.parse_args()
    if args.relaxation_iterations < 0:
        parser.error('Nonnegative iteration budget required')
    report = audit(args.mesh, args.relaxation_iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['status'] == 'passed' else 1)
