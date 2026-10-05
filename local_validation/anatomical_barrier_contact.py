"""Quasistatic gravity/support diagnostic on the unchanged anatomical mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.plane_barrier import PlaneBarrier
from ct_appendectomy.tissue_metrics import tetra_health


def run(mesh, out, iterations):
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(mesh); rest = z['points'].astype(float); tets = z['tets']
    rest -= rest.mean(0)
    elastic = ElasticReference(rest, tets, 12000., .45)
    faces = np.concatenate([tets[:, f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if counts.max() > 2:
        raise ValueError('Nonmanifold mesh')
    surface = unique[counts == 1]
    triangles = rest[surface]
    areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)/2
    weights = np.zeros(len(rest))
    mass = np.zeros(len(rest))
    for corner in range(3):
        np.add.at(weights, surface[:,corner], areas/3)
    for corner in range(4):
        np.add.at(mass, tets[:,corner], 1050*elastic.volume/4)
    loads = np.zeros_like(rest); loads[:,1] = -9.81*mass
    plane = float(rest[:,1].min()-.0008)
    contact = PlaneBarrier([0,1,0], plane, .0008, 1e8, weights)
    started = time.perf_counter()
    solved = elastic.solve(rest, loads, np.zeros_like(rest, bool),
                           max_iterations=iterations, method='lbfgs', contact=contact)
    x = solved['positions']
    _, contact_gradient = contact.energy_gradient(x)
    gaps = x[:,1]-plane
    health = tetra_health(rest, x, tets)
    net = loads.sum(0)-contact_gradient.sum(0)
    np.savez_compressed(out/'final_state.npz', rest=rest, points=x, tets=tets,
                        loads=loads, contact_force=-contact_gradient, mass=mass)
    report = dict(status=solved['status'], elapsed_s=time.perf_counter()-started,
        scope='CPU quasistatic unpinned gravity/support diagnostic; no dynamic equivalence, friction, self contact, calibration, or clinical acceptance',
        mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((root/'ct_appendectomy'/name).read_bytes()).hexdigest()
                       for name in ('constrained_elasticity.py','plane_barrier.py','determinant_step.py')},
        parameters=dict(young_pa=12000., poisson=.45, density_kg_m3=1050.,
            activation_m=.0008, barrier_stiffness_n_m3=1e8,
            weights='one third of adjacent rest surface triangle areas in square meters',
            provenance='engineering placeholders'),
        nodes=len(rest), tetrahedra=len(tets), mass_kg=float(mass.sum()),
        residual_n=solved['residual_n'], net_force_n=net.tolist(),
        minimum_gap_m=float(gaps.min()), health=health,
        history=solved['history'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=120)
    a = p.parse_args()
    if a.iterations < 1: p.error('Positive iteration count required')
    run(a.mesh, a.out, a.iterations)
