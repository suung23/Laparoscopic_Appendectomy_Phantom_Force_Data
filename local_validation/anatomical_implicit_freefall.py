"""Short real-mesh implicit freefall audit, no contact acceptance."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.implicit_reference import implicit_step
from ct_appendectomy.tissue_metrics import tetra_health

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tolerance',type=float,default=1e-7)
p.add_argument('--out',type=Path,default=root/'validation/anatomical-implicit-freefall.json')
args=p.parse_args()

mesh = root/'out/ct_appendectomy/boundary_preserved_mesh/mesh.npz'
z = np.load(mesh); rest = z['points'].astype(float); rest -= rest.mean(0)
tets = z['tets']; elastic = ElasticReference(rest,tets,12000.,.45)
mass = np.zeros(len(rest))
for corner in range(4): np.add.at(mass,tets[:,corner],1050*elastic.volume/4)
x = rest.copy(); velocity = x*0; gravity = np.array([0.,-9.81,0]); dt = 1/480
rows = []; started = time.perf_counter()
for tick in range(1,11):
    solved = implicit_step(elastic,x,velocity,mass,mass[:,None]*gravity,dt,tolerance=args.tolerance)
    x,velocity = solved['positions'],solved['velocity']
    expected = rest + gravity*dt**2*tick*(tick+1)/2
    rows.append(dict(tick=tick, residual_n=solved['residual_n'],
        position_error_m=float(np.linalg.norm(x-expected,axis=1).max()),
        velocity_error_m_s=float(np.linalg.norm(velocity-tick*dt*gravity,axis=1).max()),
        health=tetra_health(rest,x,tets)))
report = dict(status='passed' if all(r['position_error_m']<1e-7 and r['velocity_error_m_s']<1e-5 and r['health']['inverted_tetrahedra']==0 for r in rows) else 'failed',
    scope='10-step freefall only; no contact, physical damping, empirical mechanics or realtime acceptance',
    method='implicit Euler', tolerance_n=args.tolerance, dt_s=dt, elapsed_s=time.perf_counter()-started,
    mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
    source_sha256={name:hashlib.sha256((root/'ct_appendectomy'/name).read_bytes()).hexdigest()
        for name in ('implicit_reference.py','constrained_elasticity.py','elastic_preconditioner.py','determinant_step.py')},rows=rows)
args.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
