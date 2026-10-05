"""Every-step implicit contact audit; not empirical tissue validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.implicit_reference import implicit_step, ImplicitConvergenceError
from ct_appendectomy.plane_barrier import PlaneBarrier
from ct_appendectomy.tissue_metrics import tetra_health

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps',type=int,default=30)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.steps<1: p.error('Positive step count required')
a.out.mkdir(parents=True,exist_ok=True)
mesh=root/'out/ct_appendectomy/boundary_preserved_mesh/mesh.npz'
z=np.load(mesh); rest=z['points'].astype(float); rest-=rest.mean(0); tets=z['tets']
elastic=ElasticReference(rest,tets,12000.,.45)
faces=np.concatenate([tets[:,f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
unique,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
surface=unique[counts==1]; q=rest[surface]
areas=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2
weights=np.zeros(len(rest)); mass=np.zeros(len(rest))
for i in range(3): np.add.at(weights,surface[:,i],areas/3)
for i in range(4): np.add.at(mass,tets[:,i],1050*elastic.volume/4)
loads=np.zeros_like(rest); loads[:,1]=-9.81*mass
plane=float(rest[:,1].min()-.0008)
contact=PlaneBarrier([0,1,0],plane,.0008,1e8,weights)
x=rest.copy(); velocity=x*0; dt=1/480; rows=[]; started=time.perf_counter()
report=dict(status='running',scope='Frictionless implicit Euler plane-contact diagnostic; uncalibrated material and contact, numerical damping, no self contact or surgical acceptance',
    mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),dt_s=dt,requested_steps=a.steps,
    parameters=dict(young_pa=12000.,poisson=.45,density_kg_m3=1050.,activation_m=.0008,barrier_stiffness_n_m3=1e8,tolerance_n=1e-11),
    source_sha256={name:hashlib.sha256((root/'ct_appendectomy'/name).read_bytes()).hexdigest()
        for name in ('implicit_reference.py','constrained_elasticity.py','elastic_preconditioner.py','determinant_step.py','plane_barrier.py')},rows=rows)
def save():
    report['elapsed_s']=time.perf_counter()-started
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    for tick in range(1,a.steps+1):
        result=implicit_step(elastic,x,velocity,mass,loads,dt,contact=contact)
        next_x,next_v=result['positions'],result['velocity']
        _,cg=contact.energy_gradient(next_x)
        momentum=(mass[:,None]*(next_v-velocity)).sum(0)
        impulse=dt*(loads.sum(0)-cg.sum(0))
        health=tetra_health(rest,next_x,tets)
        rows.append(dict(tick=tick,residual_n=result['residual_n'],
            newton_refinements=result.get('newton_refinements',[]),
            contact_force_n=(-cg.sum(0)).tolist(),momentum_balance_error_ns=float(np.linalg.norm(momentum-impulse)),
            minimum_gap_m=float((next_x[:,1]-plane).min()),health=health))
        x,velocity=next_x,next_v
        save()
    report['status']='passed' if all(r['minimum_gap_m']>0 and r['health']['inverted_tetrahedra']==0 and r['momentum_balance_error_ns']<1e-8 for r in rows) else 'failed'
except Exception as exc:
    report.update(status='failed',error=str(exc),failed_tick=len(rows)+1)
    if isinstance(exc,ImplicitConvergenceError):
        report['unaccepted_solve']=dict(residual_n=exc.result['residual_n'],history=exc.result['history'])
        np.savez_compressed(a.out/'unaccepted_iterate.npz',points=exc.result['positions'])
finally:
    np.savez_compressed(a.out/'last_accepted_state.npz',rest=rest,points=x,velocity=velocity,tets=tets)
    save()
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
