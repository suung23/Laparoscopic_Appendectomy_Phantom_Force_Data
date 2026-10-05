"""Same anatomical tetrahedra in Newton VBD; diagnostic, not Isaac replacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(root/'.cache/warp'))
sys.path.insert(0,str(root/'ct_appendectomy'))
from tissue_metrics import tetra_health
import warp as wp
import newton

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--steps',type=int,default=960);p.add_argument('--iterations',type=int,default=12)
a=p.parse_args()
if a.steps<1 or a.iterations<1:p.error('Positive step and iteration counts required')
a.out.mkdir(parents=True,exist_ok=True)
z=np.load(a.mesh);rest=z['points'].copy();rest-=rest.mean(0);tets=z['tets']
report=dict(status='starting',engine='Newton '+newton.__version__,mesh_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),
    physics_dt_s=1/480,iterations=a.iterations,steps=a.steps,particle_radius_m=.0004,
    scope='Same tetrahedra/material/gravity; contact algorithm, particle radius and solver iterations differ. No force calibration or surgical acceptance.')
def save():(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    wp.init();builder=newton.ModelBuilder(up_axis=newton.Axis.Y,gravity=-9.81)
    E=12000.;nu=.45;floor=float(rest[:,1].min()-.0008)
    builder.add_soft_mesh(pos=wp.vec3(),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(),
        vertices=[wp.vec3(*x) for x in rest],indices=tets.ravel().tolist(),density=1050.,
        k_mu=E/(2*(1+nu)),k_lambda=E*nu/((1+nu)*(1-2*nu)),k_damp=0.,particle_radius=.0004,validate_mesh=True)
    builder.add_ground_plane(height=floor);builder.color();model=builder.finalize(device='cuda:0')
    actual_tets=model.tet_indices.numpy().reshape(-1,4)
    if not np.array_equal(actual_tets,tets):
        raise RuntimeError('Newton tetrahedron indexing differs from diagnostic mesh')
    actual_rest=model.particle_q.numpy()
    if not np.array_equal(actual_rest,rest.astype(np.float32)):
        raise RuntimeError('Newton rest positions differ from expected float32 conversion')
    cells=rest[tets]
    volumes=np.einsum('ij,ij->i',np.cross(cells[:,1]-cells[:,0],cells[:,2]-cells[:,0]),cells[:,3]-cells[:,0])/6
    expected_mass=np.zeros(len(rest))
    for corner in range(4):np.add.at(expected_mass,tets[:,corner],1050.*volumes/4)
    actual_mass=model.particle_mass.numpy()
    if np.any(actual_mass<=0) or not np.isfinite(actual_mass).all():
        raise RuntimeError('Nonpositive or nonfinite unconstrained particle mass')
    report['model_audit']=dict(exact_topology=True,exact_float32_rest=True,
        total_mass_kg=float(actual_mass.sum()),minimum_particle_mass_kg=float(actual_mass.min()),
        maximum_relative_lumped_mass_error=float(np.max(np.abs(actual_mass-expected_mass)/expected_mass)))
    if not np.allclose(actual_mass,expected_mass,rtol=1e-4,atol=0):
        raise RuntimeError('Solver masses disagree with density-weighted tetrahedral volume')
    model.soft_contact_ke=1000.;model.soft_contact_kd=0.;model.soft_contact_mu=.5
    solver=newton.solvers.SolverVBD(model,iterations=a.iterations,particle_enable_self_contact=False)
    pipeline=newton.CollisionPipeline(model,soft_contact_margin=.0008,enable_rigid_soft_full_surface_contact=True)
    s0,s1=model.state(),model.state();control=model.control();contacts=pipeline.contacts()
    rows=[];first=None;worst=0;minimum=1.;penetration=0.
    for tick in range(a.steps+1):
        x=s0.particle_q.numpy();health=tetra_health(rest,x,tets)
        if health['inverted_tetrahedra'] and first is None:
            first=tick
            np.savez_compressed(a.out/'first_inversion.npz',tick=tick,points=x,
                                rest=rest,tets=tets,velocity=s0.particle_qd.numpy())
        worst=max(worst,health['inverted_tetrahedra']);minimum=min(minimum,health['minimum_signed_volume_ratio'])
        depth=float(max(0,floor-x[:,1].min()));penetration=max(penetration,depth)
        if tick%60==0 or tick==a.steps:
            rows.append(dict(tick=tick,health=health,penetration_m=depth))
            np.savez_compressed(a.out/f'state_{tick}.npz',points=x,rest=rest,tets=tets,velocity=s0.particle_qd.numpy())
        if tick<a.steps:
            s0.clear_forces();pipeline.collide(s0,contacts);solver.step(s0,s1,control,contacts,1/480);s0,s1=s1,s0
    report.update(status='diagnostic_complete',first_inversion_tick=first,maximum_inversions=worst,
        minimum_volume_ratio=minimum,maximum_penetration_m=penetration,rows=rows)
except Exception as exc:
    report.update(status='failed',error=str(exc));raise
finally:save()
