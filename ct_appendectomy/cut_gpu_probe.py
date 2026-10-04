"""Newton GPU independence probe for an already split solid appendix mesh.

One half is fixed; the other receives a uniform initial velocity. This tests
mechanical disconnection after remeshing, not blade forces or live cutting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/warp'))
import warp as wp
import newton
from newton.solvers import SolverVBD
from tissue_metrics import tetra_health


def run(source,output):
    data=np.load(source)
    points=data['points'];cells=data['tetrahedra'];side=data['side']
    fixed=np.unique(cells[side<0]);free=np.unique(cells[side>0])
    if np.intersect1d(fixed,free).size:raise ValueError('Cut sides still share nodes')
    wp.init()
    device=wp.get_device('cuda:0')
    if not device.is_cuda:raise RuntimeError('CUDA required; no CPU fallback')
    output.mkdir(parents=True,exist_ok=True)
    with wp.ScopedDevice(device):
        builder=newton.ModelBuilder(up_axis=newton.Axis.Z,gravity=0.)
        young=12000.;nu=.45
        builder.add_soft_mesh(pos=wp.vec3(0,0,0),rot=wp.quat_identity(),scale=1.,
            vel=wp.vec3(0,0,.01),vertices=[wp.vec3(*q) for q in points],
            indices=cells.ravel().tolist(),density=1050.,
            k_mu=young/(2*(1+nu)),k_lambda=young*nu/((1+nu)*(1-2*nu)),k_damp=.01)
        for identity in fixed:
            builder.particle_flags[int(identity)] &= ~int(newton.ParticleFlags.ACTIVE)
            builder.particle_mass[int(identity)]=0.
            builder.particle_qd[int(identity)]=wp.vec3(0,0,0)
        builder.color()
        model=builder.finalize(device=device)
        solver=SolverVBD(model,iterations=8,particle_enable_self_contact=False)
        s0,s1=model.state(),model.state()
        control,contacts=model.control(),model.contacts()
        rest=s0.particle_q.numpy().copy()
        started=time.perf_counter()
        for _ in range(480):
            s0.clear_forces()
            model.collide(s0,contacts)
            solver.step(s0,s1,control,contacts,1/240)
            s0,s1=s1,s0
        wp.synchronize()
        final=s0.particle_q.numpy()
        elapsed=time.perf_counter()-started
    health=tetra_health(rest,final,cells)
    drift=float(np.linalg.norm(final[fixed]-rest[fixed],axis=1).max())
    translation=(final[free]-rest[free]).mean(axis=0)
    report={'engine':'Newton 1.4 / Warp CUDA VBD','scope':'post-cut disconnected solid FEM coupon',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'device':device.name,'simulated_seconds':2.,'wall_seconds_including_kernel_load':elapsed,
            'fixed_half_max_drift_m':drift,'free_half_mean_displacement_m':translation.tolist(),
            'expected_free_displacement_m':[0,0,.02],'tissue_health':health,
            'live_cut_during_step':False,'isaac_coupling':False,'appendectomy_completed':False}
    passed=np.isfinite(final).all() and drift<1e-6 and np.linalg.norm(translation-[0,0,.02])<.002 and health['inverted_tetrahedra']==0
    report['status']='passed' if passed else 'failed'
    np.savez_compressed(output/'post_cut_state.npz',rest=rest,current=final,tetrahedra=cells,side=side)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('Post-cut GPU independence probe failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.out)
