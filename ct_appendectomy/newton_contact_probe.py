"""Newton VBD CUDA contact candidate: two spherical tools on atlas appendix.

Matched contact-disabled control; this is not integrated into the Isaac scene,
does not use CAD jaws, and does not establish material or surgical fidelity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from itertools import permutations
import time
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/warp'))
import warp as wp
import newton
from newton.solvers import SolverVBD
from tetra_cut import signed_volumes,boundary_faces
from tissue_metrics import tetra_health


def isotropic_lame_parameters(youngs_modulus_pa, poissons_ratio):
    """Map an isotropic Young's modulus prior to Newton's Lamé coefficients."""
    young=float(youngs_modulus_pa);nu=float(poissons_ratio)
    if not np.isfinite(young) or young<=0 or not np.isfinite(nu) or not 0<=nu<.5:
        raise ValueError('Young modulus must be positive and Poisson ratio in [0, 0.5)')
    return young/(2*(1+nu)), young*nu/((1+nu)*(1-2*nu))


def volume_mesh(source):
    mesh=trimesh.load(source,force='mesh')
    if not mesh.is_watertight:raise ValueError('Watertight appendix surface required')
    grid=mesh.voxelized(.002).fill();lookup={};corners=[];cells=[]
    for index in grid.sparse_indices:
        ids={}
        for corner in np.ndindex(2,2,2):
            key=tuple(index+corner)
            if key not in lookup:lookup[key]=len(corners);corners.append(key)
            ids[corner]=lookup[key]
        for order in permutations(range(3)):
            one=np.zeros(3,int);one[order[0]]=1
            two=one.copy();two[order[1]]=1
            cells.append([ids[(0,0,0)],ids[tuple(one)],ids[tuple(two)],ids[(1,1,1)]])
    points=trimesh.transform_points(np.asarray(corners,float)-.5,grid.transform)
    cells=np.asarray(cells)
    reverse=signed_volumes(points,cells)<0
    cells[reverse,:2]=cells[reverse,:2][:,::-1]
    boundary_faces(points,cells)
    return points,cells


def simulate(points,cells,base,contact,substeps=4,iterations=12,use_graph=False,
             youngs_modulus_pa=12000.,poissons_ratio=.45):
    builder=newton.ModelBuilder(up_axis=newton.Axis.Y,gravity=-9.81)
    k_mu,k_lambda=isotropic_lame_parameters(youngs_modulus_pa,poissons_ratio)
    builder.add_soft_mesh(pos=wp.vec3(),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(),
        vertices=[wp.vec3(*q) for q in points],indices=cells.ravel().tolist(),density=1050.,
        k_mu=k_mu,k_lambda=k_lambda,k_damp=.01,
        particle_radius=.0004,validate_mesh=True)
    distance=np.linalg.norm(points-base,axis=1)
    pin=np.flatnonzero(distance<max(.004,distance.min()+.002))
    for i in pin:
        builder.particle_flags[int(i)] &= ~int(newton.ParticleFlags.ACTIVE)
        builder.particle_mass[int(i)]=0.
    builder.add_ground_plane(height=float(points[:,1].min()-.0008))
    axis=points[distance.argmax()]-base;axis/=np.linalg.norm(axis)
    axial=(points-base)@axis
    positions=[]
    for fraction in (.4,.75):
        band=points[np.abs(axial-fraction*axial.max())<.003]
        center=band.mean(0);center[1]=band[:,1].max()+.010
        positions.append(center)
        body=builder.add_body(xform=wp.transform(wp.vec3(*center),wp.quat_identity()),is_kinematic=True)
        builder.add_shape_sphere(body,radius=.003,cfg=newton.ModelBuilder.ShapeConfig(
            has_particle_collision=contact,has_shape_collision=False,mu=.5))
    builder.color();model=builder.finalize(device='cuda:0')
    model.soft_contact_ke=1000.;model.soft_contact_kd=1.;model.soft_contact_mu=.5
    solver=SolverVBD(model,iterations=iterations,particle_enable_self_contact=False)
    pipeline=newton.CollisionPipeline(model,soft_contact_margin=.0008,enable_rigid_soft_full_surface_contact=True)
    s0,s1=model.state(),model.state();control=model.control();contacts=pipeline.contacts()
    poses=s0.body_q.numpy();start_positions=np.asarray(positions)
    kinematic_targets=wp.array(poses,dtype=wp.transform,device='cuda:0')
    worst=0;minimum=1.;trace=[];reference=None;pressed=None
    graph=None
    def integrate():
        nonlocal s0,s1
        for _ in range(substeps):
            if use_graph:wp.copy(s0.body_q,kinematic_targets)
            else:s0.body_q.assign(poses)
            s0.clear_forces();pipeline.collide(s0,contacts)
            solver.step(s0,s1,control,contacts,1/(60*substeps))
            s0,s1=s1,s0
    started=time.perf_counter()
    for frame in range(210):
        # 1 s settle; 1 s approach; .5 s hold; 1 s retreat, 60 Hz commands.
        alpha=float(np.clip((frame-60)/60,0,1)) if frame<150 else float(np.clip((210-frame)/60,0,1))
        poses[:,:3]=start_positions+[0,-.009*alpha,0]
        kinematic_targets.assign(poses)
        if use_graph and frame==1:
            with wp.ScopedCapture(device='cuda:0') as capture:integrate()
            graph=capture.graph
        if graph is not None:wp.capture_launch(graph)
        else:integrate()
        q=s0.particle_q.numpy()
        if frame==59:reference=q.copy()
        if frame==149:pressed=q.copy()
        health=tetra_health(points,q,cells)
        worst=max(worst,health['inverted_tetrahedra']);minimum=min(minimum,health['minimum_signed_volume_ratio'])
        trace.append({'frame':frame,'inverted_tetrahedra':health['inverted_tetrahedra'],
                      'min_volume_ratio':health['minimum_signed_volume_ratio']})
    wp.synchronize();elapsed=time.perf_counter()-started
    return {'contact_enabled':contact,'youngs_modulus_pa':float(youngs_modulus_pa),
        'poissons_ratio':float(poissons_ratio),'worst_inverted_tetrahedra':worst,'minimum_volume_ratio':minimum,
        'pin_drift_m':float(np.linalg.norm(q[pin]-points[pin],axis=1).max()),
        'wall_seconds_including_kernel_load':elapsed,'simulated_seconds':3.5},reference,pressed,q,trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'out/ct_appendectomy/cavity/appendix.ply')
    p.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/newton_contact')
    p.add_argument('--substeps',type=int,default=4)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--cuda-graph',action='store_true',help='Experimental negative-control path; failed contact stability validation')
    p.add_argument('--youngs-modulus-pa',type=float,default=12000.)
    p.add_argument('--poissons-ratio',type=float,default=.45)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.substeps<1 or a.iterations<1:p.error('Positive integration settings required')
    if a.cuda_graph and a.substeps%2:p.error('CUDA graph requires an even substep count for state ping-pong')
    report={'status':'starting','engine':f'Newton {newton.__version__} / Warp CUDA VBD',
            'scope':'dual spherical contact coupon','isaac_integrated':False,'cad_jaws':False,
            'substeps':a.substeps,'iterations':a.iterations,'cuda_graph':a.cuda_graph,
            'material':{'youngs_modulus_pa':a.youngs_modulus_pa,'poissons_ratio':a.poissons_ratio,
                        'interpretation':'uncalibrated isotropic parameter sweep; not a fit to appendix force-displacement data'}}
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        wp.init()
        if not wp.get_device('cuda:0').is_cuda:raise RuntimeError('CUDA required')
        points,cells=volume_mesh(a.source)
        manifest=json.loads((a.source.parent/'scene_manifest.json').read_text())
        report.update(nodes=len(points),tetrahedra=len(cells),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest())
        results=[];pressed=[]
        for contact in (False,True):
            metrics,reference,press,final,trace=simulate(points,cells,np.array(manifest['ct_anchor_m']),contact,
                a.substeps,a.iterations,a.cuda_graph,a.youngs_modulus_pa,a.poissons_ratio)
            results.append(metrics);pressed.append(press)
            np.savez_compressed(a.out/f'contact_{contact}.npz',rest=points,reference=reference,pressed=press,final=final,tetrahedra=cells)
            (a.out/f'trace_{contact}.json').write_text(json.dumps(trace))
        delta=np.linalg.norm(pressed[1]-pressed[0],axis=1)
        report.update(runs=results,contact_effect_max_m=float(delta.max()),contact_effect_p95_m=float(np.percentile(delta,95)))
        passed=all(r['worst_inverted_tetrahedra']==0 and r['pin_drift_m']<1e-6 for r in results) and delta.max()>.0002
        report['status']='passed' if passed else 'failed'
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:save()


if __name__=='__main__':main()
