"""Synthetic layer-load -> membrane dynamics -> USD coupled reference run.
No tool contact, RL policy, or calibrated material claim.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,Vt
from membrane_reference import MembraneReference
from membrane_dynamics import MembraneDynamics
from attached_visual_layer import AttachedVisualLayer
from layer_force_mapping import pullback_forces
from build_fat_layer import evaluate

def run(scene,parent_file,layer_file,out,steps,dt=1/240,artery_binding=None):
    if not np.isfinite(dt) or dt<=0 or steps<1:raise ValueError("Positive timestep and steps required")
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    follower=AttachedVisualLayer(stage,'/SurgicalRegion/mesoappendix_FJ3397','/SurgicalRegion/authored_fat_layer',parent_file,layer_file)
    artery=AttachedVisualLayer(stage,'/SurgicalRegion/mesoappendix_FJ3397','/SurgicalRegion/appendicular_artery_FJ3410',parent_file,artery_binding) if artery_binding else None
    artery_rest=np.array(artery.layer.GetPointsAttr().Get()) if artery else None
    z=np.load(parent_file);p=z['points'];layer=np.load(layer_file)
    if p.shape!=(1625,3):raise ValueError("This commissioning setup requires the authored 25 by 65 membrane grid")
    fixed=np.r_[np.arange(65),np.arange(len(p)-65,len(p))]
    dynamics=MembraneDynamics(MembraneReference(p,z['faces'],12000,.45,.0002,1e-8),p,1000,fixed)
    patch=np.argsort(np.linalg.norm(layer['points']-p[710],axis=1))[:100]
    force=np.zeros_like(layer['points']);force[patch,2]=.0002/len(patch)
    reports=[];trajectory=[p.copy()];start=time.perf_counter();error=None
    try:
        for step in range(steps):
            current=evaluate(dynamics.points,layer['nodes'],layer['weights'],layer['frame_offsets'])
            mapped=pullback_forces(dynamics.points,layer['nodes'],layer['weights'],layer['frame_offsets'],force)
            torque_error=np.linalg.norm(np.cross(dynamics.points,mapped).sum(axis=0)-np.cross(current,force).sum(axis=0))
            info=dynamics.step(dt,mapped,maxiter=3000,tolerance=1e-6)
            mesh=trimesh.Trimesh(dynamics.points,z['faces'],process=False)
            follower.parent.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(dynamics.points.astype(np.float32)))
            follower.parent.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
            follower.parent.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.bounds.astype(np.float32)))
            if artery:
                artery_points=artery.update();info['artery_max_motion_mm']=float(np.linalg.norm(artery_points-artery_rest,axis=1).max()*1000)
            displayed=follower.update();trajectory.append(dynamics.points.copy())
            info.update(step=step,load_torque_error_Nm=float(torque_error),layer_max_motion_mm=float(np.linalg.norm(displayed-layer['points'],axis=1).max()*1000));reports.append(info)
    except Exception as exc:error=str(exc)
    stage.GetRootLayer().Save()
    np.savez_compressed(out/'trajectory.npz',points=np.array(trajectory),dt=dt)
    report={'status':'completed reference run' if error is None else 'failed','error':error,'steps':reports,'dt':dt,'simulated_seconds':len(reports)*dt,'wall_seconds':time.perf_counter()-start,'fixed_error_m':float(np.abs(dynamics.points[fixed]-p[fixed]).max()),'synthetic_force_N':force.sum(axis=0).tolist(),'scope':'CPU reference dynamics and USD display coupled; layer force lagged at step start','tool_contact':False,'RL':False,'Isaac_physics':False,'material_calibrated':False,'artery_visual_following':artery is not None,'vessel_mechanics':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--layer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--dt',type=float,default=1/240)
    p.add_argument('--artery-binding',type=Path)
    a=p.parse_args();run(a.scene,a.parent,a.layer,a.out,a.steps,a.dt,a.artery_binding)
