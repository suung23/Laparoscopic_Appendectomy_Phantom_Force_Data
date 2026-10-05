"""Layer-load / fixed CAD penalty -> membrane dynamics -> USD reference run.
No controlled robot contact, RL policy, or calibrated material claim.
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

def run(scene,parent_file,layer_file,out,steps,dt=1/240,artery_binding=None,artery_shape=False,cad=None,approach=None,full_tool_contact=False,friction_mu=0.,solver_tolerance=1e-6,defect_tolerance=None):
    if not np.isfinite(dt) or dt<=0 or steps<1:raise ValueError("Positive timestep and steps required")
    if not np.isfinite(solver_tolerance) or solver_tolerance<=0:raise ValueError('Positive solver tolerance required')
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    follower=AttachedVisualLayer(stage,'/SurgicalRegion/mesoappendix_FJ3397','/SurgicalRegion/authored_fat_layer',parent_file,layer_file)
    artery=AttachedVisualLayer(stage,'/SurgicalRegion/mesoappendix_FJ3397','/SurgicalRegion/appendicular_artery_FJ3410',parent_file,artery_binding) if artery_binding else None
    if artery:artery.shape_regularization=artery_shape
    artery_rest=np.array(artery.layer.GetPointsAttr().Get()) if artery else None
    z=np.load(parent_file);p=z['points'];layer=np.load(layer_file)
    if p.shape!=(1625,3):raise ValueError("This commissioning setup requires the authored 25 by 65 membrane grid")
    fixed=np.r_[np.arange(65),np.arange(len(p)-65,len(p))]
    dynamics=MembraneDynamics(MembraneReference(p,z['faces'],12000,.45,.0002,1e-8),p,1000,fixed)
    patch=np.argsort(np.linalg.norm(layer['points']-p[710],axis=1))[:100]
    force=np.zeros_like(layer['points']);force[patch,2]=.0002/len(patch)
    jaw=None;poses=None;local_parts=None;parts=None;jaw_angles=None
    if full_tool_contact and (not cad or not approach):raise ValueError('Full tool contact requires CAD and approach')
    if not np.isfinite(friction_mu) or friction_mu<0 or (friction_mu and not full_tool_contact):raise ValueError('Friction requires full tool contact and nonnegative coefficient')
    if approach and not cad:raise ValueError("Approach requires CAD")
    if cad:
        from cad_penalty_contact import load_jaw,penalty
        from build_scene import mesh_prim
        local_jaw=load_jaw(cad);jaw=local_jaw.copy()
        if full_tool_contact:
            from cad_penalty_contact import compound_penalty
            from pxr import UsdGeom
            source=Usd.Stage.Open(str(cad.resolve()))
            names=[p.GetName() for p in source.GetPrimAtPath('/Tool').GetChildren() if UsdGeom.Mesh(source.GetPrimAtPath(str(p.GetPath())+'/mesh'))]
            local_parts={name:load_jaw(cad,name) for name in names}
        if approach:
            data=np.load(approach);poses=data['poses']
            if 'jaw_half_angles' in data:
                if not full_tool_contact:raise ValueError('Articulated trajectory requires full tool contact')
                jaw_angles=data['jaw_half_angles']
                if jaw_angles.shape!=(len(poses),) or not np.isfinite(jaw_angles).all() or (jaw_angles<0).any() or (jaw_angles>np.deg2rad(37)).any():raise ValueError('Invalid jaw angles')
                if len(jaw_angles)>1 and np.abs(np.diff(jaw_angles)).max()/dt>.6000001:raise ValueError('Jaw angular speed exceeds legacy control limit')
            if 'dt' in data and not np.isclose(float(data['dt']),dt,rtol=0,atol=1e-12):raise ValueError('Trajectory timestep mismatch')
            if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses) or not np.isfinite(poses).all():raise ValueError('Invalid approach poses')
            if not np.allclose(data['jaw_points'],local_jaw.vertices) or not np.array_equal(data['jaw_faces'],local_jaw.faces):raise ValueError('Approach CAD mismatch')
            axes=poses[:,:3,:3]
            if not np.allclose(axes.transpose(0,2,1)@axes,np.eye(3),atol=1e-8) or not np.allclose(np.linalg.det(axes),1,atol=1e-8):raise ValueError('Invalid rotations')
            delta=poses[:,:3,3]-data['port'];direction=poses[:,:3,2];insertion=np.sum(delta*direction,axis=1)
            if np.linalg.norm(np.cross(delta,direction),axis=1).max()>1e-8 or (insertion<.03).any() or (insertion>.25).any():raise ValueError('Invalid pivot or insertion')
            if len(poses)>1 and np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).max()/dt>.02000001:raise ValueError('Insertion speed exceeds limit')
            from rosota_controller import aperture_metrics
            if not all(aperture_metrics(data['port'],data['port_axes'][:,2],pose[:3,3],pose[:3,2])['aperture_clear'] for pose in poses):raise ValueError('Trocar aperture violation')
            if len(poses)>1:
                angles=np.arccos(np.clip((np.trace(axes[:-1].transpose(0,2,1)@axes[1:],axis1=1,axis2=2)-1)/2,-1,1))
                if angles.max()/dt>.4000001:raise ValueError('Rotation speed exceeds limit')
            jaw.vertices=local_jaw.vertices@poses[0,:3,:3].T+poses[0,:3,3]
        else:jaw.apply_translation(layer['points'][710]-jaw.centroid)
        tissue=trimesh.Trimesh(layer['points'],layer['faces'],process=False)
        area=np.zeros(len(layer['points']));np.add.at(area,tissue.faces.ravel(),np.repeat(tissue.area_faces/3,3))
        mesh_prim(stage,'/World/FixedCADContactJaw',jaw,(.45,.48,.5),roughness=.2)
    reports=[];trajectory=[p.copy()];start=time.perf_counter();error=None
    previous_layer=None;previous_transforms=None
    try:
        for step in range(steps):
            if poses is not None:
                pose=poses[min(step,len(poses)-1)]
                jaw.vertices=local_jaw.vertices@pose[:3,:3].T+pose[:3,3]
                if local_parts is not None:
                    from cad_penalty_contact import part_pose
                    parts={name:mesh.copy() for name,mesh in local_parts.items()}
                    half_angle=float(jaw_angles[min(step,len(jaw_angles)-1)]) if jaw_angles is not None else 0.
                    transforms={}
                    for name,mesh in parts.items():
                        transform=part_pose(pose,name,half_angle)
                        transforms[name]=transform
                        mesh.vertices=local_parts[name].vertices@transform[:3,:3].T+transform[:3,3]
                mesh_prim(stage,'/World/FixedCADContactJaw',jaw,(.45,.48,.5),roughness=.2)
            current=evaluate(dynamics.points,layer['nodes'],layer['weights'],layer['frame_offsets'])
            relative=None
            if parts is not None and friction_mu:
                tissue_velocity=(current-previous_layer)/dt if previous_layer is not None else np.zeros_like(current)
                relative={}
                for name,transform in transforms.items():
                    old=previous_transforms[name] if previous_transforms is not None else transform
                    local=(current-transform[:3,3])@transform[:3,:3]
                    old_world=local@old[:3,:3].T+old[:3,3]
                    relative[name]=tissue_velocity-(current-old_world)/dt
            previous_layer=current.copy()
            if parts is not None:previous_transforms=transforms
            contact_before=None
            if jaw is not None:force,contact_before=compound_penalty(parts,current,area,1e5,relative,friction_mu) if parts is not None else penalty(jaw,current,area,1e5)
            if poses is not None and step==0 and contact_before["active_nodes"]:raise ValueError("Approach begins in sampled contact")
            mapped=pullback_forces(dynamics.points,layer['nodes'],layer['weights'],layer['frame_offsets'],force)
            torque_error=np.linalg.norm(np.cross(dynamics.points,mapped).sum(axis=0)-np.cross(current,force).sum(axis=0))
            info=dynamics.step(dt,mapped,maxiter=3000,tolerance=solver_tolerance,defect_tolerance=defect_tolerance)
            mesh=trimesh.Trimesh(dynamics.points,z['faces'],process=False)
            follower.parent.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(dynamics.points.astype(np.float32)))
            follower.parent.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
            follower.parent.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.bounds.astype(np.float32)))
            if artery:
                artery_points=artery.update();info['artery_max_motion_mm']=float(np.linalg.norm(artery_points-artery_rest,axis=1).max()*1000)
            displayed=follower.update();trajectory.append(dynamics.points.copy())
            if jaw is not None:
                _,contact_after=compound_penalty(parts,displayed,area,1e5) if parts is not None else penalty(jaw,displayed,area,1e5)
                info.update(contact_before=contact_before,contact_after=contact_after)
            info.update(step=step,load_torque_error_Nm=float(torque_error),layer_max_motion_mm=float(np.linalg.norm(displayed-layer['points'],axis=1).max()*1000));reports.append(info)
    except Exception as exc:error=str(exc)
    stage.GetRootLayer().Save()
    np.savez_compressed(out/'trajectory.npz',points=np.array(trajectory),dt=dt)
    report={'status':'completed reference run' if error is None else 'failed','error':error,'steps':reports,'dt':dt,'simulated_seconds':len(reports)*dt,'wall_seconds':time.perf_counter()-start,'fixed_error_m':float(np.abs(dynamics.points[fixed]-p[fixed]).max()),'last_layer_force_N':force.sum(axis=0).tolist(),'force_source':'CAD area-weighted penalty' if jaw is not None else 'synthetic nodal load','scope':'CPU reference dynamics and USD display coupled; layer force lagged at step start','tool_contact':jaw is not None,'RCM_controlled':False,'kinematic_pivot_trajectory':poses is not None,'friction':False,'collision_geometry':'convex CAD hull' if jaw is not None else None,'contact_stiffness_N_per_m3':1e5 if jaw is not None else None,'tool_pose':('recorded kinematic pivot trajectory; no FR3 controller' if poses is not None else 'fixed synthetic placement') if jaw is not None else None,'RL':False,'Isaac_physics':False,'material_calibrated':False,'artery_visual_following':artery is not None,'vessel_mechanics':False,'artery_shape_regularization':artery_shape}
    report.update(full_tool_contact=full_tool_contact,compound_contact_model='additive part potentials; overlapping hulls may over-stiffen' if full_tool_contact else None)
    report['articulated_jaw_trajectory']=jaw_angles is not None
    report['solver_tolerance_N']=solver_tolerance
    report['inertial_defect_tolerance_m']=defect_tolerance
    report.update(friction=bool(friction_mu),friction_mu=friction_mu,friction_model='lagged regularized sliding Coulomb; no static friction',friction_regularization_speed_m_s=.001)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--layer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--dt',type=float,default=1/240)
    p.add_argument('--artery-binding',type=Path)
    p.add_argument('--artery-shape',action='store_true')
    p.add_argument('--cad',type=Path)
    p.add_argument('--approach',type=Path)
    p.add_argument('--full-tool-contact',action='store_true')
    p.add_argument('--friction-mu',type=float,default=0.)
    p.add_argument('--solver-tolerance',type=float,default=1e-6)
    p.add_argument('--defect-tolerance',type=float,default=None)
    a=p.parse_args();run(a.scene,a.parent,a.layer,a.out,a.steps,a.dt,a.artery_binding,a.artery_shape,a.cad,a.approach,a.full_tool_contact,a.friction_mu,a.solver_tolerance,a.defect_tolerance)
