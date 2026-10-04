"""Isolated known-load articulation diagnostic; no tissue or clinical claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--robot-scene',type=Path,help='Extract the authored FR3 left arm only for a matched CPU/GPU test')
    p.add_argument('--reset-seed',action='store_true',help='Reproduce the task initialization calls on the extracted arm')
    p.add_argument('--keep-scene',action='store_true',help='Retain the surrounding scene for a GPU differential test')
    p.add_argument('--exclude',choices=['/World/Atlas','/World/FR3/Right'],help='Remove one scene component only in the diagnostic')
    p.add_argument('--external-every-iteration',action='store_true',help='Test the documented TGS external-force integration option')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.reset_seed and not a.robot_scene:p.error('--reset-seed requires --robot-scene')
    if a.keep_scene and not (a.robot_scene and a.gpu):p.error('--keep-scene requires --robot-scene and --gpu')
    if a.exclude and not a.keep_scene:p.error('--exclude requires --keep-scene')
    if (a.out/'report.json').exists():p.error('Use a fresh output directory')
    os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/isaac-warp'))
    os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
    from isaacsim import SimulationApp
    runtime=ROOT/'.cache/kit'
    tokens={n:runtime/n for n in ('shared_documents','app_documents','omni_documents','omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
    for path in tokens.values():path.mkdir(parents=True,exist_ok=True)
    app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','extra_args':[
        '--/app/vulkan=false','--/app/d3d12=true','--/telemetry/enableNVDF=false',
        '--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false',
        '--/privacy/extraDiagnosticDataOptIn=false','--/crashreporter/enabled=false']+
        [f'--/app/tokens/{k}={v.as_posix()}' for k,v in tokens.items()]})
    report=dict(status='starting',gpu=a.gpu,gravity_m_s2=0.,robot_collisions=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation,RigidPrim,SingleRigidPrim
        from fr3_ct_scene import body,joint
        from mount_sensor_probe import evaluate_response
        from scipy.spatial.transform import Rotation
        omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.Xform.Define(stage,'/Robot')
        T=np.eye(4)
        for name in ('base','arm','shaft'):
            prim=body(stage,'/Robot/'+name,T)
            mass=UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(.04)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        root=joint(stage,'/Robot/root',None,'/Robot/base',T,True)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        art=PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
        art.CreateSolverPositionIterationCountAttr(64)
        art.CreateSolverVelocityIterationCountAttr(8)
        joint(stage,'/Robot/hinge','/Robot/base','/Robot/arm',T,kp=5e5,kd=16650)
        joint(stage,'/Robot/mount','/Robot/arm','/Robot/shaft',T,True)
        robot_path='/Robot';mount_name='mount'
        if a.robot_scene:
            report['robot_scene_sha256']=hashlib.sha256(a.robot_scene.read_bytes()).hexdigest()
            omni.usd.get_context().open_stage(str(a.robot_scene.resolve()))
            stage=omni.usd.get_context().get_stage()
            if not a.keep_scene:
                for prim in list(stage.GetPseudoRoot().GetChildren()):
                    if str(prim.GetPath())!='/World':stage.RemovePrim(prim.GetPath())
                for prim in list(stage.GetPrimAtPath('/World').GetChildren()):
                    if prim.GetName() not in ('FR3','physicsScene'):stage.RemovePrim(prim.GetPath())
                stage.RemovePrim('/World/FR3/Right')
            report['surrounding_scene_retained']=a.keep_scene
            if a.exclude:
                if not stage.GetPrimAtPath(a.exclude):raise ValueError('Excluded component is absent')
                attachments=[]
                for prim in stage.Traverse():
                    targets=[str(t) for rel in prim.GetRelationships() if rel.GetName().startswith('omniphysics:src') for t in rel.GetTargets()]
                    if any(t==a.exclude or t.startswith(a.exclude+'/') for t in targets):attachments.append(prim.GetPath())
                for path in attachments:stage.RemovePrim(path)
                report['removed_dependent_attachments']=len(attachments)
                stage.RemovePrim(a.exclude)
            report['excluded_component']=a.exclude
            robot_path='/World/FR3/Left';mount_name='tool_mount'
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(robot_path+'/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        sim=SimulationContext(stage_units_in_meters=1.,physics_dt=1/480,rendering_dt=1/30)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr(0.)
                ps=PhysxSchema.PhysxSceneAPI.Apply(prim)
                ps.CreateEnableGPUDynamicsAttr(a.gpu)
                ps.CreateBroadphaseTypeAttr('GPU' if a.gpu else 'MBP')
                ps.CreateEnableExternalForcesEveryIterationAttr(a.external_every_iteration)
        report['external_forces_every_iteration']=a.external_every_iteration
        stage.GetRootLayer().Export(str(a.out/'scene.usda'))
        sim.initialize_physics();sim.play()
        robot=SingleArticulation(robot_path,name='fixture',reset_xform_properties=False);robot.initialize()
        link=RigidPrim(robot_path+'/shaft',name='load',reset_xform_properties=False);link.initialize()
        row=int(robot._articulation_view._metadata.joint_indices[mount_name])+1
        if a.reset_seed:
            from isaacsim.core.utils.types import ArticulationAction
            indices=np.array([robot.get_dof_index('fr3_joint'+str(i)) for i in range(1,8)])
            jaws=np.array([robot.get_dof_index('jaw_left'),robot.get_dof_index('jaw_right')])
            seed=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
            robot.set_joint_positions(seed,joint_indices=indices)
            robot.set_joint_positions(np.deg2rad([37.,-37.]),joint_indices=jaws)
        for _ in range(480):
            if a.reset_seed:robot.apply_action(ArticulationAction(joint_positions=seed,joint_velocities=np.zeros(7),joint_indices=indices))
            sim.step(render=False)
        report['reset_seed']=a.reset_seed
        report['warmup_ticks']=480
        joint_frames=[];arm_indices=[]
        if a.robot_scene:
            for i in range(1,8):
                j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(robot_path+'/fr3_joint'+str(i)))
                parent=SingleRigidPrim(str(j.GetBody0Rel().GetTargets()[0]),name='joint_parent_'+str(i),reset_xform_properties=False)
                parent.initialize()
                local_rot=j.GetLocalRot0Attr().Get()
                local_R=Rotation.from_quat([*local_rot.GetImaginary(),local_rot.GetReal()]).as_matrix()
                local_axis=local_R@np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
                joint_frames.append((parent,np.asarray(j.GetLocalPos0Attr().Get(),float),local_axis))
                arm_indices.append(robot.get_dof_index('fr3_joint'+str(i)))
        def load_jacobian():
            from static_load_check import point_jacobian
            origins=[];axes=[]
            for parent,origin,axis in joint_frames:
                p,q=parent.get_world_pose();R=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
                origins.append(p+R@origin);axes.append(R@axis)
            p,q=link.get_world_poses();R=Rotation.from_quat(np.asarray(q)[0,[1,2,3,0]]).as_matrix()
            com,_=link.get_coms()
            com=np.asarray(com)
            if com.shape!=(1,1,3):raise ValueError('Unexpected center-of-mass tensor shape')
            return point_jacobian(origins,axes,p[0]+R@com[0,0])
        torque_response=[];torque_expected=[];torque_release=[]
        response=[];noise=[];release=[];trace=[];expected=[]
        for axis in range(3):
            samples=[];efforts=[]
            for tick in range(480):
                f=np.eye(3,dtype=np.float32)[axis]*(.1 if 120<=tick<360 else 0.)
                link.apply_forces(f.reshape(1,3),is_global=True)
                sim.step(render=False)
                force=np.asarray(robot.get_measured_joint_forces())[row,:3]
                samples.append(force.copy())
                trace.append(dict(axis=axis,tick=tick,force=force.tolist(),
                    speed_m_s=float(np.linalg.norm(link.get_linear_velocities()))))
                if joint_frames:
                    tau=np.asarray(robot.get_measured_joint_efforts(joint_indices=np.array(arm_indices)),float)
                    efforts.append(tau)
                    trace[-1]['projected_joint_torque_nm']=tau.tolist()
                if tick==359:
                    _,q=link.get_world_poses()
                    R=Rotation.from_quat(np.asarray(q)[0,[1,2,3,0]]).as_matrix()
                    expected.append((R.T@(np.eye(3)[axis]*.1)).tolist())
                    if joint_frames:torque_expected.append((-load_jacobian().T@(np.eye(3)[axis]*.1)).tolist())
            values=np.array(samples);baseline=values[88:120].mean(0)
            response.append((values[328:360].mean(0)-baseline).tolist())
            noise.append(float(np.linalg.norm(values[88:120]-baseline,axis=1).max()))
            release.append(float(np.linalg.norm(values[448:480].mean(0)-baseline)))
            if efforts:
                tau=np.array(efforts);tare=tau[88:120].mean(0)
                torque_response.append((tau[328:360].mean(0)-tare).tolist())
                torque_release.append((tau[448:480].mean(0)-tare).tolist())
        if joint_frames:
            from static_load_check import assess_torque
            report['projected_torque_check']=assess_torque(torque_response,torque_expected,torque_release)
        report.update(evaluate_response(response,expected,release,noise),
            selected_row=row,trace=trace,physics_ticks=len(trace))
    except Exception as e:
        report.update(status='error',error=str(e));raise
    finally:
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        app.close()


if __name__=='__main__':main()
