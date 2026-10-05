"""Unit-covariance diagnostic for isolated appendix free fall, not a surgery task."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mesh',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--units',type=float,nargs='+',default=[1.,.01,.1],help='Meters per simulation length unit')
p.add_argument('--tensor-readback',action='store_true',help='Compare independent PhysX tensor and USD buffers')
p.add_argument('--solver-iterations',type=int,default=64)
p.add_argument('--support-gap',type=float,help='Diagnostic horizontal plane gap in meters; disables free-fall acceptance')
p.add_argument('--steps',type=int,default=180,help='Physics steps, a multiple of 60 and at least 180')
p.add_argument('--physics-hz',type=int,choices=[480,960,1920],default=480)
p.add_argument('--audit-every-step',action='store_true',help='Track transient health using live GPU observations at every physics step')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if any(v<=0 for v in a.units):p.error('Units must be positive')
if not 1<=a.solver_iterations<=255:p.error('Solver iterations must be within [1,255]')
if a.steps<180 or a.steps%60:p.error('Steps must be a multiple of 60 and at least 180')
if a.audit_every_step and not a.tensor_readback:p.error('Every-step audit requires live tensor readback')
if a.support_gap is not None and (not __import__('math').isfinite(a.support_gap) or a.support_gap<0):p.error('Nonnegative finite support gap required')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'ct_appendectomy'))
from isaacsim import SimulationApp
os.environ.setdefault('WARP_CACHE_PATH',str(root/'.cache/isaac-warp'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(root/'.cache/optix'))
tokens={n:root/'.cache/kit'/n for n in ('shared_documents','app_documents','omni_documents','omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
for path in tokens.values():path.mkdir(parents=True,exist_ok=True)
app=SimulationApp({'headless':True,'extra_args':['--/app/vulkan=false','--/app/d3d12=true',
    '--/telemetry/enableNVDF=false','--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false',
    '--/privacy/extraDiagnosticDataOptIn=false','--/crashreporter/enabled=false']+
    [f'--/app/tokens/{k}={v.as_posix()}' for k,v in tokens.items()]})
report=dict(status='running',mesh_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),cases=[],
    solver_iterations=a.solver_iterations,
    physics_steps=a.steps,physics_dt_s=1/a.physics_hz,sample_interval_s=60/a.physics_hz,
    tensor_pipeline='warp_cuda' if a.tensor_readback else 'default_cpu',
    scope='Isolated free fall with zero damping, no attachments, robot, contact or rendering. Not loaded tissue validation.',
    material_si=dict(density_kg_m3=1050,youngs_modulus_pa=12000,poissons_ratio=.45),
    gates=dict(max_translation_removed_rms_m=1e-5,max_acceleration_error_m_s2=.1,minimum_volume_ratio=.99))
if a.support_gap is not None:
    report.update(scope='Isolated plane contact diagnostic; not free fall, equilibrium, force calibration or surgical acceptance.',support_gap_m=a.support_gap)
try:
    import numpy as np
    import omni.usd
    from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Sdf,Gf,Vt
    from omni.physx.scripts import deformableUtils as du
    from isaacsim.core.api import SimulationContext
    from tissue_metrics import tetra_health
    data=np.load(a.mesh);rest=np.asarray(data['points'],float);rest-=rest.mean(0);tets=np.asarray(data['tets'],int)
    for mpu in a.units:
        print(f'BEGIN_UNIT_CASE meters_per_unit={mpu}',flush=True)
        SimulationContext.clear_instance();omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage();UsdGeom.SetStageMetersPerUnit(stage,mpu);UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
        UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        mesh=UsdGeom.TetMesh.Define(stage,'/World/Tissue')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((rest/mpu).astype(np.float32)))
        mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(tets.astype(np.int32)))
        if not du.set_physics_volume_deformable_body(stage,Sdf.Path('/World/Tissue')):raise RuntimeError('Deformable authoring failed')
        prim=mesh.GetPrim()
        if not prim.ApplyAPI('PhysxBaseDeformableBodyAPI'):
            raise RuntimeError('Cannot apply PhysX deformable parameter schema')
        prim.CreateAttribute('physxDeformableBody:solverPositionIterationCount',Sdf.ValueTypeNames.UInt).Set(a.solver_iterations)
        prim.CreateAttribute('physxDeformableBody:linearDamping',Sdf.ValueTypeNames.Float).Set(0.)
        # A free-fall covariance test must not introduce unit-dependent default
        # settling/sleep thresholds or hidden settling damping.
        prim.CreateAttribute('physxDeformableBody:settlingDamping',Sdf.ValueTypeNames.Float).Set(0.)
        prim.CreateAttribute('physxDeformableBody:settlingThreshold',Sdf.ValueTypeNames.Float).Set(0.)
        prim.CreateAttribute('physxDeformableBody:sleepThreshold',Sdf.ValueTypeNames.Float).Set(0.)
        prim.CreateAttribute('physxDeformableBody:selfCollision',Sdf.ValueTypeNames.Bool).Set(False)
        du.add_deformable_material(stage,'/World/Material',density=1050*mpu**3,youngs_modulus=12000*mpu,poissons_ratio=.45)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/Material'),UsdShade.Tokens.weakerThanDescendants,'physics')
        if a.support_gap is not None:
            floor_y=float(rest[:,1].min()-a.support_gap)
            floor=UsdGeom.Plane.Define(stage,'/World/Support')
            floor.CreateAxisAttr('Y');floor.AddTranslateOp().Set(Gf.Vec3d(0,floor_y/mpu,0))
            UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
            for collider,offset in [(prim,.0005),(floor.GetPrim(),.0003)]:
                api=PhysxSchema.PhysxCollisionAPI.Apply(collider)
                api.CreateContactOffsetAttr(offset/mpu);api.CreateRestOffsetAttr(0.)
        pipeline=dict(backend='warp',device='cuda:0') if a.tensor_readback else {}
        sim=SimulationContext(stage_units_in_meters=mpu,physics_dt=1/a.physics_hz,rendering_dt=1/30,**pipeline)
        scene=UsdPhysics.Scene.Get(stage,sim.get_physics_context().prim_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr(9.81/mpu)
        ps=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());ps.CreateEnableGPUDynamicsAttr(True);ps.CreateBroadphaseTypeAttr('GPU')
        stage.GetRootLayer().Export(str(a.out/f'scene_{mpu}.usda'))
        sim.initialize_physics();sim.play()
        if not np.array_equal(np.asarray(mesh.GetTetVertexIndicesAttr().Get(),int),tets):
            raise RuntimeError('Simulation changed mesh indexing; do not compare mismatched nodes')
        baseline=np.asarray(mesh.GetPointsAttr().Get(),float)*mpu
        if a.tensor_readback:
            from tissue_observer import TissueTensorObserver
            observer=TissueTensorObserver.bind(stage,'/World/Tissue','/World/Tissue',omni.usd.get_context().get_stage_id())
            baseline=observer.snapshot(omni.usd.get_context().get_stage_id())['position_m']
        rows=[];centers=[]
        transient=dict(samples=0,first_inversion_tick=None,maximum_inverted_elements=0,
                       minimum_signed_volume_ratio=1.,maximum_plane_penetration_m=0.)
        for tick in range(a.steps+1):
            every_state=observer.snapshot(omni.usd.get_context().get_stage_id()) if a.audit_every_step else None
            if every_state is not None:
                health=tetra_health(rest,every_state['position_m'],tets)
                transient['samples']+=1
                if health['inverted_tetrahedra'] and transient['first_inversion_tick'] is None:
                    transient['first_inversion_tick']=tick
                    np.savez_compressed(a.out/f'first_inversion_{mpu}.npz',tick=tick,
                        points=every_state['position_m'],velocity=every_state['velocity_m_s'],rest=rest,tets=tets)
                transient['maximum_inverted_elements']=max(transient['maximum_inverted_elements'],health['inverted_tetrahedra'])
                transient['minimum_signed_volume_ratio']=min(transient['minimum_signed_volume_ratio'],health['minimum_signed_volume_ratio'])
                if a.support_gap is not None:
                    transient['maximum_plane_penetration_m']=max(transient['maximum_plane_penetration_m'],float(floor_y-every_state['position_m'][:,1].min()))
            if tick%60==0:
                usd_points=np.asarray(mesh.GetPointsAttr().Get(),float)*mpu
                live=every_state if every_state is not None else (observer.snapshot(omni.usd.get_context().get_stage_id()) if a.tensor_readback else None)
                points=live['position_m'] if live is not None else usd_points
                displacement=points-baseline;translation=displacement.mean(0)
                rms=float(np.sqrt(np.mean(np.sum((displacement-translation)**2,axis=1))))
                rows.append(dict(tick=tick,health=tetra_health(rest,points,tets),translation_removed_rms_m=rms))
                if a.support_gap is not None:
                    depth=floor_y-points[:,1]
                    rows[-1].update(maximum_plane_penetration_m=float(max(0,depth.max())),
                                    nodes_below_plane_over_0_1mm=int((depth>.0001).sum()))
                if a.tensor_readback:
                    discrepancy=np.linalg.norm(points-usd_points,axis=1)
                    rows[-1].update(usd_health=tetra_health(rest,usd_points,tets),
                        tensor_usd_max_difference_m=float(discrepancy.max()),
                        tensor_usd_different_nodes=int(np.sum(discrepancy>1e-7)))
                    np.savez_compressed(a.out/f'tensor_{mpu}_{tick}.npz',points=points,velocity=live['velocity_m_s'],rest=rest,tets=tets)
                centers.append(points.mean(0))
                np.savez_compressed(a.out/f'state_{mpu}_{tick}.npz',points=points,rest=rest,tets=tets)
            if tick<a.steps:sim.step(render=False)
        acceleration=np.diff(np.asarray(centers),n=2,axis=0)/((60/a.physics_hz)**2)
        error=np.linalg.norm(acceleration-np.array([0,-9.81,0]),axis=1)
        passed=all(r['health']['inverted_tetrahedra']==0 and r['health']['minimum_signed_volume_ratio']>=.99 and r['translation_removed_rms_m']<=1e-5 for r in rows) and bool(np.max(error)<=.1)
        if a.audit_every_step:passed=passed and transient['maximum_inverted_elements']==0 and transient['minimum_signed_volume_ratio']>=.99
        if a.support_gap is not None:passed=False  # Contact snapshots cannot establish the free-fall gate.
        report['cases'].append(dict(meters_per_unit=mpu,status='geometry_dynamics_passed' if passed else 'failed',
            observation_source='live_physx_tensor' if a.tensor_readback else 'usd',
            every_step_health=transient if a.audit_every_step else None,
            authored_density=1050*mpu**3,authored_youngs_modulus=12000*mpu,authored_gravity=9.81/mpu,
            settling_damping=0.,settling_threshold=0.,sleep_threshold=0.,
            acceleration_m_s2=acceleration.tolist(),max_acceleration_error_m_s2=float(error.max()),rows=rows))
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.tensor_readback:observer.close();del observer
        sim.stop();SimulationContext.clear_instance();omni.usd.get_context().close_stage()
        print(f'END_UNIT_CASE meters_per_unit={mpu}',flush=True)
    report['status']='geometry_dynamics_passed_engine_log_required' if all(c['status']=='geometry_dynamics_passed' for c in report['cases']) else 'failed'
    if a.support_gap is not None:report['status']='contact_diagnostic_only'
except Exception as exc:
    report.update(status='failed',error=str(exc));raise
finally:
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    app.close()
