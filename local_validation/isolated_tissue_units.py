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
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if any(v<=0 for v in a.units):p.error('Units must be positive')
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
    scope='Isolated free fall with zero damping, no attachments, robot, contact or rendering. Not loaded tissue validation.',
    material_si=dict(density_kg_m3=1050,youngs_modulus_pa=12000,poissons_ratio=.45),
    gates=dict(max_translation_removed_rms_m=1e-5,max_acceleration_error_m_s2=.1,minimum_volume_ratio=.99))
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
        prim.CreateAttribute('physxDeformableBody:solverPositionIterationCount',Sdf.ValueTypeNames.UInt).Set(64)
        prim.CreateAttribute('physxDeformableBody:linearDamping',Sdf.ValueTypeNames.Float).Set(0.)
        prim.CreateAttribute('physxDeformableBody:selfCollision',Sdf.ValueTypeNames.Bool).Set(False)
        du.add_deformable_material(stage,'/World/Material',density=1050*mpu**3,youngs_modulus=12000*mpu,poissons_ratio=.45)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/Material'),UsdShade.Tokens.weakerThanDescendants,'physics')
        sim=SimulationContext(stage_units_in_meters=mpu,physics_dt=1/480,rendering_dt=1/30)
        scene=UsdPhysics.Scene.Get(stage,sim.get_physics_context().prim_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr(9.81/mpu)
        ps=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());ps.CreateEnableGPUDynamicsAttr(True);ps.CreateBroadphaseTypeAttr('GPU')
        stage.GetRootLayer().Export(str(a.out/f'scene_{mpu}.usda'))
        sim.initialize_physics();sim.play()
        if not np.array_equal(np.asarray(mesh.GetTetVertexIndicesAttr().Get(),int),tets):
            raise RuntimeError('Simulation changed mesh indexing; do not compare mismatched nodes')
        baseline=np.asarray(mesh.GetPointsAttr().Get(),float)*mpu
        if a.tensor_readback:
            import omni.physics.tensors as tensors
            from node_correspondence import node_permutation
            tensor_sim=tensors.create_simulation_view('torch')
            body=tensor_sim.create_volume_deformable_body_view('/World/Tissue')
            tensor_rest=body.get_rest_nodal_positions().cpu().numpy()[0]
            tensor_tets=body.get_simulation_element_indices().cpu().numpy()[0]
            permutation=node_permutation(np.asarray(prim.GetAttribute('omniphysics:restShapePoints').Get()),tets,tensor_rest,tensor_tets)
        rows=[];centers=[]
        for tick in range(181):
            if tick%60==0:
                points=np.asarray(mesh.GetPointsAttr().Get(),float)*mpu
                displacement=points-baseline;translation=displacement.mean(0)
                rms=float(np.sqrt(np.mean(np.sum((displacement-translation)**2,axis=1))))
                rows.append(dict(tick=tick,health=tetra_health(rest,points,tets),translation_removed_rms_m=rms))
                if a.tensor_readback:
                    tensor_points=body.get_simulation_nodal_positions().cpu().numpy()[0][permutation].astype(float)*mpu
                    discrepancy=np.linalg.norm(tensor_points-points,axis=1)
                    rows[-1].update(tensor_health=tetra_health(rest,tensor_points,tets),
                        tensor_usd_max_difference_m=float(discrepancy.max()),
                        tensor_usd_different_nodes=int(np.sum(discrepancy>1e-7)))
                    np.savez_compressed(a.out/f'tensor_{mpu}_{tick}.npz',points=tensor_points,rest=rest,tets=tets)
                centers.append(points.mean(0))
                np.savez_compressed(a.out/f'state_{mpu}_{tick}.npz',points=points,rest=rest,tets=tets)
            if tick<180:sim.step(render=False)
        acceleration=np.diff(np.asarray(centers),n=2,axis=0)/(.125**2)
        error=np.linalg.norm(acceleration-np.array([0,-9.81,0]),axis=1)
        passed=all(r['health']['inverted_tetrahedra']==0 and r['health']['minimum_signed_volume_ratio']>=.99 and r['translation_removed_rms_m']<=1e-5 for r in rows) and bool(np.max(error)<=.1)
        report['cases'].append(dict(meters_per_unit=mpu,status='geometry_dynamics_passed' if passed else 'failed',
            authored_density=1050*mpu**3,authored_youngs_modulus=12000*mpu,authored_gravity=9.81/mpu,
            acceleration_m_s2=acceleration.tolist(),max_acceleration_error_m_s2=float(error.max()),rows=rows))
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.tensor_readback:del body,tensor_sim
        sim.stop();SimulationContext.clear_instance();omni.usd.get_context().close_stage()
        print(f'END_UNIT_CASE meters_per_unit={mpu}',flush=True)
    report['status']='geometry_dynamics_passed_engine_log_required' if all(c['status']=='geometry_dynamics_passed' for c in report['cases']) else 'failed'
except Exception as exc:
    report.update(status='failed',error=str(exc));raise
finally:
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    app.close()
