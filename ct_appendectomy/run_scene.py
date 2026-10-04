"""Isaac Sim renderer and optional PhysX FEM appendix cook probe.

Run with an Isaac Sim 5.1 Python environment. A successful render alone does
not constitute a successful physics or appendectomy evaluation.
"""
from pathlib import Path
import argparse
import json
import os

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/isaac-warp'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
p=argparse.ArgumentParser()
p.add_argument('--scene',type=Path,default=ROOT/'out/ct_appendectomy/ct_appendectomy.usda')
p.add_argument('--headless',action='store_true')
p.add_argument('--frames',type=int,default=120)
p.add_argument('--physics',action='store_true')
p.add_argument('--contact-probe',action='store_true')
p.add_argument('--experimental-ct-contact',action='store_true',help='Unvalidated atlas overlap can destabilize tissue')
p.add_argument('--disable-tool-contact',action='store_true',help='Negative control for the contact probe')
p.add_argument('--rl-smoke',action='store_true')
p.add_argument('--retraction-probe',action='store_true',help='Prescribed FEM attachment displacement, not jaw contact')
p.add_argument('--interactive',action='store_true',help='Keep the Isaac window open after the probe')
p.add_argument('--lock-endoscope',action='store_true',help='Restore authored endoscope pose on each interactive frame')
p.add_argument('--graphics',choices=('vulkan','d3d12'),default='d3d12')
p.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/isaac_probe')
a=p.parse_args()
if a.rl_smoke: a.retraction_probe=True
if a.retraction_probe or a.contact_probe: a.physics=True
if a.contact_probe and a.retraction_probe: p.error("Choose contact or attachment retraction, not both")
if a.disable_tool_contact and not a.contact_probe: p.error("Negative control requires --contact-probe")
if a.experimental_ct_contact and not a.contact_probe: p.error("CT contact requires --contact-probe")
if not a.scene.is_file(): p.error('Build the CT scene first with build_scene.py')
if a.frames<1: p.error('--frames must be positive')
if a.interactive and a.headless: p.error('--interactive requires a visible window')
if a.interactive and a.rl_smoke: p.error('--rl-smoke is a batch validation; use --retraction-probe for interactive viewing')
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'report.json').write_text(json.dumps({'status':'starting','appendectomy_completed':False}))
(ROOT/'.cache/optix').mkdir(parents=True,exist_ok=True)

from isaacsim import SimulationApp
runtime=ROOT/'.cache/kit'
tokens={name:runtime/name for name in ('shared_documents','app_documents','omni_documents','omni_config',
    'omni_logs','cache','omni_cache','omni_global_cache')}
for path in tokens.values(): path.mkdir(parents=True,exist_ok=True)
app=SimulationApp({'headless':a.headless,'width':1280,'height':720,
    'renderer':'RaytracedLighting','sync_loads':True,'hide_ui':a.headless,
    'extra_args':[f'--/app/vulkan={str(a.graphics=="vulkan").lower()}',
                  f'--/app/d3d12={str(a.graphics=="d3d12").lower()}',
                  '--/telemetry/enableNVDF=false','--/telemetry/enableSentry=false',
                  '--/telemetry/useOpenEndpoint=false','--/privacy/extraDiagnosticDataOptIn=false',
                  '--/crashreporter/enabled=false',f'--/app/window/enabled={str(not a.headless).lower()}']+
                  [f'--/app/tokens/{name}={path.as_posix()}' for name,path in tokens.items()]})
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Sdf,Gf
    from isaacsim.core.api import SimulationContext
    omni.usd.get_context().open_stage(str(a.scene.resolve()))
    for _ in range(30): app.update()
    stage=omni.usd.get_context().get_stage()
    if not a.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport: viewport.set_active_camera('/World/Endoscope')
    report={'engine':'Isaac Sim 5.1 / PhysX','mode':'FEM retraction + Gymnasium' if a.rl_smoke else ('FEM retraction' if a.retraction_probe else ('FEM cook probe' if a.physics else 'render only')),
        'appendectomy_completed':False,'appendix_patient_specific':False,'registration_validated':False}
    simulation=None
    retraction=None
    contact=None
    def force_gpu_zero_gravity():
        # SimulationContext can create its own /physicsScene in addition to the authored scene.
        for item in stage.Traverse():
            if item.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(item).CreateGravityMagnitudeAttr(0.)
                api=PhysxSchema.PhysxSceneAPI.Apply(item)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr('GPU')
    if a.physics:
        from omni.physx.scripts import deformableUtils as du
        body='/World/Atlas/appendix'
        if not hasattr(du,'create_auto_volume_deformable_hierarchy'):
            raise RuntimeError('Required volume deformable API unavailable; no rigid fallback')
        ok=du.create_auto_volume_deformable_hierarchy(stage,body,body+'/sim',body+'/coll',body+'/render',
            simulation_hex_mesh_enabled=True,cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True)
        if not ok: raise RuntimeError('PhysX deformable authoring failed')
        prim=stage.GetPrimAtPath(body)
        prim.CreateAttribute('physxDeformableBody:resolution',Sdf.ValueTypeNames.UInt).Set(32)
        du.add_deformable_material(stage,'/World/TissueMaterial',density=1050.,youngs_modulus=12000.,
            poissons_ratio=.45,dynamic_friction=.5,static_friction=.5)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/TissueMaterial'),
            UsdShade.Tokens.weakerThanDescendants,'physics')
        UsdGeom.Imageable(stage.GetPrimAtPath(body+'/coll')).CreateVisibilityAttr().Set('invisible')
        physics=UsdPhysics.Scene.Define(stage,'/World/physicsScene')
        physics.CreateGravityMagnitudeAttr(0.)
        gpu=PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        gpu.CreateEnableGPUDynamicsAttr(True)
        gpu.CreateBroadphaseTypeAttr('GPU')
        simulation=SimulationContext(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/60)
        # SimulationContext may re-author gravity; keep this explicitly a zero-g cook test.
        physics.CreateGravityMagnitudeAttr(0.)
        gpu.CreateEnableGPUDynamicsAttr(True)
        gpu.CreateBroadphaseTypeAttr('GPU')
        force_gpu_zero_gravity()
        simulation.initialize_physics()
        for _ in range(240): app.update()
        sim_mesh=stage.GetPrimAtPath(body+'/sim')
        points=sim_mesh.GetAttribute('points').Get()
        if points is None or len(points)==0: raise RuntimeError('PhysX did not cook appendix simulation nodes')
        initial=np.asarray(points,dtype=float)
        if not np.isfinite(initial).all(): raise RuntimeError('Invalid cooked nodes')
        report['cooked_nodes']=len(initial)
        report['gravity_m_s2']=0.
        report['material_identified']=False
        if a.retraction_probe or a.contact_probe:
            from attachments import author_retraction
            manifest=json.loads((a.scene.parent/'scene_manifest.json').read_text())
            simulation.stop()
            if a.contact_probe:
                from contact import author_contact
                contact=author_contact(stage,body+'/sim',initial,manifest['ct_anchor_m'],a.disable_tool_contact,a.experimental_ct_contact)
            else:
                retraction=author_retraction(stage,body+'/sim',initial,manifest['ct_anchor_m'])
            cooked=a.out/'attached.usda'
            stage.GetRootLayer().Export(str(cooked))
            SimulationContext.clear_instance()
            omni.usd.get_context().open_stage(str(cooked.resolve()))
            for _ in range(30): app.update()
            stage=omni.usd.get_context().get_stage()
            simulation=SimulationContext(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/60)
            physics=UsdPhysics.Scene(stage.GetPrimAtPath('/World/physicsScene'))
            physics.CreateGravityMagnitudeAttr(0.)
            PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim()).CreateEnableGPUDynamicsAttr(True)
            force_gpu_zero_gravity()
            simulation.initialize_physics()
            sim_mesh=stage.GetPrimAtPath(body+'/sim')
            if retraction: retraction['translation']=UsdGeom.Xformable(stage.GetPrimAtPath('/World/RetractionHandle')).GetOrderedXformOps()[0]
            if contact: contact['translation']=UsdGeom.Xformable(stage.GetPrimAtPath('/World/BluntProbe')).GetOrderedXformOps()[0]
        simulation.play()
    product=rep.create.render_product('/World/Endoscope',(1280,720))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    for frame in range(a.frames):
        if contact:
            contact["translation"].Set(Gf.Vec3d(*(contact["origin"]+np.array([0,-.012*(frame+1)/a.frames,0]))))
        if retraction:
            lift=.01*(frame+1)/a.frames
            retraction['translation'].Set(Gf.Vec3d(*(retraction['origin']+np.array([0,0,lift]))))
        if simulation: simulation.step(render=True)
        else: app.update()
    rep.orchestrator.step(rt_subframes=8)
    pixels=np.asarray(rgb.get_data())
    if pixels.ndim!=3 or pixels.shape[:2]!=(720,1280): raise RuntimeError('No valid RTX camera image')
    from PIL import Image
    Image.fromarray(pixels[:,:,:3].astype(np.uint8)).save(a.out/'endoscope.png')
    report['image_std']=float(pixels[:,:,:3].std())
    if report['image_std']<1: raise RuntimeError('Camera image is empty or constant')
    if simulation:
        final=np.asarray(sim_mesh.GetAttribute('points').Get(),dtype=float)
        report['finite_nodes']=bool(np.isfinite(final).all())
        if not report['finite_nodes']: raise RuntimeError('Non-finite appendix simulation state')
        report['max_node_displacement_m']=float(np.linalg.norm(final-initial,axis=1).max())
        report['physics_validation']='cook/finite-state only; zero gravity, no contact or grasp test'
        if retraction:
            report['pin_drift_m']=float(np.linalg.norm(final[retraction['pin']]-initial[retraction['pin']],axis=1).max())
            report['grip_lift_m']=float((final[retraction['grip'],2]-initial[retraction['grip'],2]).mean())
            report['physics_validation']='prescribed 10mm attachment retraction; not contact-based grasp'
            if report['pin_drift_m']>.001 or report['grip_lift_m']<.005:
                raise RuntimeError('Retraction attachment did not hold base and lift tissue as commanded')
    if contact:
        report['mode']='blunt-tip contact' if not a.disable_tool_contact else 'contact negative control'
        report['physics_validation']='base-pinned FEM, 12mm kinematic sphere motion; no grasp'
        report['tool_collision_enabled']=not a.disable_tool_contact
        report['ct_static_colliders_authored']=len(contact['colliders'])
        report['ct_contact_enabled']=a.experimental_ct_contact
        report['pin_drift_m']=float(np.linalg.norm(final[contact['pin']]-initial[contact['pin']],axis=1).max())
        report['mean_node_displacement_m']=float(np.linalg.norm(final-initial,axis=1).mean())
        if report['pin_drift_m']>.001: raise RuntimeError('Contact probe base drift exceeds 1mm')
        if report['max_node_displacement_m']>.05: raise RuntimeError('Contact probe unstable: displacement exceeds 50mm')
        if not a.disable_tool_contact and report['max_node_displacement_m']<.001:
            raise RuntimeError('No measurable contact-induced tissue displacement')
    if a.rl_smoke:
        from retraction_env import AppendixRetractionEnv
        simulation.stop()
        env=AppendixRetractionEnv(app,a.out/'attached.usda',render_mode='rgb_array')
        obs,info=env.reset(seed=42)
        baseline=obs.copy()
        image=env.render()
        if image.std()<1: raise RuntimeError('RL endoscope image empty')
        Image.fromarray(image).save(a.out/'rl_endoscope.png')
        render_state_error=float(np.max(np.abs(env._observe()-baseline)))
        if render_state_error>1e-5: raise RuntimeError('Rendering advanced physical state')
        assert env.observation_space.contains(obs)
        try:
            env.step([float('nan'),0,0])
        except ValueError:
            pass
        else:
            raise RuntimeError('Invalid action accepted')
        total_reward=0.
        for count in range(160):
            obs,reward,terminated,truncated,info=env.step([0,0,.1])
            assert env.observation_space.contains(obs)
            total_reward+=reward
            if terminated or truncated: break
        if not info['is_success']: raise RuntimeError('RL retraction did not reach target')
        success_info=dict(info)
        try:
            env.step([0,0,0])
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Finished episode accepted another action')
        obs,info=env.reset(seed=42)
        reset_error=float(np.max(np.abs(obs-baseline)))
        if reset_error>1e-5: raise RuntimeError('Reset failed to restore initial state')
        env.max_steps=2
        env.step([0,0,0])
        _,_,term,trunc,_=env.step([0,0,0])
        if term or not trunc: raise RuntimeError('Time-limit semantics failed')
        report['rl_validation']={'status':'passed','steps_to_success':count+1,
            'return':total_reward,'reset_max_error':reset_error,
            'rgb_shape':list(image.shape),'render_state_error':render_state_error,
            'invalid_action_rejected':True,'terminal_guard':True,'time_limit_truncates':True,
            'success_info':success_info}
        simulation=env.sim
        env.close()
    # Stage reload for FEM attachments can reset the viewport camera selection.
    if not a.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport: viewport.set_active_camera('/World/Endoscope')
    report['frames']=a.frames
    report['status']='passed'
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    camera_op=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0]
    camera_pose=camera_op.Get()
    if a.interactive and simulation: simulation.play()
    while a.interactive and app.is_running():
        if a.lock_endoscope:
            camera_op.Set(camera_pose)
            if viewport: viewport.set_active_camera('/World/Endoscope')
        if simulation: simulation.step(render=True)
        else: app.update()
except BaseException as exc:
    import traceback
    failure=dict(locals().get('report',{}))
    failure.update(status='failed',error_type=type(exc).__name__,error=str(exc),appendectomy_completed=False)
    (a.out/'report.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
    traceback.print_exc()
    raise
finally:
    app.close()
