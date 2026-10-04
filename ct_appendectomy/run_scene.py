"""Isaac Sim renderer and optional PhysX FEM appendix cook probe.

Run with an Isaac Sim 5.1 Python environment. A successful render alone does
not constitute a successful physics or appendectomy evaluation.
"""
from pathlib import Path
import argparse
import json
import os
import hashlib

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/isaac-warp'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
p=argparse.ArgumentParser()
p.add_argument('--scene',type=Path,default=ROOT/'out/ct_appendectomy/ct_appendectomy.usda')
p.add_argument('--headless',action='store_true')
p.add_argument('--frames',type=int,default=120)
p.add_argument('--physics',action='store_true')
p.add_argument('--contact-probe',action='store_true')
p.add_argument('--forceps-probe',action='store_true')
p.add_argument('--forceps-rl-smoke',action='store_true')
p.add_argument('--experimental-ct-contact',action='store_true',help='Unvalidated atlas overlap can destabilize tissue')
p.add_argument('--disable-tool-contact',action='store_true',help='Negative control for the contact probe')
p.add_argument('--rl-smoke',action='store_true')
p.add_argument('--retraction-probe',action='store_true',help='Prescribed FEM attachment displacement, not jaw contact')
p.add_argument('--interactive',action='store_true',help='Keep the Isaac window open after the probe')
p.add_argument('--lock-endoscope',action='store_true',help='Restore authored endoscope pose on each interactive frame')
p.add_argument('--explore-cavity',action='store_true',help='Posterior support, supine gravity, Rosota CAD and movable scope')
p.add_argument('--record-scan',action='store_true',help='Record camera exploration PNG frames after settling')
p.add_argument('--disable-support',action='store_true',help='Support collision negative control')
p.add_argument('--require-appendix-visible',action='store_true',help='Require appendix semantic pixels in the captured RTX frame')
p.add_argument('--graphics',choices=('vulkan','d3d12'),default='d3d12')
p.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/isaac_probe')
a=p.parse_args()
if a.explore_cavity:
    a.physics=True
    if a.forceps_probe or a.forceps_rl_smoke or a.contact_probe or a.retraction_probe or a.rl_smoke:
        p.error('Cavity exploration has distinct physics; do not mix legacy probe/checkpoints')
if (a.record_scan or a.disable_support) and not a.explore_cavity:p.error('Requires --explore-cavity')
if a.forceps_rl_smoke: a.forceps_probe=True
if a.rl_smoke: a.retraction_probe=True
if a.retraction_probe or a.contact_probe or a.forceps_probe: a.physics=True
if sum((a.contact_probe,a.retraction_probe,a.forceps_probe))>1: p.error("Choose one physics probe")
if a.disable_tool_contact and not (a.contact_probe or a.forceps_probe): p.error("Negative control requires a contact probe")
if a.experimental_ct_contact and not a.contact_probe: p.error("CT contact requires --contact-probe")
if not a.scene.is_file(): p.error('Build the CT scene first with build_scene.py')
if a.frames<1: p.error('--frames must be positive')
if a.interactive and a.headless: p.error('--interactive requires a visible window')
if a.interactive and (a.rl_smoke or a.forceps_rl_smoke): p.error('RL smoke checks are batch validations; use a physics probe for interactive viewing')
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
    report['source_scene_sha256']=hashlib.sha256(a.scene.read_bytes()).hexdigest()
    simulation=None
    retraction=None
    contact=None
    forceps=None
    cavity_tool=None
    scope=None
    gravity_ready=False
    if a.explore_cavity:
        manifest=json.loads((a.scene.parent/'scene_manifest.json').read_text())
        cavity=manifest['cavity_exploration']
        support=stage.GetPrimAtPath('/World/PosteriorSupport/render')
        if not support:raise RuntimeError('Run prepare_cavity.py first')
        UsdPhysics.CollisionAPI(support).CreateCollisionEnabledAttr(not a.disable_support)
        api=PhysxSchema.PhysxCollisionAPI.Apply(support)
        api.CreateContactOffsetAttr(.0003);api.CreateRestOffsetAttr(0.)
    def force_gpu_zero_gravity():
        # SimulationContext can create its own /physicsScene in addition to the authored scene.
        for item in stage.Traverse():
            if item.IsA(UsdPhysics.Scene):
                scene_physics=UsdPhysics.Scene(item)
                scene_physics.CreateGravityMagnitudeAttr(9.81 if a.explore_cavity and gravity_ready else 0.)
                if a.explore_cavity:scene_physics.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0))
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
        if a.explore_cavity:
            prim.CreateAttribute('physxDeformableBody:solverPositionIterationCount',Sdf.ValueTypeNames.UInt).Set(64)
            prim.CreateAttribute('physxDeformableBody:linearDamping',Sdf.ValueTypeNames.Float).Set(.1)
        # Explicit surgical-scale margins: auto-selected offsets are unsuitable
        # for millimetre jaws and can push tissue before physical contact.
        tissue_collision=PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(body+'/coll'))
        tissue_collision.CreateContactOffsetAttr(.0005)
        tissue_collision.CreateRestOffsetAttr(0.)
        report['tissue_collision_offsets_m']={'contact':.0005,'rest':0.}
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
        simulation=SimulationContext(stage_units_in_meters=1.,physics_dt=1/(240 if a.explore_cavity else 120),rendering_dt=1/60)
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
        report['gravity_m_s2']=9.81 if a.explore_cavity else 0.
        report['material_identified']=False
        if a.retraction_probe or a.contact_probe or a.forceps_probe or a.explore_cavity:
            from attachments import author_retraction
            manifest=json.loads((a.scene.parent/'scene_manifest.json').read_text())
            simulation.stop()
            if a.explore_cavity:
                import rosota_forceps
                cavity_tool=rosota_forceps.author(stage,body+'/sim',initial,manifest['ct_anchor_m'],a.scene.parent/'rosota_distal.usda')
            elif a.forceps_probe:
                from forceps import author_forceps,move_forceps,probe_command
                forceps=author_forceps(stage,body+'/sim',initial,manifest['ct_anchor_m'],a.disable_tool_contact)
            elif a.contact_probe:
                from contact import author_contact
                contact=author_contact(stage,body+'/sim',initial,manifest['ct_anchor_m'],a.disable_tool_contact,a.experimental_ct_contact)
            else:
                retraction=author_retraction(stage,body+'/sim',initial,manifest['ct_anchor_m'])
            cooked=a.out/'attached.usda'
            stage.GetRootLayer().Export(str(cooked))
            if forceps:
                config={'stage_sha256':hashlib.sha256(cooked.read_bytes()).hexdigest(),
                        'forceps':{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in forceps.items()}}
                (a.out/'forceps_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            SimulationContext.clear_instance()
            omni.usd.get_context().open_stage(str(cooked.resolve()))
            for _ in range(30): app.update()
            stage=omni.usd.get_context().get_stage()
            simulation=SimulationContext(stage_units_in_meters=1.,physics_dt=1/(240 if a.explore_cavity else 120),rendering_dt=1/60)
            physics=UsdPhysics.Scene(stage.GetPrimAtPath('/World/physicsScene'))
            physics.CreateGravityMagnitudeAttr(0.)
            PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim()).CreateEnableGPUDynamicsAttr(True)
            gravity_ready=True
            force_gpu_zero_gravity()
            simulation.initialize_physics()
            sim_mesh=stage.GetPrimAtPath(body+'/sim')
            if retraction: retraction['translation']=UsdGeom.Xformable(stage.GetPrimAtPath('/World/RetractionHandle')).GetOrderedXformOps()[0]
            if contact: contact['translation']=UsdGeom.Xformable(stage.GetPrimAtPath('/World/BluntProbe')).GetOrderedXformOps()[0]
        simulation.play()
    if a.explore_cavity:
        from endoscope_control import EndoscopeControl
        scope=EndoscopeControl(stage,cavity['camera_travel_radius_m'])
    from isaacsim.core.utils.semantics import add_labels
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and path.startswith('/World/CT/'):
            add_labels(prim,[prim.GetName()])
        elif prim.IsA(UsdGeom.Mesh) and path.startswith('/World/Atlas/') and prim.GetName()=='render':
            add_labels(prim,[prim.GetParent().GetName()])
        elif path=='/World/BodyEnvelope/render':
            add_labels(prim,['body_envelope'])
    product=rep.create.render_product('/World/Endoscope',(1280,720))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    semantic=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False})
    semantic.attach([product])
    depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    depth.attach([product])
    for frame in range(a.frames):
        if forceps:
            closure,lift=probe_command((frame+1)/a.frames)
            move_forceps(stage,forceps,closure,lift)
        if contact:
            contact["translation"].Set(Gf.Vec3d(*(contact["origin"]+np.array([0,-.012*(frame+1)/a.frames,0]))))
        if retraction:
            lift=.01*(frame+1)/a.frames
            retraction['translation'].Set(Gf.Vec3d(*(retraction['origin']+np.array([0,0,lift]))))
        if simulation: simulation.step(render=True)
        else: app.update()
        if forceps and (frame==0 or (frame+1)%max(1,a.frames//10)==0):
            current=np.asarray(sim_mesh.GetAttribute('points').Get(),dtype=float)
            report.setdefault('forceps_trajectory',[]).append({'frame':frame+1,
                'closure':float(closure),'command_lift_m':float(lift),
                'max_displacement_m':float(np.linalg.norm(current-initial,axis=1).max())})
        if forceps and frame+1==int(.45*a.frames):
            current=np.asarray(sim_mesh.GetAttribute('points').Get(),dtype=float)
            forceps['closed_centroid']=current[forceps['grip']].mean(axis=0)
        if forceps and frame+1==int(.75*a.frames):
            current=np.asarray(sim_mesh.GetAttribute('points').Get(),dtype=float)
            lift_delta=current[forceps['grip']].mean(axis=0)-forceps['closed_centroid']
            report['forceps_lift_checkpoint_m']=float(lift_delta[2])
            report['forceps_tracking_error_m']=float(np.linalg.norm(lift_delta-np.array([0,0,.008])))
    # A newly attached render product may not have a host buffer on its first step.
    for capture_attempt in range(4):
        rep.orchestrator.step(rt_subframes=8,delta_time=0.0)
        pixels=np.asarray(rgb.get_data())
        semantic_data=semantic.get_data()
        depth_data=np.asarray(depth.get_data())
        if pixels.ndim==3 and pixels.shape[:2]==(720,1280) and isinstance(semantic_data,dict) and np.asarray(semantic_data.get('data',[])).shape==(720,1280) and depth_data.shape==(720,1280):
            break
    if pixels.ndim!=3 or pixels.shape[:2]!=(720,1280): raise RuntimeError('No valid RTX camera image')
    from PIL import Image
    Image.fromarray(pixels[:,:,:3].astype(np.uint8)).save(a.out/'endoscope.png')
    labels=np.asarray(semantic_data.get('data',[]))
    if labels.shape!=(720,1280) or depth_data.shape!=(720,1280):
        raise RuntimeError('Missing synchronized semantic/depth frame')
    label_info=semantic_data['info']['idToLabels']
    fractions={}
    for identity,description in label_info.items():
        name=description.get('class',str(identity))
        fractions[name]=fractions.get(name,0.)+float(np.mean(labels==int(identity)))
    np.savez_compressed(a.out/'camera_ground_truth.npz',semantic=labels,depth_m=depth_data)
    (a.out/'semantic_labels.json').write_text(json.dumps(label_info,indent=2),encoding='utf-8')
    report['camera_ground_truth']={'resolution':[1280,720],'semantic_frame_fraction':fractions,
        'depth_finite_fraction':float(np.isfinite(depth_data).mean()),
        'appendix_visibility_gate_passed':fractions.get('appendix',0.)>=.005,
        'threshold_frame_fraction':.005,'clinical_view_validated':False}
    if a.require_appendix_visible and fractions.get('appendix',0.)<.005:
        raise RuntimeError('RTX appendix visibility below 0.5 percent of frame')
    report['image_std']=float(pixels[:,:,:3].std())
    if report['image_std']<1: raise RuntimeError('Camera image is empty or constant')
    if simulation:
        final=np.asarray(sim_mesh.GetAttribute('points').Get(),dtype=float)
        report['finite_nodes']=bool(np.isfinite(final).all())
        if not report['finite_nodes']: raise RuntimeError('Non-finite appendix simulation state')
        report['max_node_displacement_m']=float(np.linalg.norm(final-initial,axis=1).max())
        from tissue_metrics import tetra_health
        tet_indices=np.asarray(sim_mesh.GetAttribute('tetVertexIndices').Get(),dtype=int)
        report['tissue_health']=tetra_health(initial,final,tet_indices)
        np.savez_compressed(a.out/'tissue_state.npz',rest=initial,current=final,tetrahedra=tet_indices)
        report['physics_validation']='cook/finite-state only; zero gravity, no contact or grasp test'
        if a.explore_cavity:
            floor=cavity['posterior_support_y_m']
            penetration=float(max(0.,floor-final[:,1].min()))
            collision_points=np.asarray(stage.GetPrimAtPath(body+'/coll').GetAttribute('points').Get(),dtype=float)
            surface_penetration=float(max(0.,floor-collision_points[:,1].min()))
            near=int(np.count_nonzero(np.abs(final[:,1]-floor)<.0015))
            report['mode']='supine cavity exploration with Rosota CAD'
            report['physics_validation']='gravity settling on engineering posterior support; CT organs static'
            report['cavity_support']={'enabled':not a.disable_support,'floor_y_m':floor,
                'simulation_cage_penetration_m':penetration,'collision_surface_penetration_m':surface_penetration,'nodes_within_1_5mm':near,
                'mean_posterior_displacement_m':float(np.mean(initial[:,1]-final[:,1])),
                'pin_drift_m':float(np.linalg.norm(final[cavity_tool['pin']]-initial[cavity_tool['pin']],axis=1).max())}
            if report['tissue_health']['inverted_tetrahedra']:raise RuntimeError('Gravity settling inverted tissue')
            if not a.disable_support and (surface_penetration>.001 or near==0):raise RuntimeError('Posterior support contact gate failed')
            if report['cavity_support']['pin_drift_m']>.001:raise RuntimeError('Appendix base drift')
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
    if forceps:
        report['mode']='hinged forceps contact probe'
        report['physics_validation']='close/hold/lift/open with two collision jaws; no grip attachment'
        report['tool_collision_enabled']=not a.disable_tool_contact
        report['pin_drift_m']=float(np.linalg.norm(final[forceps['pin']]-initial[forceps['pin']],axis=1).max())
        report['forceps_grasp_lift_passed']=report.get('forceps_lift_checkpoint_m',0)>=.004 and report.get('forceps_tracking_error_m',1)<.004
        report['forceps_geometry']={'jaw_length_m':forceps['length'],'closed_angle_rad':forceps['closed_angle'],
                                   'local_tissue_diameter_m':forceps['diameter_m'],'clinical_cad':False}
        if report['pin_drift_m']>.001 or report['max_node_displacement_m']>.05:
            raise RuntimeError('Forceps contact stability gate failed')
        if not a.disable_tool_contact and not report['forceps_grasp_lift_passed']:
            raise RuntimeError('Jaws did not retain tissue during the 8mm lift')
        if report['tissue_health']['inverted_tetrahedra']:
            raise RuntimeError('Forceps contact inverted tissue tetrahedra')
    if a.forceps_rl_smoke:
        from forceps_env import AppendixForcepsEnv
        simulation.stop()
        env=AppendixForcepsEnv(app,a.out/'attached.usda',a.out/'forceps_config.json',render_mode='rgb_array')
        obs,info=env.reset(seed=42)
        baseline=obs.copy()
        image=env.render()
        render_error=float(np.max(np.abs(env._observe()-baseline)))
        report['forceps_reset_render_state_error']=render_error
        if render_error>1e-5:raise RuntimeError('Forceps RGB render advanced physics')
        Image.fromarray(image).save(a.out/'rl_endoscope.png')
        try:env.step([float('nan'),0])
        except ValueError:pass
        else:raise RuntimeError('Invalid forceps action accepted')
        for count in range(220):
            action=[1./1.8,0.] if count<72 else ([0.,1./1.8] if 108<=count<181 else [0.,0.])
            obs,reward,terminated,truncated,info=env.step(action)
            if not env.observation_space.contains(obs):raise RuntimeError('Invalid forceps observation')
            if terminated or truncated:break
        report['forceps_rl_validation']={'status':'failed' if not info['is_success'] else 'baseline_passed',
            'steps':count+1,'last_info':info,'last_observation':obs.tolist(),
            'terminated':terminated,'truncated':truncated}
        Image.fromarray(env.render()).save(a.out/'rl_final.png')
        if not info['is_success']:raise RuntimeError('Forceps RL scripted baseline failed')
        success_info=dict(info)
        try:env.step([0,0])
        except RuntimeError:pass
        else:raise RuntimeError('Finished forceps episode accepted action')
        obs,info=env.reset(seed=42)
        reset_error=float(np.max(np.abs(obs-baseline)))
        report['forceps_rl_validation']['reset_max_error']=reset_error
        report['forceps_rl_validation']['initial_observation']=baseline.tolist()
        report['forceps_rl_validation']['reset_observation']=obs.tolist()
        if reset_error>1e-5:raise RuntimeError('Forceps reset failed')
        env.max_steps=2
        env.step([0,0])
        _,_,term,trunc,_=env.step([0,0])
        if term or not trunc:raise RuntimeError('Forceps time limit failed')
        report['forceps_rl_validation']={'status':'passed','steps_to_success':count+1,
            'reset_max_error':reset_error,'render_state_error':render_error,
            'invalid_action_rejected':True,'terminal_guard':True,'time_limit_truncates':True,
            'success_info':success_info,'trained_policy':False}
        simulation=env.sim
        env.close()
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
    if a.record_scan:
        from PIL import Image
        scan_dir=a.out/'scan_frames';scan_dir.mkdir(exist_ok=True)
        simulation.pause()
        poses=[]
        for i in range(90):
            pose=scope.scan(i/89)
            rep.orchestrator.step(rt_subframes=4,delta_time=0.)
            image=np.asarray(rgb.get_data())[:,:,:3]
            if image.shape!=(720,1280,3):raise RuntimeError('Missing scan image')
            Image.fromarray(image).save(scan_dir/f'{i:04d}.png')
            poses.append(pose.tolist())
        report['endoscope_scan']={'frames':90,'poses':poses,'travel_radius_m':scope.travel,
                                 'physics_paused_during_scan':True}
        scope.apply(0,0,0,0,0,0)
    report['frames']=a.frames
    report['status']='passed'
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if a.interactive:
        stage=omni.usd.get_context().get_stage()
        camera_op=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0]
        camera_pose=camera_op.Get()
        if scope:scope.create_ui()
    forceps_controls=None
    if a.interactive and (forceps or cavity_tool):
        import omni.ui as ui
        forceps_window=ui.Window('Rosota laparoscopic gripper' if cavity_tool else 'Forceps contact fixture',width=340,height=180,position_x=390,position_y=80)
        with forceps_window.frame:
            with ui.VStack(spacing=8):
                ui.Label('CAD jaws / convex contact' if cavity_tool else 'Jaw contact and lift; cutting not integrated')
                ui.Label('Jaw closure')
                closure_slider=ui.FloatSlider(min=0.,max=1.)
                closure_slider.model.set_value(0.)
                ui.Label('Anterior withdrawal / posterior advance (mm)' if cavity_tool else 'Lift (mm)')
                lift_slider=ui.FloatSlider(min=-35. if cavity_tool else 0.,max=16.)
                lift_slider.model.set_value(0. if cavity_tool else 8.)
        forceps_controls=np.array([0.,0. if cavity_tool else .008])
    if a.interactive and simulation: simulation.play()
    while a.interactive and app.is_running():
        if scope:scope.update_ui()
        if forceps_controls is not None:
            target=np.array([closure_slider.model.get_value_as_float(),lift_slider.model.get_value_as_float()*.001])
            forceps_controls+=np.clip(target-forceps_controls,[-.01,-.0002],[.01,.0002])
            if cavity_tool:rosota_forceps.move(stage,cavity_tool,*forceps_controls)
            else:move_forceps(stage,forceps,*forceps_controls)
        if a.lock_endoscope and not scope:
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
