"""Record and verify the real CT articulation/RCM path, without learned claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fr3-urdf',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=ROOT/'out/ct_appendectomy/multitask_assets')
    p.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/rosota_articulation')
    p.add_argument('--steps',type=int,default=90)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--contact-probe',action='store_true',help='Slow axial approach, torque-limited close, reopen; not RL')
    p.add_argument('--contact-travel-mm',type=float,default=18.,help='Use 0 for matched free-space jaw-cycle control')
    p.add_argument('--surface-approach',action='store_true',help='Aim at nearest live tissue nodes instead of axial motion')
    p.add_argument('--close-samples',type=int,default=150,help='30 Hz closure samples; allow measured jaws to reach contact')
    p.add_argument('--approach-samples',type=int,default=150,help='0 skips approach for a free-space jaw-cycle control')
    p.add_argument('--grasp-probe','--baseline-task',dest='baseline_task',action='store_true',
        help='Experimental CT assisted grasp/pull/release; not full Rosota baseline parity')
    p.add_argument('--grasp-approach-samples',type=int,default=240)
    p.add_argument('--attachment-probe',action='store_true',help='Isolated activation diagnostic, not a robot grasp')
    p.add_argument('--solver-grasp',action='store_true',help='Experimental dynamic-link attachment; remains assisted adhesion')
    p.add_argument('--coupling-probe',action='store_true',help='Paired enabled/disabled constraint load-path test, not a grasp')
    p.add_argument('--readback-probe',action='store_true',help='Diagnostic moving anchor checks physics-only observation freshness')
    p.add_argument('--mount-sensor-probe',action='store_true',help='Known 0.1 N external force sensor calibration without tissue attachment')
    p.add_argument('--diagnostic-zero-gravity',action='store_true',help='Only for isolating the mount sensor failure, never a task configuration')
    a=p.parse_args()
    if a.readback_probe and any((a.mount_sensor_probe,a.coupling_probe,a.baseline_task,a.contact_probe,a.attachment_probe,a.solver_grasp)):
        p.error('Readback probe is an isolated diagnostic')
    if a.diagnostic_zero_gravity and not a.mount_sensor_probe:p.error('Zero-gravity diagnostic requires --mount-sensor-probe')
    if a.coupling_probe and (a.baseline_task or a.contact_probe or a.attachment_probe):p.error('Coupling probe is a separate diagnostic')
    if a.mount_sensor_probe and (a.baseline_task or a.contact_probe or a.attachment_probe or a.coupling_probe or a.solver_grasp):p.error('Mount sensor probe is a separate diagnostic')
    if a.solver_grasp and not (a.baseline_task or a.coupling_probe):p.error('--solver-grasp requires --grasp-probe or --coupling-probe')
    if a.steps<1:p.error('--steps must be positive')
    if a.grasp_approach_samples<1:p.error('--grasp-approach-samples must be positive')
    if a.baseline_task and a.contact_probe:p.error('Choose baseline task or contact probe')
    if a.surface_approach and not a.contact_probe:p.error('--surface-approach requires --contact-probe')
    if a.close_samples<1:p.error('--close-samples must be positive')
    if a.approach_samples<0:p.error('--approach-samples must be nonnegative')
    if not 0<=a.contact_travel_mm<=100:p.error('--contact-travel-mm must be finite and in [0,100]')
    a.out.mkdir(parents=True,exist_ok=True)
    frames=a.out/'frames';frames.mkdir(exist_ok=True)
    if any(frames.glob('*.png')):p.error('Use a fresh output folder')
    report={'status':'starting','learned_policy':False,'policy':'geometric two-tip target tracking'}
    report['implementation_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ('rosota_ct_env.py','solver_grasp.py','tissue_grasp.py','coupling_probe.py','settlement.py','mount_sensor_probe.py','node_correspondence.py','readback_probe.py')}
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for n,d in [('WARP_CACHE_PATH','isaac-warp'),('OPTIX_CACHE_PATH','optix')]:os.environ.setdefault(n,str(ROOT/'.cache'/d))
    from isaacsim import SimulationApp
    runtime=ROOT/'.cache/kit'
    tokens={n:runtime/n for n in ('shared_documents','app_documents','omni_documents','omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
    for path in tokens.values():path.mkdir(parents=True,exist_ok=True)
    app=SimulationApp({'headless':a.headless,'renderer':'RaytracedLighting','sync_loads':True,
        'extra_args':['--/app/vulkan=false','--/app/d3d12=true','--/telemetry/enableNVDF=false',
        '--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false','--/privacy/extraDiagnosticDataOptIn=false',
        '--/crashreporter/enabled=false']+[f'--/app/tokens/{k}={v.as_posix()}' for k,v in tokens.items()]})
    env=None
    try:
        import numpy as np
        from PIL import Image,ImageDraw
        from rosota_ct_env import RosotaCTEnv
        env=RosotaCTEnv(app,a.assets,a.fr3_urdf,a.out,max_steps=a.steps,
            grasp_mode='solver_attachment' if a.solver_grasp or a.coupling_probe else ('rosota_attachment' if a.baseline_task else 'none'),
            workspace='appendix_patches' if a.baseline_task or a.coupling_probe else 'legacy',
            settle_policy='adaptive' if a.coupling_probe else 'fixed',
            diagnostic_no_tool_collision=a.mount_sensor_probe or a.readback_probe,diagnostic_zero_gravity=a.diagnostic_zero_gravity)
        obs,info=env.reset(seed=42);report.update(status='running',**info);save()
        if a.readback_probe:
            from readback_probe import run
            report.update(run(env));return
        if a.mount_sensor_probe:
            from mount_sensor_probe import run
            report.update(run(env));return
        if a.coupling_probe:
            from coupling_probe import run
            report.update(run(env));return
        if a.attachment_probe:
            from attachment_probe import run
            report.update(run(env));return
        if a.contact_probe:
            report['policy']=('scripted live-surface approach/close/reopen probe' if a.surface_approach
                else 'scripted free-space jaw-cycle control' if a.approach_samples==0
                else 'scripted slow axial contact/close/reopen probe')
            report['requested_travel_mm']=a.contact_travel_mm
            report['approach_samples']=a.approach_samples;report['close_samples']=a.close_samples
            save()
        n=0;start=obs['tool_pose'][:,:3,3].copy();max_move=np.zeros(2)
        initial_tissue=env.points().copy()
        targets=[]
        for tip in start:
            point=initial_tissue[np.argmin(np.linalg.norm(initial_tissue-tip,axis=1))]
            direction=point-tip
            targets.append(point+.002*direction/max(np.linalg.norm(direction),1e-9))
        report['surface_approach']=a.surface_approach
        if a.baseline_task:report['policy']='scripted Rosota-style attachment-assisted grasp/pull/release'
        mechanics=[];pull_start=None;pull_measure=None;approached={h:False for h in env.specs}
        approach_end=a.grasp_approach_samples
        if a.baseline_task:
            # Gravity settlement changes the target after fixture authoring.
            # Budget the path from measured live tissue, with 7.2 mm margin.
            distance=max(x['distance_m'] for x in info['grasp_geometry'].values())
            approach_end=max(approach_end,int(np.ceil(distance/.00012))+60)
        close_end=approach_end+150;pull_end=close_end+60
        withdraw_start=pull_end+15;withdraw_end=withdraw_start+60;release_end=withdraw_end+90
        def capture():
            nonlocal n
            before=env.points().copy();rgb=env.render()
            if np.max(np.abs(env.points()-before))>1e-7:raise RuntimeError('Rendering moved tissue')
            im=Image.fromarray(rgb);d=ImageDraw.Draw(im);d.rectangle((0,0,640,38),fill=(15,20,25))
            d.text((8,4),'ROSOTA CONTROLLER / PHYSX FR3 JOINT FEEDBACK',fill='white')
            d.text((8,21),f'Engineering ports | Scope RCM | step {env.steps}',fill='white')
            if a.contact_probe:
                d.rectangle((0,320,640,360),fill=(15,20,25))
                for i,h in enumerate(env.specs):
                    g=env.guards[h]
                    d.text((8,323+17*i),f'{h}: mount {g.last["residual_force_n"]:.3f} N | arm stop={g.latched} | jaw hold={g.pinch_latched}',fill='white')
            if a.baseline_task:
                d.rectangle((0,320,640,360),fill=(15,20,25))
                d.text((8,323),f'ROSOTA ATTACHMENT ASSIST | held: {list(env.grasp.held)}',fill='white')
                d.text((8,340),'Measured robot motion -> FEM patch constraints -> tissue deformation',fill='white')
            im.save(frames/f'{n:05d}.png');n+=1
        capture()
        for step in range(a.steps):
            tools=np.zeros((2,1,7));tools[:,:,6]=100
            # Request motion in the current URDF tool_tip frame (-z inserts).
            for i in range(2):
                error=env.goal[i]-obs['tool_pose'][i,:3,3]
                delta=obs['tool_pose'][i,:3,:3].T@error
                length=np.linalg.norm(delta)
                if length>.00012:delta*=.00012/length
                tools[i,0,:3]=delta;tools[i,0,5]=.0005 if step<45 else -.0005
            scope=np.array([.02*np.sin(step/20),.015*np.cos(step/20),0.,0.])
            if a.contact_probe:
                tools[:,:,:6]=0;scope[:]=0
                # Original Rosota tool frame: -z inserts through the port.
                if step<a.approach_samples:tools[:,0,2]=-a.contact_travel_mm/1000/a.approach_samples
                elif step<a.approach_samples+a.close_samples:tools[:,:,6]=0
                if a.surface_approach and step<a.approach_samples:
                    for i in range(2):
                        delta=obs['tool_pose'][i,:3,:3].T@(targets[i]-obs['tool_pose'][i,:3,3])
                        tools[i,0,:3]=delta*min(1.,.00012/max(np.linalg.norm(delta),1e-9))
            if a.baseline_task:
                tools[:,:,:6]=0;scope[:]=0
                tools[:,:,6]=100 if step<approach_end or step>=pull_end else 0
                if step<approach_end:
                    for i,patch in enumerate(env.target_patches):
                        h=list(env.specs)[i]
                        local=np.asarray(info['grasp_geometry'][h]['centre_in_shaft_frame_m'])
                        # Stop once the patch enters the mouth, before driving
                        # the rigid tip to the centre of deformable tissue.
                        if np.all(np.abs(local)<[.0035,.003,.0045]):approached[h]=True
                        if approached[h]:continue
                        target=env.points()[env.config['data']['clusters'][patch]['indices']].mean(0)
                        delta=obs['tool_pose'][i,:3,:3].T@(target-obs['tool_pose'][i,:3,3])
                        tools[i,0,:3]=delta*min(1.,.00012/max(np.linalg.norm(delta),1e-9))
                if step==close_end:
                    from deformation_metrics import DeformationMeasure
                    pull_start=obs['tool_pose'][:,:3,3].copy()
                    pull_measure=DeformationMeasure(env.points(),env.tets)
                if close_end<=step<pull_end:
                    for i,h in enumerate(env.specs):
                        if h in env.grasp.held:
                            goal=pull_start[i]+np.array([0,.002*(step-close_end+1)/60,0])
                            tools[i,0,:3]=obs['tool_pose'][i,:3,:3].T@(goal-obs['tool_pose'][i,:3,3])
                if withdraw_start<=step<withdraw_end:
                    # Open first, then withdraw 7.2 mm along each shaft's
                    # outward tool axis, so the recovery state is unloaded.
                    tools[:,0,2]=.00012
            obs,reward,term,trunc,info=env.step({'tools':tools,'scope':scope})
            assert env.observation_space.contains(obs)
            max_move=np.maximum(max_move,np.linalg.norm(obs['tool_pose'][:,:3,3]-start,axis=1))
            mechanics.append(dict(step=env.steps,held=dict(info['held_patches']),
                pull_relative=pull_measure.measure(env.points()) if pull_measure else None,
                **info['deformation']))
            if a.baseline_task and env.steps%30==0:
                (a.out/'progress.json').write_text(json.dumps(dict(step=env.steps,
                    phase_ends=[approach_end,close_end,pull_end,release_end],
                    geometry=info['grasp_geometry'],held=info['held_patches'],
                    jaws=info['jaw_position_rad'],failure=info['failure'])),encoding='utf-8')
            if step%6==0 or env.done:capture()
            if env.done:break
            if a.baseline_task and env.steps>=release_end:break
        report.update(**info,status='passed' if not info['failure'] else 'failed',steps=env.steps,
            frames=n,fps=5.,control_hz=30,physics_hz=480,simulated_seconds=env.steps/30,
            physics_ticks_checked=len(env.trace),tip_excursion_m=max_move.tolist(),
            joint_excursion_rad=np.max(np.abs(obs['joint_position']-env.initial_q),axis=1).tolist(),
            scope_eye_excursion_m=float(np.linalg.norm(env.scope.pose[:3,3]-env.initial_scope[:3,3])),
            max_bite_m={h:max(x['trocar'][h]['bite_m'] for x in env.trace) for h in env.specs},
            max_body_fk_error_m=max(max(x['body_fk_error_m']) for x in env.trace))
        report['contact_summary']={h:dict(
            peak_residual_n=max(x['contact'][h]['residual_force_n'] for x in env.trace),
            stopped=any(x['contact'][h]['motion_latched'] for x in env.trace),
            jaw_limit_ticks=sum(x['contact'][h]['jaw_limited'] for x in env.trace)) for h in env.specs}
        if a.contact_probe:
            report['contact_validation']='unverified: reaction is not independent tissue contact evidence'
        report['final_jaw_position_rad']=obs['jaw_position'].tolist()
        np.savez_compressed(a.out/'tissue_motion.npz',initial=initial_tissue,final=env.points())
        (a.out/'deformation_trace.json').write_text(json.dumps(mechanics),encoding='utf-8')
        report['grasp_events']=env.grasp.events
        if a.baseline_task:
            report['peak_held_hands']=max(len(x['held']) for x in mechanics)
            report['peak_nonrigid_rms_m']=max(x['nonrigid_rms_m'] for x in mechanics)
            loaded=[x['pull_relative']['nonrigid_rms_m'] for x in mechanics
                if close_end<x['step']<=pull_end and x['pull_relative'] and len(x['held'])==2]
            peak=max(loaded,default=0.)
            final=mechanics[-1]['pull_relative']
            held_residual=final['nonrigid_rms_m'] if final else None
            loaded_free=[x['nonrigid_rms_m'] for x in mechanics
                if close_end<x['step']<=pull_end and len(x['held'])==2]
            free_peak=max(loaded_free,default=0.)
            residual=mechanics[-1]['nonrigid_rms_m']
            report['pull_deformation_rms_m']=peak
            report['post_release_residual_rms_m']=residual
            report['post_release_held_reference_rms_m']=held_residual
            report['loaded_free_reference_rms_m']=free_peak
            report['recovery_reference']='settled ungrasped reset shape; pull increment uses the separate pre-pull held shape'
            report['recovery_fraction']=1-residual/free_peak if free_peak>0 else None
            report['withdraw_samples']=[withdraw_start,withdraw_end]
            report['material_calibrated']=False
            report['acceptance_scope']='single engineering assisted grasp/pull/release; not full Rosota parity'
            report['phase_ends']=[approach_end,close_end,pull_end,release_end]
            if (env.steps<release_end or report['peak_held_hands']<2 or env.grasp.held
                    or peak<.0001 or report['recovery_fraction'] is None
                    or report['recovery_fraction']<.1):
                report['status']='failed';report['task_failure']='Incomplete dual grasp/pull deformation/release recovery'
        if not a.baseline_task and not a.contact_probe and not info['failure'] and (min(max_move)<.001 or report['scope_eye_excursion_m']<.0001):
            raise RuntimeError('Insufficient actual tool/scope movement')
        (a.out/'physics_trace.json').write_text(json.dumps(env.trace),encoding='utf-8')
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        save()
        if env:env.close()
        app.close()


if __name__=='__main__':main()
