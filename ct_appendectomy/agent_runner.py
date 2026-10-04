"""Run local robot-agent episodes, train PPO, or replay a checkpoint in Isaac."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'out/ct_appendectomy/suite/forceps')
    parser.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/agent')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--policy',choices=('scripted','random','checkpoint'),default='scripted')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--train-steps',type=int,default=0)
    parser.add_argument('--rollout-steps',type=int,default=256)
    parser.add_argument('--episodes',type=int,default=3,help='0: continuous visible demo until window closes')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--rgb-observation',action='store_true')
    parser.add_argument('--demo-warm-start',action='store_true',help='Explicit behavior cloning before PPO; recorded separately')
    parser.add_argument('--contract-smoke',action='store_true',help='Check rendering, invalid action and regrasp reference before running')
    parser.add_argument('--record-every',type=int,default=0,help='Save a real RTX frame every N evaluation actions; 0 disables recording')
    args=parser.parse_args()
    if args.train_steps<0 or args.rollout_steps<2 or args.episodes<0 or args.record_every<0:parser.error('Invalid run limits')
    if args.headless and not args.episodes:parser.error('Headless evaluation must have a finite episode limit')
    if args.policy=='checkpoint' and not args.checkpoint and not args.train_steps:parser.error('Checkpoint required')
    if args.rgb_observation and args.train_steps:parser.error('Bundled PPO uses privileged state; RGB API is for external policies')
    if args.demo_warm_start and (not args.train_steps or args.checkpoint):parser.error('Warm start requires new training without a checkpoint')
    stage=args.assets/'attached.usda';config=args.assets/'forceps_config.json'
    if not stage.is_file() or not config.is_file():parser.error('Prepare forceps assets with run-agent.ps1 -Prepare')
    args.out.mkdir(parents=True,exist_ok=True)
    report={'status':'starting','run_id':uuid.uuid4().hex,'appendectomy_completed':False,'policy':args.policy,
            'seed':args.seed,'training_steps_requested':args.train_steps,'episodes':[]}
    def write_report():
        temp=args.out/'report.tmp'
        temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(args.out/'report.json')
    write_report()
    os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/isaac-warp'))
    os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
    from isaacsim import SimulationApp
    runtime=ROOT/'.cache/kit'
    tokens={name:runtime/name for name in ('shared_documents','app_documents','omni_documents',
        'omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
    for path in tokens.values():path.mkdir(parents=True,exist_ok=True)
    app=SimulationApp({'headless':args.headless,'width':1280,'height':720,'renderer':'RaytracedLighting',
        'sync_loads':True,'hide_ui':args.headless,'extra_args':[
        '--/app/vulkan=false','--/app/d3d12=true','--/telemetry/enableNVDF=false',
        '--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false',
        '--/privacy/extraDiagnosticDataOptIn=false','--/crashreporter/enabled=false']+
        [f'--/app/tokens/{key}={value.as_posix()}' for key,value in tokens.items()]})
    env=None
    try:
        import numpy as np
        import gymnasium as gym
        import torch
        from PIL import Image
        from agent_env import bind_simulation_app,scripted_action,ENV_ID,STATE_NAMES
        from ppo_agent import ActorCritic,advantages,update,save_checkpoint,load_checkpoint
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        bind_simulation_app(app)
        env=gym.make(ENV_ID,stage_path=stage,config_path=config,
                     observation_mode='rgb_state' if args.rgb_observation else 'state')
        model=ActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
        contract={'env_id':ENV_ID,'state_names':list(STATE_NAMES),
                  'stage_sha256':hashlib.sha256(stage.read_bytes()).hexdigest(),
                  'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
                  'gamma':.99,'action_count':2}
        trained_steps=0
        warm_start_used=bool(args.demo_warm_start)
        if args.checkpoint:
            metadata=load_checkpoint(args.checkpoint,model,optimizer,contract)
            trained_steps=metadata['trained_steps']
            warm_start_used=warm_start_used or metadata.get('demo_warm_start',False)
        report.update(contract,trained_steps_before=trained_steps,observation_mode=env.get_wrapper_attr('observation_mode'),
                      anatomical_randomization=False,compact_state_is_privileged=True,
                      checkpoint_loaded=bool(args.checkpoint),demo_warm_start=args.demo_warm_start,
                      engine='Isaac Sim 5.1 / PhysX GPU',control_dt_s=1/30,
                      torch_version=str(torch.__version__),policy_device='cpu')
        panel=None;paused=False;requested_policy=None
        if not args.headless:
            import omni.ui as ui
            panel=ui.Window('Robot RL agent',width=390,height=280)
            with panel.frame:
                with ui.VStack(spacing=5):
                    ui.Label('Contact-lift task / 2 actuator commands')
                    policy_label=ui.Label('Policy: '+('PPO training' if args.train_steps else args.policy))
                    status_label=ui.Label('Initializing episode')
                    reward_label=ui.Label('')
                    state_label=ui.Label('')
                    pause_label=ui.Label('Running')
                    def toggle():
                        nonlocal paused
                        paused=not paused
                        pause_label.text='Paused' if paused else 'Running'
                    ui.Button('Pause / resume',clicked_fn=toggle)
                    ui.Label('Select policy for the next episode:')
                    def choose_policy(name):
                        nonlocal requested_policy
                        requested_policy=name
                    with ui.HStack():
                        ui.Button('Scripted',clicked_fn=lambda:choose_policy('scripted'))
                        ui.Button('Random',clicked_fn=lambda:choose_policy('random'))
                        ui.Button('Learned',enabled=bool(args.checkpoint or args.train_steps),clicked_fn=lambda:choose_policy('checkpoint'))
        def visible_update(phase,episode,step,reward,info):
            if args.headless:return
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport:viewport.set_active_camera('/World/Endoscope')
            policy_label.text='Policy: '+phase
            status_label.text=f'{phase} | episode {episode} | step {step}'
            reward_label.text=f'Return {reward:.3f} | success {info.get("is_success",False)}'
            state_label.text=f'Lift {info.get("lift_m",0)*1000:.2f} mm | base drift {info.get("pin_drift_m",0)*1000:.3f} mm'
            if paused:
                env.unwrapped.sim.pause()
                while paused and app.is_running():app.update()
                if app.is_running():env.unwrapped.sim.play()
        def reset(seed):
            obs,info=env.reset(seed=seed)
            visible_update('reset',0,0,0,info)
            return obs
        def state(obs):return obs['state'] if isinstance(obs,dict) else obs
        def predict(obs,deterministic=False):
            with torch.no_grad():return model.evaluate(torch.as_tensor(state(obs))[None],deterministic=deterministic)
        if args.contract_smoke:
            baseline=reset(args.seed)
            core=env.unwrapped
            before=core._observe().copy()
            image=env.render()
            render_error=float(np.max(np.abs(core._observe()-before)))
            if image.shape!=(360,640,3) or render_error>1e-5:raise RuntimeError('RGB contract failed')
            try:env.step([float('nan'),0])
            except ValueError:pass
            else:raise RuntimeError('Invalid action accepted')
            if not np.array_equal(core._observe(),before):raise RuntimeError('Invalid action changed state')
            for _ in range(5):env.step([0,1])
            for _ in range(72):env.step([1/1.8,0])
            env.step([0,1/1.8])
            if core.closed_reference is None or abs(core.reference_command_lift-.001)>1e-6:
                raise RuntimeError('Nonzero-height grasp reference failed')
            restored=reset(args.seed)
            reset_error=float(np.max(np.abs(state(restored)-state(baseline))))
            if reset_error>1e-5:raise RuntimeError('Agent reset failed')
            report['contract_smoke']={'status':'passed','render_state_error':render_error,
                'reset_error':reset_error,'invalid_action_rejected_without_mutation':True,
                'nonzero_height_reference_passed':True,'gym_registered':True}
            write_report()
        if args.demo_warm_start:
            observations=[];actions=[]
            obs=reset(args.seed)
            for step in range(220):
                action=scripted_action(step)
                observations.append(obs);actions.append(action)
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:break
            if not info['is_success']:raise RuntimeError('Demonstration failed; refusing warm start')
            x=torch.as_tensor(np.asarray(observations));y=torch.as_tensor(np.asarray(actions))
            bc_optimizer=torch.optim.Adam(model.actor.parameters(),lr=.002)
            for _ in range(1000):
                loss=(model.actor(x).tanh()-y).square().mean()
                bc_optimizer.zero_grad();loss.backward();bc_optimizer.step()
            with torch.no_grad():model.log_std.fill_(-3.)
            report['demonstration']={'steps':len(actions),'is_success':True,'cloning_mse':float(loss.detach())}
        initial_parameters=torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
        started=time.perf_counter();total=0;episode=0;episode_return=0.;episode_steps=0
        if args.train_steps:
            obs=reset(args.seed)
            training_episodes=[];updates=[]
            with (args.out/'training.jsonl').open('a',encoding='utf-8') as journal:
                while total<args.train_steps and app.is_running():
                    batch={key:[] for key in ('obs','raw','log_prob','value','next_value','reward','terminated','ended')}
                    for _ in range(min(args.rollout_steps,args.train_steps-total)):
                        if not app.is_running():break
                        action,raw,logp,value,_=predict(obs)
                        next_obs,reward,term,trunc,info=env.step(action[0].numpy())
                        with torch.no_grad():next_value=predict(next_obs,True)[3].item()
                        entries=(obs,raw[0].numpy(),logp.item(),value.item(),next_value,reward,term,term or trunc)
                        for key,value in zip(batch,entries):batch[key].append(value)
                        total+=1;episode_steps+=1;episode_return+=reward
                        visible_update('training',episode+1,episode_steps,episode_return,info)
                        obs=next_obs
                        if term or trunc:
                            result={'run_id':report['run_id'],'episode':episode,'steps':episode_steps,'return':episode_return,
                                    'success':info['is_success'],'termination_reason':info['termination_reason']}
                            training_episodes.append(result)
                            journal.write(json.dumps(result)+'\n');journal.flush()
                            episode+=1;episode_return=0.;episode_steps=0
                            obs=reset(args.seed+episode)
                    if not batch['obs']:break
                    adv,returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['ended'])
                    batch['advantage']=adv;batch['return']=returns
                    metrics=update(model,optimizer,batch)
                    metrics['steps']=total;updates.append(metrics)
                    checkpoint_metadata=dict(contract,trained_steps=trained_steps+total,seed=args.seed,
                                             demo_warm_start=warm_start_used,run_id=report['run_id'])
                    save_checkpoint(args.out/'policy.pt',model,optimizer,checkpoint_metadata)
                    report['training']={'steps':total,'updates':updates,'episodes':training_episodes}
                    write_report()
                    print(json.dumps({'training_steps':total,'last_update':metrics}),flush=True)
            if total!=args.train_steps:raise RuntimeError('Training interrupted before requested step count')
            final_parameters=torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
            report['training']['parameter_l2_change']=float(torch.linalg.vector_norm(final_parameters-initial_parameters))
            if report['training']['parameter_l2_change']==0:raise RuntimeError('PPO did not update parameters')
            report['training']['wall_seconds']=time.perf_counter()-started
            # Verify serialization by reloading the actual saved checkpoint.
            load_checkpoint(args.out/'policy.pt',model,None,contract)
            report['checkpoint_roundtrip_passed']=True
            report['checkpoint_sha256']=hashlib.sha256((args.out/'policy.pt').read_bytes()).hexdigest()
        evaluation_policy='checkpoint' if args.train_steps else args.policy
        report['evaluation_policy']=evaluation_policy
        report['status']='running'
        write_report()
        episode=0
        with (args.out/'episodes.jsonl').open('a',encoding='utf-8') as journal:
            while app.is_running() and (not args.episodes or episode<args.episodes):
                if requested_policy is not None:
                    evaluation_policy=requested_policy
                    requested_policy=None
                    report['evaluation_policy']=evaluation_policy
                seed=args.seed+10000+episode
                obs=reset(seed);episode_return=0.;trajectory=[]
                recorded=[]
                frame_directory=args.out/('frames_'+report['run_id']+'_'+str(episode))
                def record_frame(action_step,frame_info):
                    frame_directory.mkdir(parents=True,exist_ok=True)
                    name=f'{len(recorded):05d}.png'
                    Image.fromarray(env.render()).save(frame_directory/name)
                    recorded.append({'file':name,'action_step':action_step,'simulation_time_s':action_step/30,
                                     'lift_m':frame_info.get('lift_m',0),'success':frame_info.get('is_success',False)})
                if args.record_every:record_frame(0,{})
                for step in range(220):
                    if not app.is_running():break
                    if evaluation_policy=='scripted':action=scripted_action(step)
                    elif evaluation_policy=='random':action=env.action_space.sample()
                    else:action=predict(obs,True)[0][0].numpy()
                    before=state(obs).copy()
                    obs,reward,term,trunc,info=env.step(action)
                    if not env.observation_space.contains(obs):raise RuntimeError('Observation contract violation')
                    episode_return+=reward
                    trajectory.append({'step':step,'observation':before.tolist(),'action':action.tolist(),
                                       'next_observation':state(obs).tolist(),'reward':reward,
                                       'terminated':term,'truncated':trunc,
                                       'termination_reason':info['termination_reason']})
                    visible_update(evaluation_policy,episode+1,step+1,episode_return,info)
                    if (step+1)%25==0:
                        report['progress']={'episode':episode,'step':step+1,'policy':evaluation_policy,'return':episode_return}
                        write_report()
                    if args.record_every and ((step+1)%args.record_every==0 or term or trunc):
                        record_frame(step+1,info)
                    if term or trunc:break
                if not app.is_running():break
                image=env.render()
                Image.fromarray(image).save(args.out/'latest_endoscope.png')
                result={'run_id':report['run_id'],'policy':evaluation_policy,'episode':episode,'seed':seed,'steps':step+1,'return':episode_return,
                        'success':info['is_success'],'terminated':term,'truncated':trunc,
                        'termination_reason':info['termination_reason'],'lift_m':info['lift_m'],
                        'pin_drift_m':info['pin_drift_m'],'inverted_tetrahedra':info['tissue_health']['inverted_tetrahedra']}
                journal.write(json.dumps(result)+'\n');journal.flush()
                if args.record_every:
                    recording={'policy':evaluation_policy,'control_dt_s':1/30,'capture_every_actions':args.record_every,
                               'frames':recorded,'result':result,'synthetic_frames':False}
                    (frame_directory/'recording.json').write_text(json.dumps(recording,indent=2),encoding='utf-8')
                    report.setdefault('recordings',[]).append(str(frame_directory.relative_to(args.out)))
                # Keep bounded current-session summaries during continuous demos.
                report['episodes']=(report['episodes']+[result])[-100:]
                (args.out/'latest_trajectory.json').write_text(json.dumps(trajectory),encoding='utf-8')
                write_report();print(json.dumps(result),flush=True)
                episode+=1
        report['status']='passed' if args.episodes and episode==args.episodes else 'stopped'
        report['evaluation_episode_count']=episode
        report['evaluation_successes']=sum(int(item['success']) for item in report['episodes'])
        report['evaluation_summary_window']=len(report['episodes'])
        report['task_mastery_claimed']=False
        write_report()
    except BaseException as exc:
        import traceback
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());write_report()
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
