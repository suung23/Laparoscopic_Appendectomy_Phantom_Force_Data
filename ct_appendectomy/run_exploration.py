"""Record autonomous geometric bimanual exploration in real Isaac frames."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=ROOT/'out/ct_appendectomy/multitask_assets')
    p.add_argument('--out',type=Path,default=ROOT/'out/ct_appendectomy/exploration')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--record-every',type=int,default=12)
    p.add_argument('--policy',choices=('geometric','checkpoint'),default='geometric')
    p.add_argument('--train-steps',type=int,default=0)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--max-steps',type=int,default=1800)
    a=p.parse_args()
    if a.record_every<1:p.error('Positive frame interval required')
    if a.train_steps<0:p.error('Nonnegative training steps required')
    if a.max_steps<1:p.error('Positive episode limit required')
    if a.policy=='checkpoint' and not (a.train_steps or a.checkpoint):p.error('Training or checkpoint required')
    a.out.mkdir(parents=True,exist_ok=True);frames=a.out/'frames';frames.mkdir(exist_ok=True)
    if any(frames.glob('*.png')):p.error('Choose a fresh output directory')
    report={'status':'starting','policy':'reactive geometric coverage planner' if a.policy=='geometric' else 'PPO',
            'learned_policy':a.policy=='checkpoint'}
    def save():(a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    for name,folder in [('WARP_CACHE_PATH','isaac-warp'),('OPTIX_CACHE_PATH','optix')]:
        os.environ.setdefault(name,str(ROOT/'.cache'/folder))
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
        from exploration_env import ExplorationEnv
        env=ExplorationEnv(app,a.assets,max_steps=a.max_steps)
        if a.train_steps or a.checkpoint:
            from bimanual_training import train
            model=train(env,a.train_steps,10000,a.out,a.checkpoint)
        obs,info=env.reset(seed=a.seed)
        assert env.observation_space.contains(obs)
        trajectory=[];timings=[];frame_count=0
        def capture():
            nonlocal frame_count
            before=env.base.control.points().copy()
            pixels=env.render()
            if np.max(np.abs(env.base.control.points()-before))>1e-7:raise RuntimeError('Capture advanced physics')
            im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,36),fill=(15,20,25))
            caption='GEOMETRIC EXPLORATION | NOT LEARNED RL' if a.policy=='geometric' else 'LEARNED PPO EXPLORATION | EVALUATION'
            draw.text((8,4),caption,fill='white')
            draw.text((8,20),f'Both tools + scope | coverage {info["coverage"]:.0%} | step {env.steps}',fill='white')
            if env.done:
                draw.rectangle((0,340,640,360),fill=(15,20,25))
                draw.text((8,344),'PASS' if info['success'] else 'FAILED: '+(info['failure'] or 'time limit'),fill='white')
            im.save(frames/f'{frame_count:05d}.png');frame_count+=1
        capture()
        while not env.done:
            start=time.perf_counter()
            if a.policy=='checkpoint':
                import torch
                with torch.no_grad():action=model.evaluate(torch.as_tensor(obs)[None],deterministic=True)[0][0].numpy()
            else:action=env.reference_action()
            obs,reward,term,trunc,info=env.step(action)
            timings.append(time.perf_counter()-start)
            assert env.observation_space.contains(obs) and np.isfinite(obs).all()
            trajectory.append({'step':env.steps,'positions_m':env.positions().tolist(),
                'camera':env.camera.tolist(),'coverage':info['coverage'],'failure':info['failure']})
            if env.steps%a.record_every==0 or env.done:capture()
        report.update(status='passed' if info['success'] else 'failed',**info,
            steps=env.steps,frames=frame_count,fps=60/a.record_every,
            control_step_p50_ms=float(np.percentile(timings,50)*1000),
            control_step_p95_ms=float(np.percentile(timings,95)*1000),
            tissue_health=env.base._metrics()['health'],seed=a.seed,
            engine='Isaac Sim 5.1 / PhysX GPU',action_dimensions=16,state_dimensions=42)
        (a.out/'trajectory.json').write_text(json.dumps(trajectory),encoding='utf-8')
        if info['success'] and (min(info['tool_excursion_m'])<.005 or not info['camera_changed']):
            raise RuntimeError('Coverage without real tool/camera movement')
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        save()
        if env:env.close()
        app.close()


if __name__=='__main__':main()
