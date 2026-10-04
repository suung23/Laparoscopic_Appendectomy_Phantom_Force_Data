"""Record and verify the real CT articulation/RCM path, without learned claims."""
import argparse
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
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    frames=a.out/'frames';frames.mkdir(exist_ok=True)
    if any(frames.glob('*.png')):p.error('Use a fresh output folder')
    report={'status':'starting','learned_policy':False,'policy':'geometric two-tip target tracking'}
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
        env=RosotaCTEnv(app,a.assets,a.fr3_urdf,a.out,max_steps=a.steps)
        obs,info=env.reset(seed=42);report.update(status='running',**info);save()
        n=0;start=obs['tool_pose'][:,:3,3].copy();max_move=np.zeros(2)
        def capture():
            nonlocal n
            before=env.points().copy();rgb=env.render()
            if np.max(np.abs(env.points()-before))>1e-7:raise RuntimeError('Rendering moved tissue')
            im=Image.fromarray(rgb);d=ImageDraw.Draw(im);d.rectangle((0,0,640,38),fill=(15,20,25))
            d.text((8,4),'ROSOTA CONTROLLER / PHYSX FR3 JOINT FEEDBACK',fill='white')
            d.text((8,21),f'Engineering ports | Scope RCM | step {env.steps}',fill='white')
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
            obs,reward,term,trunc,info=env.step({'tools':tools,'scope':scope})
            assert env.observation_space.contains(obs)
            max_move=np.maximum(max_move,np.linalg.norm(obs['tool_pose'][:,:3,3]-start,axis=1))
            if step%6==0 or env.done:capture()
            if env.done:break
        report.update(**info,status='passed' if not info['failure'] else 'failed',steps=env.steps,
            frames=n,fps=5.,control_hz=30,physics_hz=480,simulated_seconds=env.steps/30,
            physics_ticks_checked=len(env.trace),tip_excursion_m=max_move.tolist(),
            joint_excursion_rad=np.max(np.abs(obs['joint_position']-env.initial_q),axis=1).tolist(),
            scope_eye_excursion_m=float(np.linalg.norm(env.scope.pose[:3,3]-env.initial_scope[:3,3])),
            max_bite_m={h:max(x['trocar'][h]['bite_m'] for x in env.trace) for h in env.specs},
            max_body_fk_error_m=max(max(x['body_fk_error_m']) for x in env.trace))
        if not info['failure'] and (min(max_move)<.001 or report['scope_eye_excursion_m']<.0001):
            raise RuntimeError('Insufficient actual tool/scope movement')
        (a.out/'physics_trace.json').write_text(json.dumps(env.trace),encoding='utf-8')
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        save()
        if env:env.close()
        app.close()


if __name__=='__main__':main()
