"""Run real PhysX task episodes. Scripted reference is not a learned policy."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=ROOT/'out/ct_appendectomy/multitask_assets')
    p.add_argument('--out', type=Path, default=ROOT/'out/ct_appendectomy/multitask_benchmark')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--tasks', nargs='+', choices=['reach', 'dual_grasp', 'retract'], default=['reach', 'dual_grasp', 'retract'])
    p.add_argument('--policy', choices=['scripted', 'random', 'checkpoint'], default='scripted')
    p.add_argument('--train-steps', type=int, default=0)
    p.add_argument('--training-seed', type=int, default=10000)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--check-rgb', action='store_true')
    p.add_argument('--compare-scripted', action='store_true')
    a = p.parse_args()
    if a.train_steps < 0: p.error('Nonnegative training steps required')
    if (a.train_steps or a.checkpoint) and len(a.tasks) != 1: p.error('Train/replay one task per checkpoint')
    if a.policy == 'checkpoint' and not (a.train_steps or a.checkpoint): p.error('Checkpoint or training required')
    a.out.mkdir(parents=True, exist_ok=True)
    report = dict(status='starting', run_id=uuid.uuid4().hex, policy=a.policy,
                  learned_policy=a.policy=='checkpoint', episodes=[], appendectomy_completed=False)
    def save():
        temporary=a.out/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(a.out/'report.json')
    save()
    os.environ.setdefault('WARP_CACHE_PATH', str(ROOT/'.cache/isaac-warp'))
    os.environ.setdefault('OPTIX_CACHE_PATH', str(ROOT/'.cache/optix'))
    from isaacsim import SimulationApp
    runtime = ROOT/'.cache/kit'
    tokens = {name: runtime/name for name in ('shared_documents', 'app_documents', 'omni_documents', 'omni_config', 'omni_logs', 'cache', 'omni_cache', 'omni_global_cache')}
    for path in tokens.values(): path.mkdir(parents=True, exist_ok=True)
    app = SimulationApp({'headless': a.headless, 'renderer': 'RaytracedLighting', 'sync_loads': True,
        'extra_args': ['--/app/vulkan=false', '--/app/d3d12=true', '--/telemetry/enableNVDF=false',
                       '--/telemetry/enableSentry=false', '--/telemetry/useOpenEndpoint=false',
                       '--/privacy/extraDiagnosticDataOptIn=false', '--/crashreporter/enabled=false']+
                      [f'--/app/tokens/{key}={value.as_posix()}' for key, value in tokens.items()]})
    import numpy as np
    from bimanual_env import BimanualEnv, HANDS
    from bimanual_tools import SPEED
    env = None
    report['status']='running'
    save()
    try:
        env = BimanualEnv(app, a.assets, task=a.tasks[0])
        if a.train_steps or a.checkpoint:
            from bimanual_training import train
            model = train(env, a.train_steps, a.training_seed, a.out, a.checkpoint)
        for task in a.tasks:
            policies = ['scripted', a.policy] if a.compare_scripted and a.policy != 'scripted' else [a.policy]
            for seed, policy in [(s, p) for p in policies for s in a.seeds]:
                start = time.perf_counter()
                obs, info = env.reset(seed=seed, options={'task': task})
                if not a.headless:
                    from omni.kit.viewport.utility import get_active_viewport
                    viewport=get_active_viewport()
                    if viewport: viewport.set_active_camera('/World/Endoscope')
                reset_seconds = time.perf_counter()-start
                assert env.observation_space.contains(obs)
                if a.check_rgb:
                    from PIL import Image
                    q_before = env.control.points().copy()
                    image = env.render()
                    render_drift = float(np.max(np.abs(env.control.points()-q_before)))
                    if render_drift > 1e-7: raise RuntimeError('Rendering advanced FEM state')
                    Image.fromarray(image).save(a.out/f'{task}_{seed}_{policy}_reset.png')
                    report['render_state_max_error_m'] = max(report.get('render_state_max_error_m',0.), render_drift)
                # An invalid action must not advance physics or change commands.
                before = env.control.observation().copy()
                try: env.step(np.full(10, np.nan))
                except ValueError: pass
                else: raise AssertionError('Invalid action accepted')
                np.testing.assert_array_equal(before, env.control.observation())
                assert env.steps == 0
                rng = np.random.default_rng(seed)
                timings = []; total = 0.
                closing = False
                while not env.done:
                    if policy == 'checkpoint':
                        import torch
                        with torch.no_grad(): action = model.evaluate(torch.as_tensor(obs)[None], deterministic=True)[0][0].numpy()
                    elif policy == 'random': action = rng.uniform(-1, 1, 10)
                    else:
                        c = env.control
                        centers = c.centers()
                        distances = [np.linalg.norm(c.arms[h].pose()['position_m']-centers[i]) for h,i in zip(HANDS,env.patches)]
                        if max(distances) < .0035: closing = True
                        for hand, patch in zip(HANDS, env.patches):
                            held = c.held.get(hand)
                            target = centers[patch].copy()
                            toward_port = c.arms[hand].port-target
                            standoff = .018 if task == 'reach' else .0025
                            target += standoff*toward_port/np.linalg.norm(toward_port)
                            jaw = np.deg2rad(37)
                            if closing and task != 'reach': jaw = 0.
                            if held:
                                target = held['tool_origin'].copy()
                                if task == 'retract' and all(h in c.held for h in HANDS):
                                    target += env.patch_reference[list(HANDS).index(hand)]+[0, env.lift, 0]-held['capture_center']
                            c.target_position(hand, target, jaw)
                        actions = []
                        for hand in HANDS:
                            diff = c.targets[hand]-c.arms[hand].state
                            diff[3] = (diff[3]+np.pi)%(2*np.pi)-np.pi
                            limit = np.array([.25, .25, .25, .25, .5])
                            actions.append(np.clip(diff/(SPEED/60), -limit, limit))
                        action = np.concatenate(actions)
                    start = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(action)
                    timings.append(time.perf_counter()-start)
                    assert env.observation_space.contains(obs) and np.isfinite(reward)
                    total += reward
                report['episodes'].append(dict(seed=seed, task=task, policy=policy, steps=env.steps, return_=total,
                    terminated=terminated, truncated=truncated, reset_seconds=reset_seconds,
                    step_p50_ms=float(np.percentile(timings, 50)*1000),
                    step_p95_ms=float(np.percentile(timings, 95)*1000), **{k:v for k,v in info.items() if k != 'task'}))
                if info['failure']:
                    np.savez_compressed(a.out/f'{task}_{seed}_{policy}_failure.npz', rest=env.rest,
                        current=env.control.points(), tetrahedra=env.tets,
                        joints=env.control.observation(), patches=env.patches)
                save()
        report['status'] = 'completed'
        report['successes'] = sum(e['success'] for e in report['episodes'])
        report['by_policy'] = {policy: {'episodes': sum(e['policy']==policy for e in report['episodes']),
            'successes': sum(e['success'] for e in report['episodes'] if e['policy']==policy)}
            for policy in sorted({e['policy'] for e in report['episodes']})}
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = str(exc)
        raise
    finally:
        save()
        if env: env.close()
        app.close()


if __name__ == '__main__': main()
