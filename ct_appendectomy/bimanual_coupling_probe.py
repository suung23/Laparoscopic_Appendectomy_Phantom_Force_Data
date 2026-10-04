"""Bimanual assisted load-path test with explicit paired reset comparison."""
import hashlib
import json
import numpy as np


def run(env):
    if env.grasp_mode!='solver_attachment':raise ValueError('Solver attachment required')
    hands=['left','right'];trials={};reference=None;commands=None
    for trial,enabled in enumerate([False,True]):
        if trial:env.reset(seed=42)
        label='enabled' if enabled else 'disabled'
        initial=env.points().copy();joints=np.array([env.q(h) for h in hands]);obs=env.observe()
        start=obs['tool_pose'][:,:3,3].copy()
        if reference is None:reference=(initial.copy(),joints.copy())
        reset_error=dict(tissue_max_m=float(np.linalg.norm(initial-reference[0],axis=1).max()),
            joint_max_rad=float(np.abs(joints-reference[1]).max()))
        if reset_error['tissue_max_m']>1e-5 or reset_error['joint_max_rad']>1e-5:
            raise RuntimeError('Paired initial states differ beyond 10 um or 10 urad')
        centers=[]
        for i,h in enumerate(hands):
            index=env.target_patches[i];cluster=env.config['data']['clusters'][index]
            center=initial[cluster['indices']].mean(0);centers.append(center)
            R,_=env.grasp.body_pose_reader(h)
            env.grasp.capture(h,index,initial,center,R,evidence='bimanual engineering attachment diagnostic')
            jaws=env.robots[h].get_joint_positions(joint_indices=env.jaw_indices[h])
            env.grasp.held[h]['jaw_deg']=float(np.rad2deg(jaws[0]-jaws[1]))
            if not enabled:env.grasp.enabled(cluster).Set(False)
        if commands is None:
            directions=np.asarray(centers)-initial[env.pin].mean(0)
            commands=.002*directions/np.linalg.norm(directions,axis=1)[:,None]
        rows=[];failure=None;activation=0.
        for step in range(50):
            action=np.zeros((2,1,7));goals=start+commands*np.clip((step-3)/20,0,1)
            for i,h in enumerate(hands):
                action[i,0,:3]=obs['tool_pose'][i,:3,:3].T@(goals[i]-obs['tool_pose'][i,:3,3])
                if step==34:env.grasp.release(h)
                if step>=34:action[i,0,6]=100
            obs,_,terminated,truncated,info=env.step({'tools':action,'scope':np.zeros(4)})
            current=env.points()
            if step<4:activation=max(activation,float(np.linalg.norm(current-initial,axis=1).max()))
            forces=[];residual=[]
            for i,h in enumerate(hands):
                R,p=env.grasp.body_pose_reader(h);forces.append((R@env.sensor(h)[0][:3]).tolist())
                cluster=env.config['data']['clusters'][env.target_patches[i]]
                attachment=env.stage.GetPrimAtPath(cluster['path']+'Attachment')
                local=np.asarray(attachment.GetAttribute('omniphysics:localPositionsSrc1').Get(),float)
                residual.append(float(np.sqrt(np.mean(np.sum((current[cluster['indices']]-(local@R.T+p))**2,axis=1)))))
            rows.append(dict(step=step,force_world_n=forces,constraint_rms_m=residual,
                inverted_tetrahedra_max=max(x['tissue_health']['inverted_tetrahedra'] for x in env.trace[-16:]),
                tip_positions_m=obs['tool_pose'][:,:3,3].tolist()))
            if terminated or truncated:failure=info['failure'] or 'episode limit';break
        for h in hands:env.grasp.release(h)
        result=dict(failure=failure,reset_error=reset_error,initial_sha256=hashlib.sha256(initial.tobytes()+joints.tobytes()).hexdigest(),
            settlement=env.settlement,activation_max_m=activation,commanded_displacement_m=commands.tolist(),
            tracking_error_m=np.linalg.norm(obs['tool_pose'][:,:3,3]-start-commands,axis=1).tolist(),trace=rows)
        trials[label]=result
        (env.out/('bimanual_coupling_'+label+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Bimanual coupling '+label+' completed',flush=True)
    valid=all(t['failure'] is None and len(t['trace'])==50 and max(t['tracking_error_m'])<.0002
        and all(r['inverted_tetrahedra_max']==0 for r in t['trace']) for t in trials.values())
    effect=noise=None
    if valid:
        f={k:np.asarray([r['force_world_n'] for r in t['trace']]) for k,t in trials.items()}
        diff=(f['enabled']-f['enabled'][:4].mean(0))-(f['disabled']-f['disabled'][:4].mean(0))
        effect=np.linalg.norm(diff[30:34].mean(0),axis=1)
        noise=np.linalg.norm(diff[:4]-diff[:4].mean(0),axis=2).max(0)
    passed=bool(valid and np.all(effect>np.maximum(.001,5*noise)) and trials['enabled']['activation_max_m']<.001)
    return dict(status='passed' if passed else 'failed',policy='paired bimanual assisted constraint diagnostic',
        scope='Reaction transfer, not calibrated tissue force, friction grasp, or complete force balance',
        force_effect_n=None if effect is None else effect.tolist(),noise_n=None if noise is None else noise.tolist(),
        conditions={k:{a:b for a,b in v.items() if a!='trace'} for k,v in trials.items()},material_calibrated=False,frictional_grasp=False)
