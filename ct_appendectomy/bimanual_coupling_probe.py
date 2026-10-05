"""Bimanual assisted load-path test with explicit paired reset comparison."""
import hashlib
import json
import numpy as np
if __package__:
    from .physics_substep_audit import EpisodeSubstepHealth
else:
    from physics_substep_audit import EpisodeSubstepHealth


def summarize_solver_ticks(control_trace,solver_substeps,expected_control_ticks=16):
    """Aggregate the exact FEM-health intervals created inside one RL action."""
    if isinstance(solver_substeps,bool) or not isinstance(solver_substeps,(int,np.integer)) or solver_substeps not in (1,2,4):
        raise ValueError('solver_substeps must be 1, 2, or 4')
    if isinstance(expected_control_ticks,bool) or not isinstance(expected_control_ticks,(int,np.integer)) or expected_control_ticks<1:
        raise ValueError('expected_control_ticks must be a positive integer')
    intervals=[row.get('physics_substep_health') for row in control_trace]
    episode=EpisodeSubstepHealth()
    for interval in intervals:episode.update(interval)
    aggregate=episode.as_dict()
    valid_counts=aggregate['sampling_complete'] and all(
        1<=item['physics_ticks']<=solver_substeps for item in intervals)
    sampled_ticks=aggregate['physics_ticks']
    expected_ticks=int(expected_control_ticks)*int(solver_substeps)
    sampling_complete=bool(valid_counts)
    return dict(physics_substep_sampling_complete=sampling_complete,
        controller_ticks_observed=len(control_trace),controller_ticks_expected=int(expected_control_ticks),
        controller_interval_execution_complete=len(control_trace)==expected_control_ticks,
        physics_substep_execution_complete=bool(sampling_complete and
            len(control_trace)==expected_control_ticks and sampled_ticks==expected_ticks),
        physics_substeps_per_control_tick=int(solver_substeps),
        physics_ticks=sampled_ticks,expected_physics_ticks=expected_ticks,
        ticks_with_inversion=aggregate['ticks_with_inversion'],
        max_inverted_tetrahedra_in_one_tick=aggregate['max_inverted_tetrahedra_in_one_tick'],
        inverted_tetrahedra_max=aggregate['max_inverted_tetrahedra_in_one_tick'],
        minimum_signed_volume_ratio=aggregate['minimum_signed_volume_ratio'],
        max_p99_absolute_volume_change=aggregate['max_p99_absolute_volume_change'],
        all_sampled_ticks_inversion_free=bool(sampling_complete and aggregate['ticks_with_inversion']==0))


def run(env):
    if env.grasp_mode!='solver_attachment':raise ValueError('Solver attachment required')
    hands=['left','right'];trials={};reference=None;commands=None
    for trial,enabled in enumerate([False,True]):
        if trial:env.reset(seed=42)
        label='enabled' if enabled else 'disabled'
        initial=env.points().copy();joints=np.array([env.q(h) for h in hands]);obs=env.observe()
        start=obs['tool_pose'][:,:3,3].copy()
        if not np.isfinite(initial).all() or not np.isfinite(joints).all():raise RuntimeError('Nonfinite reset state')
        np.savez_compressed(env.out/('bimanual_initial_'+label+'.npz'),tissue=initial,joints=joints,rest=env.rest,tets=env.tets)
        from reset_comparison import compare_states
        state=dict(tissue=initial,rest=env.rest.copy(),tets=env.tets.copy(),joints=joints)
        if reference is None:reference={k:v.copy() for k,v in state.items()}
        reset_error=compare_states(reference,state)
        (env.out/('bimanual_reset_'+label+'.json')).write_text(json.dumps(reset_error,indent=2),encoding='utf-8')
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
            trace_start=len(env.trace)
            obs,_,terminated,truncated,info=env.step({'tools':action,'scope':np.zeros(4)})
            control_trace=env.trace[trace_start:]
            substep_audit=summarize_solver_ticks(control_trace,env.physics_substeps)
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
                **substep_audit,tip_positions_m=obs['tool_pose'][:,:3,3].tolist()))
            if terminated or truncated:failure=info['failure'] or 'episode limit';break
        for h in hands:env.grasp.release(h)
        result=dict(failure=failure,reset_error=reset_error,initial_sha256=hashlib.sha256(initial.tobytes()+joints.tobytes()).hexdigest(),
            settlement=env.settlement,activation_max_m=activation,commanded_displacement_m=commands.tolist(),
            tracking_error_m=np.linalg.norm(obs['tool_pose'][:,:3,3]-start-commands,axis=1).tolist(),trace=rows)
        trials[label]=result
        (env.out/('bimanual_coupling_'+label+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Bimanual coupling '+label+' completed',flush=True)
    valid=all(t['failure'] is None and len(t['trace'])==50 and max(t['tracking_error_m'])<.0002
        and all(r['physics_substep_sampling_complete'] and r['physics_substep_execution_complete']
            and r['inverted_tetrahedra_max']==0 for r in t['trace']) for t in trials.values())
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
