"""Paired constraint-load diagnostic. This is NOT contact-based grasping.

The identical commanded robot displacement is applied with the
attachment enabled and disabled. No tissue positions or rigid poses are set.
This tests the load path to the robot, not calibrated force accuracy.
"""
import json
import numpy as np
from deformation_metrics import DeformationMeasure


def run(env):
    if env.grasp_mode!='solver_attachment':raise ValueError('Dynamic solver attachment required')
    trials={};displacement=None
    for trial,enabled in enumerate((False,True)):
        if trial:env.reset(seed=42)
        label='enabled' if enabled else 'disabled'
        initial=env.points().copy();metric=DeformationMeasure(initial,env.tets)
        cluster=env.config['data']['clusters'][env.target_patches[0]]
        center=initial[cluster['indices']].mean(0)
        R,p=env.grasp.body_pose_reader('left')
        env.grasp.capture('left',env.target_patches[0],initial,center,R,
                          evidence='isolated constraint-load diagnostic; no jaw capture claim')
        jaws=env.robots['left'].get_joint_positions(joint_indices=env.jaw_indices['left'])
        env.grasp.held['left']['jaw_deg']=float(np.rad2deg(jaws[0]-jaws[1]))
        if not enabled:env.grasp.enabled(cluster).Set(False)
        rows=[];controls=[];activation_peak=0.;failure=None;loaded=None
        obs=env.observe();start=obs['tool_pose'][0,:3,3].copy()
        if displacement is None:
            direction=center-initial[env.pin].mean(0)
            direction/=np.linalg.norm(direction)
            displacement=.002*direction
        for step in range(50):
            # Four activation samples, twenty 2-mm axial ramp, ten hold, sixteen released.
            tools=np.zeros((2,1,7));tools[1,0,6]=100
            goal=start+displacement*np.clip((step-3)/20,0,1)
            tools[0,0,:3]=obs['tool_pose'][0,:3,:3].T@(goal-obs['tool_pose'][0,:3,3])
            if step==34:env.grasp.release('left')
            if step>=34:tools[0,0,6]=100
            obs,_,terminated,truncated,info=env.step({'tools':tools,'scope':np.zeros(4)})
            q=env.points()
            body_R,body_p=env.grasp.body_pose_reader('left')
            attachment=env.stage.GetPrimAtPath(cluster['path']+'Attachment')
            local=np.asarray(attachment.GetAttribute('omniphysics:localPositionsSrc1').Get(),float)
            target=local@body_R.T+body_p
            controls.append(dict(step=step+1,
                arm_effort_nm=np.asarray(env.robots['left'].get_measured_joint_efforts(joint_indices=env.indices['left']),float).tolist(),
                constraint_rms_m=float(np.sqrt(np.mean(np.sum((q[cluster['indices']]-target)**2,axis=1)))),
                tip_position_m=obs['tool_pose'][0,:3,3].tolist()))
            if step<4:activation_peak=max(activation_peak,float(np.linalg.norm(q-initial,axis=1).max()))
            for sample in env.trace[-16:]:
                rows.append(dict(tick=len(rows)+1,wrench_local=sample['contact']['left']['mount_reaction'],
                    inverted_tetrahedra=sample['tissue_health']['inverted_tetrahedra'],
                    minimum_volume_ratio=sample['tissue_health']['minimum_signed_volume_ratio']))
            if terminated or truncated:
                failure=info['failure'] or 'Unexpected episode limit';break
            if step==33:loaded=metric.measure(q)
        env.grasp.release('left')
        trials[label]=dict(activation_max_displacement_m=activation_peak,
            tool_displacement_m=(obs['tool_pose'][0,:3,3]-start).tolist(),
            commanded_displacement_m=displacement.tolist(),control_samples=controls,
            settlement=env.settlement,
            loaded_deformation=loaded,final_deformation=metric.measure(env.points()),
            failure=failure,trace=rows)
        (env.out/f'coupling_{label}.json').write_text(json.dumps(trials[label],indent=2),encoding='utf-8')
        print(f'Coupling diagnostic {label}: {len(rows)} physics ticks',flush=True)
    valid=all(t['failure'] is None and len(t['trace'])==800 and
        not any(r['inverted_tetrahedra'] for r in t['trace']) and
        np.linalg.norm(np.asarray(t['tool_displacement_m'])-t['commanded_displacement_m'])<.0002
        for t in trials.values())
    effect=None;noise=None
    if valid:
        w={k:np.array([r['wrench_local'][:3] for r in v['trace']]) for k,v in trials.items()}
        residual={k:v-v[:64].mean(0) for k,v in w.items()}
        differential=residual['enabled']-residual['disabled']
        effect=float(np.linalg.norm(differential[30*16:34*16].mean(0)))
        noise=float(np.linalg.norm(differential[:64]-differential[:64].mean(0),axis=1).max())
    passed=bool(valid and effect>max(.001,5*noise) and
        trials['enabled']['activation_max_displacement_m']<.001)
    return dict(status='passed' if passed else 'failed',
        policy='paired dynamic constraint enabled/disabled load-path diagnostic',
        acceptance_scope='engineering reaction transfer only; not calibrated tissue mechanics or contact grasp',
        prescribed_tool_displacement_m=.002,physics_ticks_per_condition=800,
        mount_force_effect_n=effect,activation_differential_noise_n=noise,
        grasp_assist=True,frictional_grasp=False,material_calibrated=False,
        conditions={k:{a:b for a,b in v.items() if a not in ('trace','control_samples')} for k,v in trials.items()})
