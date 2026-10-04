"""Known three-axis external-load test in a collision-free robot fixture."""
import json
import numpy as np


def evaluate_response(measured,expected,release,noise,tolerance_n=.005):
    measured,expected=np.asarray(measured,float),np.asarray(expected,float)
    if measured.shape!=(3,3) or expected.shape!=(3,3) or not np.isfinite([measured,expected]).all():
        raise ValueError('Three finite XYZ response rows required')
    release,noise=np.asarray(release,float),np.asarray(noise,float)
    if release.shape!=(3,) or noise.shape!=(3,) or not np.isfinite([release,noise]).all() or np.any(release<0) or np.any(noise<0):
        raise ValueError('Three finite nonnegative release and noise values required')
    if not np.isfinite(tolerance_n) or tolerance_n<=0:raise ValueError('Positive finite tolerance required')
    sign=1 if np.sum(measured*expected)>=0 else -1
    errors=np.linalg.norm(measured-sign*expected,axis=1)
    return dict(status='passed' if np.all(errors<tolerance_n) and max(release)<tolerance_n and
        np.min(np.linalg.norm(measured,axis=1))>10*max(noise) else 'failed',
        response_sign=sign,vector_error_n=errors.tolist(),response_local_n=measured.tolist(),
        expected_local_n=expected.tolist(),release_residual_n=release.tolist(),baseline_noise_n=noise.tolist(),
        tolerance_n=tolerance_n)


def run(env):
    from isaacsim.core.prims import RigidPrim
    from scipy.spatial.transform import Rotation
    if not env.diagnostic_no_tool_collision:raise ValueError('Known-load test requires collision-free fixture')
    hand='left'
    link=RigidPrim(env.specs[hand]['root']+'/shaft',name='known_load_link',reset_xform_properties=False)
    link.initialize()
    if link.count!=1 or not link.is_physics_handle_valid():raise RuntimeError('Missing rigid link force handle')
    rows=[];responses=[];expected=[];release=[];noise=[];snapshots=[]
    robot=env.robots[hand]
    metadata=robot._articulation_view._metadata
    for axis in range(3):
        force=np.eye(3,dtype=np.float32)[axis]*.1
        phase=[]
        for tick in range(480):
            load=force if 120<=tick<360 else np.zeros(3,np.float32)
            link.apply_forces(load.reshape(1,3),is_global=True)
            env.sim.step(render=False)
            wrench=env.sensor(hand)[0]
            row=dict(axis=axis,tick=tick,applied_world_n=load.tolist(),wrench_local=wrench.tolist(),
                link_speed_m_s=float(np.linalg.norm(link.get_linear_velocities())),
                arm_effort_nm=np.asarray(env.robots[hand].get_measured_joint_efforts(joint_indices=env.indices[hand]),float).tolist())
            rows.append(row);phase.append(wrench[:3])
            if tick in (119,359,479):
                snapshots.append(dict(axis=axis,tick=tick,
                    all_incoming_wrenches=np.asarray(robot.get_measured_joint_forces(),float).tolist(),
                    link_pose=[np.asarray(v).tolist() for v in link.get_world_poses()],
                    arm_q=env.q(hand).tolist()))
            if tick==359:
                _,q=link.get_world_poses()
                R=Rotation.from_quat(np.asarray(q)[0,[1,2,3,0]]).as_matrix()
                expected.append((R.T@force).tolist())
        data=np.asarray(phase);baseline=data[88:120].mean(0)
        responses.append((data[328:360].mean(0)-baseline).tolist())
        release.append(float(np.linalg.norm(data[448:480].mean(0)-baseline)))
        noise.append(float(np.linalg.norm(data[88:120]-baseline,axis=1).max()))
        print(f'Known-load sensor axis {axis}: completed',flush=True)
    result=evaluate_response(responses,expected,release,noise)
    result.update(policy='known 0.1 N rigid-link loads on three world axes',
        acceptance_scope='simulated mount sensor only; collision-free fixture, not tissue-force validation',
        applied_force_n=.1,physics_ticks=len(rows),tissue_attachment_enabled=False,
        tool_collisions_enabled=False,material_calibrated=False,trace=rows,
        force_row=env.force_rows[hand],joint_indices=dict(metadata.joint_indices),
        joint_names=list(metadata.joint_names),snapshots=snapshots)
    (env.out/'known_load_sensor.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return {k:v for k,v in result.items() if k!='trace'}
