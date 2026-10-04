"""Known local-Y torque control for gravity-compensated jaw sensing.

This isolates the sensor/model; it is not a tissue-contact experiment.
"""
import json
import numpy as np


def run(env):
    from isaacsim.core.prims import RigidPrim
    if not env.diagnostic_no_tool_collision or not env.jaw_gravity_compensation:
        raise ValueError('Requires collision-disabled fixture and gravity compensation')
    rows=[];checks=[]
    for hand in env.specs:
        for index,name in enumerate(('left','right')):
            body=RigidPrim(env.specs[hand]['root']+'/'+name,
                name=hand+'_'+name+'_known_torque',reset_xform_properties=False)
            body.initialize()
            if body.count!=1 or not body.is_physics_handle_valid():raise RuntimeError('Invalid jaw load handle')
            sign=1 if env.jaw_gravity_model.predict(hand)[index]>=0 else -1
            for magnitude in (.0001,.0004):
                trial=[]
                for tick in range(480):
                    load=sign*magnitude if 120<=tick<360 else 0.
                    body.apply_forces_and_torques_at_pos(
                        torques=np.array([[0,load,0]],np.float32),is_global=False)
                    env.sim.step(render=False)
                    raw=env.sensor(hand)[1][index]
                    gravity=env.jaw_gravity_model.predict(hand)[index]
                    corrected=float(raw-gravity)
                    row=dict(hand=hand,jaw=name,magnitude_nm=magnitude,tick=tick,
                        applied_local_y_nm=load,raw_nm=float(raw),gravity_nm=float(gravity),corrected_nm=corrected)
                    rows.append(row);trial.append(corrected)
                baseline=float(np.mean(trial[88:120]))
                response=float(np.mean(trial[328:360]))-baseline
                release=float(np.mean(trial[448:480]))-baseline
                # A positive applied torque requires a negative support torque.
                error=abs(response+sign*magnitude)
                checks.append(dict(hand=hand,jaw=name,magnitude_nm=magnitude,
                    response_nm=response,expected_nm=-sign*magnitude,error_nm=error,
                    release_nm=release,passed=bool(error<.00002 and abs(release)<.00002),
                    loaded_above_guard=bool(abs(np.mean(trial[328:360]))>=env.guards[hand].jaw_limit)))
    result=dict(status='passed' if all(x['passed'] and x['loaded_above_guard']==(x['magnitude_nm']>.0003) for x in checks) else 'failed',
        scope='known simulated torque response; not tissue force or complete controller validation',
        tolerance_nm=.00002,checks=checks,trace=rows)
    (env.out/'jaw_known_load.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return {k:v for k,v in result.items() if k!='trace'}
