"""Validate measured closure/reopening, separately from scene survival.

Engineering commissioning tolerances; not evidence of cutting or tissue force.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(report, trace):
    if report.get('approach_samples') != 0:
        raise ValueError('Requires a no-approach jaw cycle')
    ratio = report['physics_hz'] / report['control_hz']
    if ratio != int(ratio):
        raise ValueError('Nonintegral physics/control ratio')
    boundary = int(report['close_samples'] * ratio)
    if len(trace) <= boundary:
        raise ValueError('Missing reopening phase')
    result = dict(scope='measured aperture commissioning, not contact/cutting validation',
                  robot_collisions_disabled=bool(report.get('diagnostic_no_tool_collision',False)),
                  gravity_compensation_enabled=bool(report.get('jaw_gravity_compensation',False)),
                  closure_tolerance_deg=2., opening_tolerance_deg=2., hands={})
    for hand, limit in [('left', 74.), ('right', 50. if report['right_tool']=='cutter' else 74.)]:
        q = np.asarray([x['jaw_position_rad'][hand] for x in trace], float)
        if q.shape != (len(trace), 2) or not np.isfinite(q).all():
            raise ValueError('Invalid measured jaw positions')
        aperture = np.rad2deg(q[:, 0]-q[:, 1])
        closed = float(np.median(aperture[max(0,boundary-16):boundary]))
        reopened = float(np.median(aperture[-16:]))
        limits_ok = bool(np.all(np.abs(np.rad2deg(q)) <= limit/2+2))
        result['hands'][hand] = dict(closed_aperture_deg=closed,
            reopened_aperture_deg=reopened, maximum_opening_deg=limit,
            measured_range_deg=[float(aperture.min()),float(aperture.max())],
            limits_ok=limits_ok,
            closure_guard_ticks=sum(bool(x['contact'][hand]['jaw_limited']) for x in trace[:boundary]),
            passed=bool(abs(closed)<=2 and abs(reopened-limit)<=2 and limits_ok))
        if result['gravity_compensation_enabled']:
            contact=[x['contact'][hand] for x in trace]
            if not all(x.get('jaw_gravity_compensated') for x in contact):
                raise ValueError('Compensated run has missing compensation telemetry')
            raw=np.asarray([x['jaw_effort_nm'] for x in contact],float)
            gravity=np.asarray([x['jaw_gravity_nm'] for x in contact],float)
            corrected=np.asarray([x['jaw_corrected_effort_nm'] for x in contact],float)
            if not np.isfinite([raw,gravity,corrected]).all() or not np.allclose(raw-gravity,corrected,atol=1e-12,rtol=0):
                raise ValueError('Invalid raw/gravity/corrected effort correspondence')
            result['hands'][hand]['effort_audit']=dict(
                peak_raw_abs_nm=np.max(np.abs(raw),axis=0).tolist(),
                peak_corrected_abs_nm=np.max(np.abs(corrected),axis=0).tolist(),
                p99_corrected_abs_nm=np.percentile(np.abs(corrected),99,axis=0).tolist(),
                gravity_prediction_span_nm=np.ptp(gravity,axis=0).tolist())
    result['passed'] = report['status']=='passed' and all(h['passed'] for h in result['hands'].values())
    result['limitations'] = ['No-approach does not prove collision-free motion.',
                            'Joint effort is not an independent contact sensor.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=audit(json.loads((a.run/'report.json').read_text()),
                 json.loads((a.run/'physics_trace.json').read_text()))
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
