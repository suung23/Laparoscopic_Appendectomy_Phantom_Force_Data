"""Retain unsuccessful load-path experiments without promoting them to validation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS={
    'solver_coupling_v1':'Constraint-local point edits with fixed robot; runtime update behavior unverified',
    'solver_coupling_v2':'1 mm world-Y robot motion; fixed-duration reset was not verified settled',
    'solver_coupling_v3':'2 mm axial motion; directions recomputed per reset, so conditions were not exactly matched',
    'solver_coupling_v4':'Matched direction and adaptive settlement; stopped at failed reset motion gate',
    'mount_sensor_v1':'Known 0.1 N rigid-link force on/off without tissue attachments',
    'mount_sensor_v2':'Three-axis 0.1 N loads; robot collisions disabled in diagnostic only',
    'mount_sensor_v3':'Repeat three-axis loads with all incoming joint wrenches and poses recorded'}


def summarize():
    rows=[]
    for name,method in METHODS.items():
        source=ROOT/'out/ct_appendectomy'/name/'report.json'
        if not source.exists():continue
        raw=source.read_bytes();d=json.loads(raw)
        row={'trial':name,'method':method,'status':d['status'],'error':d.get('error'),
             'report_sha256':hashlib.sha256(raw).hexdigest(),
             'implementation_sha256':d.get('implementation_sha256')}
        for key in ('mount_force_effect_n','activation_differential_noise_n',
                    'measured_response_n','released_residual_n','baseline_noise_n',
                    'vector_error_n','release_residual_n','tolerance_n','force_row','joint_indices'):
            if key in d:row[key]=d[key]
        if 'conditions' in d:
            row['conditions']={k:{key:v.get(key) for key in ('activation_max_displacement_m',
                'tool_displacement_m','commanded_displacement_m','failure')} for k,v in d['conditions'].items()}
        rows.append(row)
    report={'scope':'Engineering load path diagnostics, not tissue material calibration',
            'two_way_reaction_validated':False,'frictional_grasp_validated':False,
            'trials':rows,'remaining':['Matched settled trials with repeatability control',
                'Known-load sensor sensitivity and coordinate-frame check',
                'Independent force balance/drive-torque verification',
                'Full bimanual contact and release under validated coupling']}
    (ROOT/'validation/solver-coupling.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':summarize()
