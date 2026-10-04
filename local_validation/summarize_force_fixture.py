"""Preserve differential-test evidence without treating isolated passes as task validation."""
import hashlib
import json
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRIALS=['force_fixture_cpu_v1','force_fixture_gpu_v1','force_fr3_cpu_v1',
        'force_fr3_gpu_v1','force_fr3_seed_gpu_v1','force_fr3_full_gpu_v1',
        'force_fr3_no_atlas_gpu_v1','force_fr3_full_external_gpu_v1','force_fr3_no_atlas_gpu_v2']
TRIALS += ['force_fr3_torque_full_v1','force_fr3_torque_full_v2','force_fr3_torque_no_atlas_v1']
TRIALS += ['force_spring_full_v1','force_spring_no_atlas_v1']
TRIALS += ['force_runtime_51_v1','force_runtime_61_v1','force_runtime_51_v2',
           'force_runtime_51_v3','force_runtime_61_v3','force_runtime_51_v4','force_runtime_61_v4']


def main():
    rows=[]
    for name in TRIALS:
        path=ROOT/'out'/name/'report.json'
        if not path.exists():continue
        raw=path.read_bytes();d=json.loads(raw)
        row={k:v for k,v in d.items() if k!='trace'}
        row.update(trial=name,report_sha256=hashlib.sha256(raw).hexdigest())
        if 'independent_torque_balance' in row:
            row['projected_torque_check']=row.pop('independent_torque_balance')
            row['projected_torque_check']['scope']='Analytical virtual-work reference versus projected joint-force API; not independent force sensing'
            row['label_correction']='Original report called this drive/independent torque. Installed API uses get_dof_projected_joint_forces, which shares the reaction pipeline.'
        if name.startswith('force_runtime_61_'):row['limitation']='Known-load reaction output only; live deformable response/readback and task transfer remain unverified'
        if name=='force_fr3_cpu_v1':row['limitation']='First axis began before initialization transient settled; later axes passed'
        if name=='force_fr3_no_atlas_gpu_v1':row['limitation']='Removed tissue left invalid attachment targets; needs clean repeat'
        if 'collisions' in row:
            row['robot_collisions']=row.pop('collisions')
            row['collision_scope_note']='Earlier label referred to robot colliders only'
        rows.append(row)
    report=dict(scope='Known-load rigid-link reaction diagnostics; not tissue-force calibration',
        tissue_mechanics_validated=False,task_force_sensor_validated=False,
        experiments=rows,
        interpretation='Simple fixture and isolated FR3 pass. Under 5.1 the full scene fails even at zero gravity. Under 6.1 v4 rigid reaction errors pass, but zero USD tissue displacement leaves deformable readback unresolved. Removing Atlas and its eight dependent attachments restores the measured response. External-force-every-iteration produces near-zero reported response and is not a remedy.',
        related_public_report={'url':'https://forums.developer.nvidia.com/t/get-measured-joint-forces-gives-wrong-values-if-gpu-is-used-instead-of-cpu/307612',
            'scope':'User-reported related GPU/particle-system issue in older releases; not confirmation of the cause or fix in this runtime'},
        remaining=['Separate actual dynamics from reaction reporting using a state-based known-load check or an independently instrumented solver',
            'Verify full task under gravity, tissue contact, multiple force magnitudes and poses',
            'Only then assess dynamic tissue attachment reaction and fit measured material data'])
    pair=[ROOT/'out'/n/'report.json' for n in ('force_spring_full_v1','force_spring_no_atlas_v1')]
    if all(p.exists() for p in pair):
        a,b=[json.loads(p.read_text())['trace'] for p in pair]
        if [(r['axis'],r['tick']) for r in a]!=[(r['axis'],r['tick']) for r in b]:raise ValueError('Unmatched state comparison')
        q=[np.array([r['joint_position_rad'] for r in rows]) for rows in (a,b)]
        report['spring_state_comparison']=dict(samples=len(a),max_joint_difference_rad=float(np.abs(q[0]-q[1]).max()),
            interpretation='Matched trajectories despite different reported forces. Both absolute spring-force checks failed; this does not establish calibrated dynamics.')
    (ROOT/'validation/force-fixture-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
