"""Preserve differential-test evidence without treating isolated passes as task validation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRIALS=['force_fixture_cpu_v1','force_fixture_gpu_v1','force_fr3_cpu_v1',
        'force_fr3_gpu_v1','force_fr3_seed_gpu_v1','force_fr3_full_gpu_v1',
        'force_fr3_no_atlas_gpu_v1','force_fr3_full_external_gpu_v1','force_fr3_no_atlas_gpu_v2']
TRIALS += ['force_fr3_torque_full_v1','force_fr3_torque_full_v2','force_fr3_torque_no_atlas_v1']


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
        if name=='force_fr3_cpu_v1':row['limitation']='First axis began before initialization transient settled; later axes passed'
        if name=='force_fr3_no_atlas_gpu_v1':row['limitation']='Removed tissue left invalid attachment targets; needs clean repeat'
        if 'collisions' in row:
            row['robot_collisions']=row.pop('collisions')
            row['collision_scope_note']='Earlier label referred to robot colliders only'
        rows.append(row)
    report=dict(scope='Known-load rigid-link reaction diagnostics; not tissue-force calibration',
        tissue_mechanics_validated=False,task_force_sensor_validated=False,
        experiments=rows,
        interpretation='Simple fixture and isolated FR3 pass. Full scene fails even at zero gravity. Removing Atlas and its eight dependent attachments restores the measured response. External-force-every-iteration produces near-zero reported response and is not a remedy.',
        related_public_report={'url':'https://forums.developer.nvidia.com/t/get-measured-joint-forces-gives-wrong-values-if-gpu-is-used-instead-of-cpu/307612',
            'scope':'User-reported related GPU/particle-system issue in older releases; not confirmation of the cause or fix in this runtime'},
        remaining=['Separate actual dynamics from reaction reporting using a state-based known-load check or an independently instrumented solver',
            'Verify full task under gravity, tissue contact, multiple force magnitudes and poses',
            'Only then assess dynamic tissue attachment reaction and fit measured material data'])
    (ROOT/'validation/force-fixture-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
