"""Preserve failed and completed CT trials without promoting them to parity."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    trials=[]
    fields=['status','failure','error','task_failure','steps','physics_ticks_checked',
        'phase_ends','grasp_events','peak_held_hands','pull_deformation_rms_m',
        'post_release_residual_rms_m','post_release_held_reference_rms_m',
        'loaded_free_reference_rms_m','recovery_reference','withdraw_samples',
        'recovery_fraction','grasp_geometry','contact_summary']
    paths=sorted((ROOT/'out/ct_appendectomy').glob('rosota_baseline_grasp_v*/report.json'),
                 key=lambda p:int(p.parent.name.rsplit('v',1)[1]))
    for path in paths:
        raw=json.loads(path.read_text())
        item={'run':path.parent.name,**{k:raw[k] for k in fields if k in raw}}
        if item['status'] in ('starting','running'):
            item['status']='incomplete'
        if 'recovery_fraction' in item and 'recovery_reference' not in item:
            item['recovery_reference']='legacy pre-pull held shape; does not isolate unloaded recovery'
        trace=path.parent/'physics_trace.json'
        if trace.exists():
            rows=json.loads(trace.read_text())
            item['maximum_inverted_tetrahedra']=max(x['tissue_health']['inverted_tetrahedra'] for x in rows)
            item['minimum_volume_ratio_all_ticks']=min(x['tissue_health']['minimum_signed_volume_ratio'] for x in rows)
            item['max_pin_drift_m']=max(x['pin_drift_m'] for x in rows)
        trials.append(item)
    activations=[]
    for path in sorted((ROOT/'out/ct_appendectomy').glob('attachment_activation_v*/report.json')):
        r=json.loads(path.read_text());rows=r.get('activation_trace',[])
        activations.append(dict(run=path.parent.name,status=r['status'],ticks=len(rows),
            maximum_node_change_m=max((x['node_jump_m'] for x in rows),default=None),
            maximum_inverted_tetrahedra=max((x['inverted_tetrahedra'] for x in rows),default=None)))
    names=['fr3_ct_scene.py','rosota_ct_env.py','run_rosota_ct.py','tissue_grasp.py',
           'deformation_metrics.py','attachment_probe.py','contact_control.py','rosota_controller.py']
    hashes={name:hashlib.sha256((ROOT/'ct_appendectomy'/name).read_bytes()).hexdigest() for name in names}
    report=dict(scope='CT engineering grasp and deformation trials; native reference is evaluated separately',
        current_source_sha256=hashes,trials=trials,isolated_attachment_trials=activations,
        latest_trial_passed=bool(trials and trials[-1]['status']=='passed'),
        full_rosota_parity=False,material_calibrated=False,matched_unloaded_control=False,
        friction_only_grasp=False,clinical_validation=False,
        limitations=['Right tool remains a grasper in CT mode.',
            'CT camera remains occluded; native rabbit reference has the restored endoscope view.',
            'Grasp uses explicit rigid patch constraints, not friction-only tissue contact.',
            'The kinematic patch follower does not feed its attachment reaction back to robot joints.',
            'No physical tissue severing, mesoappendix or bleeding model.',
            'An incomplete/interrupted trial is not a successful trial.'])
    (ROOT/'validation/ct-grasp-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'trials':[(x['run'],x['status']) for x in trials],
        'latest_trial_passed':report['latest_trial_passed']}))


if __name__=='__main__':main()
