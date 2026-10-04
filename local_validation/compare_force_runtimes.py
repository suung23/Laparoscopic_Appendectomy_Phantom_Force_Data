"""Require matched input and implementation before attributing a force-test change to runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(baseline,candidate):
    keys=('source_sha256','helper_sha256','robot_scene_sha256','load_protocol','gpu',
          'gravity_m_s2','robot_collisions','surrounding_scene_retained',
          'excluded_component','external_forces_every_iteration','reset_seed','warmup_ticks')
    missing=[k for k in keys if k not in baseline or k not in candidate]
    if missing:raise ValueError('Missing comparison evidence: '+', '.join(missing))
    if any(baseline[k]!=candidate[k] for k in keys):raise ValueError('Unmatched experiment configuration or source')
    if not candidate['surrounding_scene_retained'] or candidate['excluded_component'] is not None:
        raise ValueError('Full surrounding scene must be retained')
    versions=[d.get('isaacsim_version') for d in (baseline,candidate)]
    if not all(versions) or versions[0]==versions[1]:raise ValueError('Two recorded distinct runtime versions required')
    for d in (baseline,candidate):
        errors=np.asarray(d.get('vector_error_n'),float)
        if errors.shape!=(3,) or not np.isfinite(errors).all():raise ValueError('Three finite force errors required')
        if d.get('physics_ticks')!=3*d['load_protocol']['end_tick']:raise ValueError('Incomplete force protocol')
        ticks=[(r['axis'],r['tick']) for r in d.get('trace',[])]
        if ticks!=[(axis,tick) for axis in range(3) for tick in range(d['load_protocol']['end_tick'])]:
            raise ValueError('Missing or noncontiguous per-step evidence')
    torque=candidate.get('projected_torque_check',{})
    release=np.asarray(candidate.get('release_residual_n'),float)
    force_ok=(release.shape==(3,) and np.isfinite(release).all() and
        np.max(release)<.005 and np.max(candidate['vector_error_n'])<.005)
    te,tt=np.asarray(torque.get('error_nm'),float),np.asarray(torque.get('tolerance_nm'),float)
    torque_ok=te.shape==(3,) and tt.shape==(3,) and np.isfinite([te,tt]).all() and np.all(tt>0) and np.all(te<tt)
    passed=bool(candidate.get('status')=='passed' and torque.get('status')=='passed' and force_ok and torque_ok)
    return dict(status='passed known-load comparison' if passed else 'candidate not validated',
        baseline_runtime=versions[0],candidate_runtime=versions[1],matched_source=True,
        baseline_error_n=baseline['vector_error_n'],candidate_error_n=candidate['vector_error_n'],
        prior_failure_resolved=bool(passed and baseline.get('status')=='failed'),
        scope='Matched zero-gravity known-load fixture only; task contact, tissue mechanics and clinical fidelity remain unvalidated')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=[x.read_bytes() for x in (a.baseline,a.candidate)]
    result=compare(*[json.loads(x) for x in raw])
    result['report_sha256']=[hashlib.sha256(x).hexdigest() for x in raw]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
