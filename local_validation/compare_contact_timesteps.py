"""Self-convergence at a shared terminal time, no experimental ground truth."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
folders=[root/'out/ct_appendectomy'/name for name in (
    'anatomical_implicit_contact_240hz',
    'anatomical_implicit_contact_early_newton',
    'anatomical_implicit_contact_960hz')]
reports=[json.loads((p/'report.json').read_text()) for p in folders]
if any(r['status']!='passed' or len(r['rows'])!=r['requested_steps'] for r in reports):
    raise RuntimeError('All runs must finish and pass their step audits')
times=[r['dt_s']*r['requested_steps'] for r in reports]
if not np.allclose(times,times[0],rtol=0,atol=1e-15): raise RuntimeError('Terminal time mismatch')
if any(r['mesh_sha256']!=reports[0]['mesh_sha256'] or r['parameters']!=reports[0]['parameters'] for r in reports):
    raise RuntimeError('Mesh or parameters differ')
states=[np.load(p/'last_accepted_state.npz') for p in folders]
result=dict(scope='Three-resolution self-convergence on a single short plane-contact trajectory, not temporal accuracy certification or empirical tissue validation',
    terminal_time_s=times[0],dt_s=[r['dt_s'] for r in reports],
    source_versions=[r['source_sha256'] for r in reports],
    implementation_note='480 Hz predates objective/tangent caching; prior checks found unchanged cache results. No mechanics parameters changed.',comparisons={})
for field in ('points','velocity'):
    differences=[float(np.sqrt(np.mean(np.sum((states[i][field]-states[i+1][field])**2,axis=1)))) for i in (0,1)]
    result['comparisons'][field]=dict(rms_successive_differences=differences,
        units='m' if field=='points' else 'm/s',decreasing=differences[1]<differences[0],
        observed_order=float(np.log2(differences[0]/differences[1])) if min(differences)>0 else None)
(root/'validation/contact-time-refinement.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['comparisons'],indent=2))
