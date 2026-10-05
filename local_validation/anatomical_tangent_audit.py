"""Validate tangent at the actual rejected dynamic-contact iterate."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference

folder=root/'out/ct_appendectomy/anatomical_implicit_contact_stable_energy'
state=np.load(folder/'last_accepted_state.npz')
candidate=folder/'unaccepted_iterate.npz'
x=np.load(candidate)['points']
model=ElasticReference(state['rest'],state['tets'],12000.,.45)
rng=np.random.default_rng(119)
a,b=rng.normal(size=(2,*x.shape))
a/=np.linalg.norm(a); b/=np.linalg.norm(b)
ha,hb=model.hessian_vector(x,a),model.hessian_vector(x,b)
checks=[]
for step in (1e-7,1e-8,1e-9):
    difference=(model.energy_gradient(x+step*a,x*0)[1]-model.energy_gradient(x-step*a,x*0)[1])/(2*step)
    checks.append(dict(step_m=step,relative_l2_error=float(np.linalg.norm(difference-ha)/np.linalg.norm(ha))))
symmetry=abs(float(np.sum(a*hb)-np.sum(b*ha)))
report=dict(status='passed' if max(c['relative_l2_error'] for c in checks)<1e-5 and symmetry<1e-9 else 'failed',
    scope='Exact elastic tangent consistency at rejected iterate; not a successful implicit step or physical calibration',
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((root/'ct_appendectomy/constrained_elasticity.py').read_bytes()).hexdigest(),
    derivative_checks=checks,bilinear_symmetry_error=symmetry)
(root/'validation/anatomical-tangent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
