"""Compare two captured tissue trajectories without spatial fitting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from node_correspondence import node_permutation
from tissue_metrics import tetra_health


def compare(reference,candidate):
    rows=[]
    for phase in ['before_initialize','after_initialize','warmup_1','warmup_60','warmup_480','after_settlement']:
        paths=[directory/f'tissue_{phase}.npz' for directory in (reference,candidate)]
        if not all(p.exists() for p in paths):continue
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            permutation=node_permutation(a['rest'],a['tets'],b['rest'],b['tets'])
            delta=np.linalg.norm(a['points']-b['points'][permutation],axis=1)
            rows.append(dict(phase=phase,exact_rest_geometry_and_topology=True,
                max_displacement_difference_m=float(delta.max()),rms_displacement_difference_m=float(np.sqrt(np.mean(delta**2))),
                reference_health=tetra_health(a['rest'],a['points'],a['tets']),
                candidate_health=tetra_health(b['rest'],b['points'],b['tets']),
                input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]))
    return dict(status='comparison_complete' if any(r['phase']=='after_settlement' for r in rows) else 'incomplete',
        phases=rows,scope='Trajectory comparison, not an acceptance pass. Requires independently verified matched experimental inputs and timing.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(a.reference,a.candidate);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
    raise SystemExit(0 if result['status']=='comparison_complete' else 1)
