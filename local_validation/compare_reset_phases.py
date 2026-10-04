"""Locate first physical divergence between two captured reset sequences."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from node_correspondence import node_permutation


def compare(directory):
    rows=[]
    for phase in ['before_initialize','after_cooking','after_initialize','before_warmup','warmup_1','warmup_60','warmup_480','after_settlement']:
        paths=[directory/f'reset_{i}_tissue_{phase}.npz' for i in (1,2)]
        if not all(p.exists() for p in paths):continue
        a,b=[np.load(p) for p in paths]
        permutation=node_permutation(a['rest'],a['tets'],b['rest'],b['tets'])
        distance=np.linalg.norm(a['points']-b['points'][permutation],axis=1)
        rows.append(dict(phase=phase,max_displacement_m=float(distance.max()),
            rms_displacement_m=float(np.sqrt(np.mean(distance**2))),
            reordered_nodes=int(np.sum(permutation!=np.arange(len(permutation)))),
            source_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]))
    return dict(phases=rows,scope='Exact material-node comparison; no fitted spatial alignment')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(a.directory);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))
