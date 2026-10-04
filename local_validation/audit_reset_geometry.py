"""Offline audit of captured reset phases; solver-independent signed volumes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from tissue_metrics import tetra_health


def audit(directory):
    phases={}
    for phase in ['after_initialize','warmup_1','warmup_60','warmup_480','after_settlement']:
        path=directory/f'tissue_{phase}.npz'
        if not path.exists():continue
        with np.load(path) as data:
            points,rest,tets=data['points'],data['rest'],data['tets']
            health=tetra_health(rest,points,tets)
            displacement=points-rest
            translation=displacement.mean(axis=0)
            residual=np.linalg.norm(displacement-translation,axis=1)
            phases[phase]=dict(health=health,mean_nodal_translation_m=translation.tolist(),
                translation_removed_rms_m=float(np.sqrt(np.mean(residual**2))),
                translation_removed_max_m=float(residual.max()),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(phases=phases,scope='Captured geometry audit. Translation removal is not a fitted rigid rotation or physical validation. Environment failures remain failures.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.directory)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
