"""Classify captured engine errors without copying private log contents."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(path):
    counts=Counter()
    with path.open(encoding='utf-8',errors='replace') as stream:
        for line in stream:
            if 'PhysX error:' in line:
                if 'updateNodeInverseVolumes()' in line:
                    counts['small_degenerate_or_inverted_tetrahedron']+=1
                else:
                    counts['other_physx_error']+=1
            elif '[Error]' in line and '[omni.physx' in line:
                counts['other_physx_plugin_error']+=1
    return dict(status='failed' if counts else 'no_physx_errors_observed',
        counts=dict(counts),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Log-only gate, never a mechanics pass. Small-element messages do not distinguish positive tiny cells from inverted cells; inspect stored geometry separately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.log)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(1 if result['status']=='failed' else 0)
