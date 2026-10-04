"""Compare captured tissue buffers without hiding node reorder or attachment drift."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def audit(directory):
    names=['before_initialize','after_initialize','after_settlement']
    arrays=[np.load(directory/('tissue_'+name+'.npz')) for name in names]
    base=arrays[0]
    rows=[]
    for name,data in zip(names,arrays):
        points=data['points'];same_shape=points.shape==base['points'].shape
        nearest,_=cKDTree(base['points']).query(points)
        rows.append(dict(phase=name,nodes=len(points),
            connectivity_unchanged=bool(np.array_equal(data['tets'],base['tets'])),
            indexed_displacement_m=float(np.linalg.norm(points-base['points'],axis=1).max()) if same_shape else None,
            nearest_original_distance_m=float(nearest.max()),
            pin_target_error_m=float(np.linalg.norm(points[data['pin']]-data['target_world'],axis=1).max()),
            anchor_displacement_m=float(np.linalg.norm(data['anchor_world']-base['anchor_world'])),
            capture_sha256=hashlib.sha256((directory/('tissue_'+name+'.npz')).read_bytes()).hexdigest()))
    return dict(phases=rows,scope='Buffer/attachment diagnosis, not material or contact validation')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.directory);a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))
