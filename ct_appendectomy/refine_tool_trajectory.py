"""Subdivide prescribed tool command intervals without changing their end times."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def refine(source,out,factor=2):
    if factor<2:raise ValueError('Refinement factor must be at least two')
    data=dict(np.load(source));poses=data['poses'];dt=float(data['dt'])
    # Runtime applies command i over interval [i*dt,(i+1)*dt]. The first
    # command is rest; split subsequent transitions from the previous pose.
    result=[];angles=[]
    for i,current in enumerate(poses):
        previous=poses[max(i-1,0)]
        rotations=Slerp([0.,1.],Rotation.from_matrix(np.stack([previous[:3,:3],current[:3,:3]])))
        for sub in range(1,factor+1):
            t=sub/factor;pose=np.eye(4)
            pose[:3,:3]=rotations(t).as_matrix();pose[:3,3]=(1-t)*previous[:3,3]+t*current[:3,3]
            result.append(pose)
            if 'jaw_half_angles' in data:
                a=data['jaw_half_angles'];angles.append((1-t)*a[max(i-1,0)]+t*a[i])
    data['poses']=np.array(result);data['dt']=dt/factor
    if angles:data['jaw_half_angles']=np.array(angles)
    if 'phases' in data:data['phases']=np.repeat(data['phases'],factor)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**data)
    np.testing.assert_allclose(data['poses'][factor-1::factor],poses,atol=1e-12)
    print({'steps':len(result),'dt':float(data['dt']),'duration_seconds':len(result)*float(data['dt'])})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--factor',type=int,default=2)
    refine(**vars(p.parse_args()))
