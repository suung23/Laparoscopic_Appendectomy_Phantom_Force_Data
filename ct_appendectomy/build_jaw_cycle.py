"""Replace a recorded insertion hold with a prescribed jaw cycle, not grasp control."""
import argparse
from pathlib import Path
import numpy as np

def build(source,out,angle_deg=10.,speed=.6):
    data=dict(np.load(source));dt=float(data['dt']);poses=data['poses'];phases=data['phases']
    if not 0<angle_deg<=37 or not 0<speed<=.6:raise ValueError('Invalid legacy jaw command range')
    hold=np.flatnonzero(phases=='hold')
    if not len(hold) or not np.array_equal(hold,np.arange(hold[0],hold[-1]+1)):raise ValueError('One contiguous hold required')
    first,last=int(hold[0]),int(hold[-1])+1
    if not np.allclose(poses[hold],poses[first]):raise ValueError('Hold poses must be constant')
    angle=np.deg2rad(angle_deg);n=int(np.ceil(angle/(speed*dt)))+1
    data['poses']=np.concatenate([poses[:first],np.repeat(poses[first][None],2*n,axis=0),poses[last:]])
    data['jaw_half_angles']=np.r_[np.full(first,angle),np.linspace(angle,0,n),np.linspace(0,angle,n),np.full(len(poses)-last,angle)]
    data['phases']=np.r_[phases[:first],np.full(n,'jaw_close'),np.full(n,'jaw_open'),phases[last:]]
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**data)
    print({'poses':len(data['poses']),'maximum_half_angle_speed_rad_s':float(np.abs(np.diff(data['jaw_half_angles'])).max()/dt)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--angle-deg',type=float,default=10.);p.add_argument('--speed',type=float,default=.6)
    build(**vars(p.parse_args()))
