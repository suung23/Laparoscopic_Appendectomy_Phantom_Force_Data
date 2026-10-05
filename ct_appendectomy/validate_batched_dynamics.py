"""Cross-check GPU independent solves against CPU scalar membrane integration."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from membrane_reference import MembraneReference
from membrane_dynamics import MembraneDynamics
from batched_membrane_dynamics import BatchedMembraneDynamics

def validate(surface,out,steps=1):
    if steps<1:raise ValueError('Positive step count required')
    torch.set_num_threads(4);data=np.load(surface);p=data['points'];faces=data['faces']
    if p.shape!=(1625,3):raise ValueError('This fixture uses the 25x65 authored membrane')
    fixed=np.r_[np.arange(65),np.arange(len(p)-65,len(p))]
    loads=np.zeros((4,len(p),3));loads[:,710,2]=[.00005,.0001,.00015,.0002]
    model=MembraneReference(p,faces,12000,.45,.0002,1e-8)
    cpu=[];cpu_velocities=[];reports=[];start=time.perf_counter()
    amplitudes=[1. if i<max(1,steps//2) else 0. for i in range(steps)]
    for force in loads:
        dynamics=MembraneDynamics(model,p,1000,fixed)
        history=[];velocities=[];environment_reports=[]
        for amplitude in amplitudes:
            environment_reports.append(dynamics.step(.005,force*amplitude,maxiter=3000,tolerance=1e-7,defect_tolerance=1e-6))
            history.append(dynamics.points.copy());velocities.append(dynamics.velocity.copy())
        reports.append(environment_reports);cpu.append(history);cpu_velocities.append(velocities)
    cpu_seconds=time.perf_counter()-start
    gpu_model=MembraneReference(p,faces,12000,.45,.0002,1e-8,device='cuda')
    batch=BatchedMembraneDynamics(gpu_model,p,1000,fixed,4)
    forces=torch.tensor(loads,dtype=torch.float64,device='cuda');torch.cuda.synchronize();start=time.perf_counter()
    report=[];gpu_points=[];gpu_velocities=[]
    for amplitude in amplitudes:
        report.append(batch.step(.005,forces*amplitude,maxiter=3000,tolerance=1e-7,defect_tolerance=1e-6,independent=True))
        gpu_points.append(batch.points.detach().clone());gpu_velocities.append(batch.velocity.detach().clone())
    torch.cuda.synchronize();gpu_seconds=time.perf_counter()-start
    difference=np.linalg.norm(torch.stack(gpu_points,dim=1).cpu().numpy()-np.array(cpu),axis=3)
    velocity_difference=np.linalg.norm(torch.stack(gpu_velocities,dim=1).cpu().numpy()-np.array(cpu_velocities),axis=3)
    result={'surface_sha256':hashlib.sha256(surface.read_bytes()).hexdigest(),'cpu_seconds':cpu_seconds,'gpu_seconds':gpu_seconds,
            'steps':steps,'load_amplitudes':amplitudes,
            'max_position_difference_m_per_environment':difference.max(axis=(1,2)).tolist(),
            'max_velocity_difference_m_s_per_environment':velocity_difference.max(axis=(1,2)).tolist(),
            'cpu_residual_reports':reports,'gpu_residual_report':report,
            'fixed_point_error_m':float((batch.points[:,fixed]-torch.tensor(p[fixed],device='cuda')).abs().max()),
            'scope':'Implicit steps for four synthetic load/release histories; no contact trajectory, force calibration, or RL claim'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_residual_reports','gpu_residual_report')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--surface',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1)
    validate(**vars(parser.parse_args()))
