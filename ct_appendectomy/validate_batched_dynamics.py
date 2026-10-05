"""Cross-check GPU independent solves against CPU scalar membrane integration."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from membrane_reference import MembraneReference
from membrane_dynamics import MembraneDynamics
from batched_membrane_dynamics import BatchedMembraneDynamics

def validate(surface,out):
    torch.set_num_threads(4);data=np.load(surface);p=data['points'];faces=data['faces']
    if p.shape!=(1625,3):raise ValueError('This fixture uses the 25x65 authored membrane')
    fixed=np.r_[np.arange(65),np.arange(len(p)-65,len(p))]
    loads=np.zeros((4,len(p),3));loads[:,710,2]=[.00005,.0001,.00015,.0002]
    model=MembraneReference(p,faces,12000,.45,.0002,1e-8)
    cpu=[];reports=[];start=time.perf_counter()
    for force in loads:
        dynamics=MembraneDynamics(model,p,1000,fixed)
        reports.append(dynamics.step(.005,force,maxiter=3000,tolerance=1e-7,defect_tolerance=1e-6))
        cpu.append(dynamics.points.copy())
    cpu_seconds=time.perf_counter()-start
    gpu_model=MembraneReference(p,faces,12000,.45,.0002,1e-8,device='cuda')
    batch=BatchedMembraneDynamics(gpu_model,p,1000,fixed,4)
    forces=torch.tensor(loads,dtype=torch.float64,device='cuda');torch.cuda.synchronize();start=time.perf_counter()
    report=batch.step(.005,forces,maxiter=3000,tolerance=1e-7,defect_tolerance=1e-6,independent=True)
    torch.cuda.synchronize();gpu_seconds=time.perf_counter()-start
    difference=np.linalg.norm(batch.points.cpu().numpy()-np.array(cpu),axis=2)
    result={'surface_sha256':hashlib.sha256(surface.read_bytes()).hexdigest(),'cpu_seconds':cpu_seconds,'gpu_seconds':gpu_seconds,
            'max_position_difference_m_per_environment':difference.max(axis=1).tolist(),
            'max_velocity_difference_m_s_per_environment':(difference.max(axis=1)/.005).tolist(),
            'cpu_residual_reports':reports,'gpu_residual_report':report,
            'fixed_point_error_m':float((batch.points[:,fixed]-torch.tensor(p[fixed],device='cuda')).abs().max()),
            'scope':'One implicit step for four synthetic loads; no contact trajectory, force calibration, or RL claim'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_residual_reports','gpu_residual_report')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--surface',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    validate(**vars(parser.parse_args()))
