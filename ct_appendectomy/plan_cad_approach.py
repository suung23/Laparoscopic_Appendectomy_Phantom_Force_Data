"""Kinematic port-constrained CAD approach commissioning, not robot control."""
import json
from pathlib import Path
import numpy as np
import trimesh
from cad_penalty_contact import load_jaw,penalty
from instrument_kinematics import PortInstrument

def build(cad,layer_file,out):
    d=np.load(layer_file);tissue=trimesh.Trimesh(d['points'],d['faces'],process=False);local=load_jaw(cad)
    area=np.zeros(len(tissue.vertices));np.add.at(area,tissue.faces.ravel(),np.repeat(tissue.area_faces/3,3))
    index=710;target=tissue.vertices[index];forward=-tissue.vertex_normals[index];forward/=np.linalg.norm(forward)
    helper=np.eye(3)[np.argmin(np.abs(forward))];x=np.cross(helper,forward);x/=np.linalg.norm(x);y=np.cross(forward,x);axes=np.column_stack([x,y,forward])
    tip_z=local.vertices[:,2].max();start_tip=target-forward*.004;pivot=start_tip-forward*tip_z
    instrument=PortInstrument(pivot-forward*.1,axes);instrument.state[2]=.1
    history=[];poses=[]
    for step in range(61):
        pose=instrument.pose();jaw=local.copy();jaw.vertices=local.vertices@pose['axes'].T+pose['position_m']
        force,contact=penalty(jaw,tissue.vertices,area,1e5)
        metrics=instrument.trocar_metrics();record={'step':step,'insertion_m':pose['insertion_m'],'contact':contact,'trocar':metrics};history.append(record)
        transform=np.eye(4);transform[:3,:3]=pose['axes'];transform[:3,3]=pose['position_m'];poses.append(transform)
        if contact['active_nodes']>0:break
        instrument.step([0,0,1,0,0],.005)
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'approach.npz',poses=np.array(poses),jaw_points=local.vertices,jaw_faces=local.faces,port=instrument.port,port_axes=axes)
    report={'status':'contact onset found' if history[-1]['contact']['active_nodes'] else 'no sampled contact','initial_clear':history[0]['contact']['active_nodes']==0,'steps':history,'source_cad':str(cad),'source_layer':str(layer_file),'scope':'Kinematic pivot approach against frozen anatomy; nodal penetration detection only','controller':'PortInstrument legacy pivot, not Rosota FR3 TipController','port_anatomically_calibrated':False,'dynamics_integrated':False,'contact_stiffness_calibrated':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2));print(json.dumps(history[-1],indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cad',type=Path,required=True);p.add_argument('--layer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.cad,a.layer,a.out)
