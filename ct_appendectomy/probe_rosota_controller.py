"""Local kinematic replay of the original FR3 controller, not a physics demo."""
import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from rosota_controller import RosotaArmController, aperture_metrics, check_chunk, STRICT_CFG
from laparo_ik import Chain

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'newton_cell'))
from cell_spec import constants


def build_chain(source, hand, output):
    """Kinematics-only fixture: exact installed FR3 joints + Rosota CAD tail."""
    src=ET.parse(source).getroot(); root=ET.Element('robot',name='kinematics_probe')
    for i in range(1,9):
        j=src.find(f"joint[@name='fr3_joint{i}']")
        if j is None: raise ValueError('Expected FR3 joint chain')
        root.append(copy.deepcopy(j))
    c=constants();prefix='LAPARO' if hand=='left' else 'CUTTER'
    for name,xyz,rpy in [('fr3_link8_to_tool',(0,0,0),(0,0,0)),
                         ('tool_flange_to_tip',c[prefix+'_TIP_XYZ'],c[prefix+'_TIP_RPY'])]:
        j=ET.SubElement(root,'joint',name=name,type='fixed')
        ET.SubElement(j,'origin',xyz=' '.join(map(str,xyz)),rpy=' '.join(map(str,rpy)))
    ET.ElementTree(root).write(output,encoding='utf-8')
    return Chain(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fr3-urdf',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'out/rosota_controller')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','scope':'kinematic joint follower; no tissue, no articulation dynamics',
        'ct_scene_integrated':False,'controller':STRICT_CFG.dict(),
        'fr3_urdf_sha256':hashlib.sha256(a.fr3_urdf.read_bytes()).hexdigest(),'arms':{}}
    try:
        for hand in ('left','right'):
            chain=build_chain(a.fr3_urdf,hand,a.out/(hand+'.urdf'))
            q=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
            if hand=='right': q*=np.array([-1,1,-1,1,-1,1,-1])
            tip,flange=chain.fk(q);axis=-(flange@chain.R_f2t)[:,2]
            port=tip-axis*.12
            ctl=RosotaArmController(chain,np.eye(4),port,axis,q,hand=hand)
            trace=[];initial_q=q.copy();initial_tip=tip.copy()
            for sample in range(30):
                row=np.array([0.,0.,-.00005,.01,-.01,.002,70.])
                ctl.begin_substep(row,q)
                for _ in range(16):
                    result=ctl.tick(q)
                    # Explicit ideal plant. Never label its joint target as measured hardware.
                    q=result['joint_target'].copy()
                    tip,R,inward=ctl.pose(q)
                    m=aperture_metrics(port,axis,tip,inward)
                    if not m['aperture_clear']:raise AssertionError('Post-tick aperture violation')
                    trace.append(dict(**m,q=q.tolist(),roll_backlog_rad=result['roll_backlog_rad']))
            insertion_delta=float(np.dot(tip-initial_tip,axis))
            roll_delivery=float(-(q[6]-initial_q[6]))
            assert insertion_delta>0, 'Tool -z must insert'
            assert roll_delivery>0, 'Positive tip rz must decrease FR3 joint 7'
            assert abs(ctl.roll_cmd-ctl.roll_put)<.02, 'Roll debt did not settle'
            assert result['gripper_target_deg']==(.7*(74 if hand=='left' else 50))
            np.testing.assert_allclose(result['rot_rejected'],[.01/16,-.01/16])
            frozen=ctl.p_cmd.copy()
            with_bad=np.zeros((2,8,7));with_bad[1,7,0]=np.nan
            try:check_chunk(with_bad)
            except ValueError:pass
            else:raise AssertionError('Invalid chunk accepted')
            np.testing.assert_array_equal(frozen,ctl.p_cmd)
            report['arms'][hand]=dict(ticks=len(trace),insertion_delta_m=insertion_delta,
                positive_roll_delivery_rad=roll_delivery,
                max_bite_m=max(x['bite_m'] for x in trace),
                max_offset_m=max(x['offset_m'] for x in trace),
                roll_backlog_rad=ctl.roll_cmd-ctl.roll_put)
            # Large outward-sign roll near the joint stop must not create an
            # unbounded debt; reversal must command the joint away immediately.
            q=initial_q.copy();q[6]=chain.lo[6]+.002
            p0,R0=chain.fk(q);a0=-(R0@chain.R_f2t)[:,2]
            sat=RosotaArmController(chain,np.eye(4),p0-a0*.12,a0,q,hand=hand)
            for turn in (1.,-1.):
                sat.begin_substep([0,0,0,0,0,turn,100],q)
                before_reverse=q[6]
                for _ in range(16):q=sat.tick(q)['joint_target'].copy()
                assert abs(sat.roll_cmd-sat.roll_put)<.9, 'Unbounded roll debt at stop'
                if turn<0:assert q[6]>before_reverse+.01, 'Roll reversal blocked by stale debt'
            report['arms'][hand]['roll_stop_reversal_passed']=True
            (a.out/(hand+'_trace.json')).write_text(json.dumps(trace),encoding='utf-8')
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['arms'],indent=2))


if __name__=='__main__':main()
