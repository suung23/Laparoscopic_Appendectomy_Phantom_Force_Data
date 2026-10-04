"""Isolate attachment activation; explicitly not an autonomous robot grasp."""
import numpy as np
from pxr import UsdGeom
from isaacsim.core.prims import SingleRigidPrim
from tissue_metrics import tetra_health


def run(env):
    c=env.config['data']['clusters'][1]
    body=SingleRigidPrim(prim_path=c['path'],name='attachment_diagnostic',reset_xform_properties=False)
    body.initialize()
    before=env.points().copy();center=before[c['indices']].mean(0)
    report=dict(policy='isolated attachment activation; no robot grasp claim',
        handle_initial_position=np.asarray(body.get_world_pose()[0]).tolist(),
        live_patch_center=center.tolist(),
        xform_ops=[op.GetOpName() for op in UsdGeom.Xformable(env.stage.GetPrimAtPath(c['path'])).GetOrderedXformOps()])
    env.grasp.capture('fixture',1,before,center,np.eye(3),evidence='diagnostic only')
    report['handle_position_after_usd_write']=np.asarray(body.get_world_pose()[0]).tolist()
    rows=[]
    for tick in range(8):
        env.sim.step(render=False)
        q=env.points();health=tetra_health(env.rest,q,env.tets)
        rows.append(dict(tick=tick+1,handle=np.asarray(body.get_world_pose()[0]).tolist(),
            node_jump_m=float(np.linalg.norm(q-before,axis=1).max()),**health))
        if health['inverted_tetrahedra']:break
    env.grasp.release('fixture')
    report['activation_trace']=rows
    report['status']='passed' if max(x['node_jump_m'] for x in rows)<.001 and not any(x['inverted_tetrahedra'] for x in rows) else 'failed'
    return report
