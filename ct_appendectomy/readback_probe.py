"""Diagnostic-only moving-anchor check for physics-only tissue observation freshness."""
import numpy as np


def run(env):
    from isaacsim.core.prims import SingleRigidPrim
    anchor=SingleRigidPrim('/World/BaseAnchor',name='readback_anchor',reset_xform_properties=False)
    anchor.initialize();p,q=anchor.get_world_pose()
    initial=env.points().copy()
    anchor.set_world_pose(position=np.asarray(p)+np.array([.0005,0,0]),orientation=q)
    rows=[]
    for sample in range(4):
        for _ in range(60):env.sim.step(render=False)
        before=env.points().copy();time=env.sim.current_time
        env.sim.render()
        after=env.points().copy()
        if env.sim.current_time!=time:raise RuntimeError('Render advanced physics clock')
        rows.append(dict(sample=sample,physics_time_s=float(time),
            render_only_coordinate_change_m=float(np.linalg.norm(after-before,axis=1).max()),
            pre_render_displacement_m=float(np.linalg.norm(before-initial,axis=1).max()),
            post_render_displacement_m=float(np.linalg.norm(after-initial,axis=1).max())))
    stale=max(r['render_only_coordinate_change_m'] for r in rows)>1e-7
    return dict(status='failed stale physics-only observation' if stale else 'no stale readback detected',
        scope='Diagnostic 0.5 mm anchor translation; not a surgical action or equilibrium validation',
        anchor_translation_m=.0005,trace=rows,physics_ticks=240)
