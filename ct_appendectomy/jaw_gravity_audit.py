"""Read-only static jaw load audit. No controller compensation is applied."""
import numpy as np


def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu()
    return np.asarray(value.numpy() if hasattr(value,'numpy') else value)


class JawGravityModel:
    """Pose-dependent static model; does not compensate inertia or contact."""
    def __init__(self,env):
        from isaacsim.core.prims import SingleRigidPrim
        self.gravity=np.zeros(3) if env.diagnostic_zero_gravity else np.array([0.,-9.81,0.])
        self.bodies={}
        for hand,spec in env.specs.items():
            view=env.robots[hand]._articulation_view
            masses=_array(view.get_body_masses())[0]
            coms=_array(view.get_body_coms()[0])[0]
            self.bodies[hand]=[]
            for name in ('left','right'):
                i=view.body_names.index(name)
                body=SingleRigidPrim(prim_path=spec['root']+'/'+name,
                    name=hand+'_'+name+'_gravity_model',reset_xform_properties=False)
                body.initialize()
                self.bodies[hand].append((body,float(masses[i]),coms[i].copy()))

    def predict(self,hand):
        from scipy.spatial.transform import Rotation
        values=[]
        for body,mass,com in self.bodies[hand]:
            _,q=body.get_world_pose()
            R=Rotation.from_quat(_array(q)[[1,2,3,0]]).as_matrix()
            values.append(support_torque(mass,com,R,self.gravity))
        return np.asarray(values)


def support_torque(mass, com_local, rotation, gravity_world):
    """Support torque about child-frame +Y hinge; COM offset from hinge origin."""
    return float(-np.cross(np.asarray(com_local,float),
                          np.asarray(rotation,float).T @ (mass*np.asarray(gravity_world,float)))[1])


def audit(env):
    from isaacsim.core.prims import SingleRigidPrim
    from scipy.spatial.transform import Rotation
    gravity=np.zeros(3) if env.diagnostic_zero_gravity else np.array([0.,-9.81,0.])
    result=dict(scope='static load model diagnostic; no contact-force validation',hands={})
    for hand,spec in env.specs.items():
        view=env.robots[hand]._articulation_view
        masses=_array(view.get_body_masses())[0]
        coms=_array(view.get_body_coms()[0])[0]
        measured=env.sensor(hand)[1]
        records=[]
        for j,name in enumerate(('left','right')):
            index=view.body_names.index(name)
            body=SingleRigidPrim(prim_path=spec['root']+'/'+name,
                                name=hand+'_'+name+'_gravity_audit',reset_xform_properties=False)
            body.initialize()
            _,q=body.get_world_pose()
            R=Rotation.from_quat(_array(q)[[1,2,3,0]]).as_matrix()
            expected=support_torque(float(masses[index]),coms[index],R,gravity)
            records.append(dict(body=name,mass_kg=float(masses[index]),
                com_body_m=coms[index].tolist(),body_rotation_world=R.tolist(),
                expected_support_torque_nm=expected,measured_effort_nm=float(measured[j]),
                residual_nm=float(measured[j]-expected)))
        result['hands'][hand]=records
    result['assumptions']=['Jaw child joint origin equals rigid body origin; child joint rotation identity; axis +Y.',
        'Static gravity only; no inertial, drive transient, joint limit, or contact compensation.',
        'Mass and COM are simulator parameters, not independently measured physical instrument properties.']
    return result
