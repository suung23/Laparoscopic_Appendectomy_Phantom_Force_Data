"""Local bimanual workspace exploration with a separately actuated endoscope.

Coverage means two tool tips visiting workspace targets, not exposed organ area.
The reference planner is geometric and reactive, not a learned policy.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bimanual_env import BimanualEnv, HANDS

CAMERA_SPEED = np.array([15., 12., 20., .3, .3, .3])


class ExplorationEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, app, assets, max_steps=1800):
        self.base = BimanualEnv(app, assets, task='reach', max_steps=max_steps)
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1., 1., (16,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (42,), np.float32)
        self.render_mode = 'rgb_array'
        self.done = True

    def reset(self, *, seed=None, options=None):
        if options: raise ValueError('No exploration reset options in v1')
        super().reset(seed=seed)
        self.base.reset(seed=seed)
        from endoscope_control import EndoscopeControl
        from instrument_kinematics import PortInstrument
        self.scope = EndoscopeControl(self.base.control.stage, .004)
        self.camera = np.zeros(6)
        self.visited = np.zeros(7, bool)
        # A local survey around initial free-space poses. Full abdominal routing
        # requires organ/shaft collision geometry not yet present in this scene.
        offsets = np.array([[0,0,0],[-6,0,-3],[6,0,-3],[0,-5,-4],
                            [0,5,-4],[-4,-4,-5],[4,4,-5]], float)
        offsets[:, :2] *= np.pi/180
        offsets[:, 2] *= .001
        self.joint_goals = []
        self.goals = []
        for offset in offsets:
            positions=[]; joints=[]
            for j,h in enumerate(HANDS):
                original=self.base.control.arms[h]
                arm=PortInstrument(original.port, original.axes, original.limits)
                arm.state=original.state.copy()
                delta=offset.copy(); delta[0] *= 1 if j==0 else -1
                arm.state[:3]+=delta
                positions.append(arm.pose()['position_m']); joints.append(arm.state.copy())
            self.goals.append(positions); self.joint_goals.append(joints)
        self.goals=np.asarray(self.goals);self.joint_goals=np.asarray(self.joint_goals)
        self.index=0;self.hold=0;self.steps=0;self.done=False
        self.potential=self._potential()
        self.initial_positions=self.positions().copy()
        self.max_excursion=np.zeros(2)
        self.initial_camera=self.scope.pose.copy()
        self.camera_changed=False
        return self._observe(),self._info('')

    def positions(self):
        return np.array([self.base.control.arms[h].pose()['position_m'] for h in HANDS])

    def _potential(self):
        return -float(np.linalg.norm(self.goals[self.index]-self.positions(),axis=1).mean()/.03)

    def desired_camera(self):
        focus=self.goals[self.index].mean(0)
        local=self.scope.basis.T@(focus-self.scope.origin)
        pan=-np.rad2deg(np.arctan2(local[0],-local[2]))
        tilt=np.rad2deg(np.arctan2(local[1],np.linalg.norm(local[[0,2]])))
        return np.array([np.clip(pan,-30,30),np.clip(tilt,-25,25),0.,0.,0.,0.])

    def reference_action(self):
        from bimanual_tools import SPEED
        actions=[]
        for j,h in enumerate(HANDS):
            delta=self.joint_goals[self.index,j]-self.base.control.arms[h].state
            delta[3]=(delta[3]+np.pi)%(2*np.pi)-np.pi
            actions.extend(np.clip(delta/(SPEED/60),-.3,.3))
        camera=np.clip((self.desired_camera()-self.camera)/(CAMERA_SPEED/60),-1,1)
        return np.r_[actions,camera].astype(np.float32)

    def _observe(self,m=None):
        if m is None:m=self.base._metrics()
        onehot=np.eye(7)[self.index]
        health=[m['pin_drift_m']/.001,m['health']['minimum_signed_volume_ratio'],
                np.linalg.norm(self.base.control.points()-self.base.reference,axis=1).max()/.05]
        return np.r_[self.base.control.observation(),
            (self.goals[self.index]-self.positions()).ravel()/.03,
            self.camera/np.array([35,30,90,1,1,1]),self.visited.astype(float),onehot,
            1-self.steps/self.max_steps,health].astype(np.float32)

    def _info(self,failure):
        return dict(success=bool(self.visited.all() and not failure),failure=failure,
            coverage=float(self.visited.mean()),visited=self.visited.tolist(),
            target_index=int(self.index),tool_excursion_m=self.max_excursion.tolist(),
            camera_changed=self.camera_changed,coverage_type='paired tool-tip workspace targets',
            full_cavity_collision_validated=False, rosota_controller_conformant=False,
            scope_trocar_constrained=False,
            trocar={h:self.base.control.arms[h].trocar_metrics() for h in HANDS})

    def step(self,action):
        if self.done: raise RuntimeError('Reset before stepping')
        action=np.asarray(action,float)
        if action.shape!=(16,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('16 finite normalized actions required')
        candidate=self.camera+action[10:]*CAMERA_SPEED/60
        candidate=np.clip(candidate,[-35,-30,-90,-1,-1,-1],[35,30,90,1,1,1])
        radius=np.linalg.norm(candidate[3:])
        if radius>1:candidate[3:]/=radius
        self.base.control.step_action(action[:10],dt=1/60)
        self.camera=candidate
        self.scope.apply(*self.camera)
        self.base.sim.step(render=True)
        self.steps+=1;self.base.steps=self.steps
        m=self.base._metrics()
        failure=m['failure']
        self.max_excursion=np.maximum(self.max_excursion,np.linalg.norm(self.positions()-self.initial_positions,axis=1))
        self.camera_changed |= not np.allclose(self.scope.pose,self.initial_camera,atol=1e-6)
        reached=np.all(np.linalg.norm(self.goals[self.index]-self.positions(),axis=1)<.002)
        self.hold=self.hold+1 if reached and not failure else 0
        newly_visited=self.hold>=15 and not self.visited[self.index]
        if newly_visited:self.visited[self.index]=True
        term=bool(failure or self.visited.all())
        trunc=bool(self.steps>=self.max_steps and not term)
        previous_potential=self.potential
        if newly_visited and not term:
            # Choose the cheapest unvisited paired target from current state.
            candidates=np.flatnonzero(~self.visited)
            distances=np.linalg.norm(self.goals[candidates]-self.positions()[None],axis=2).sum(1)
            self.index=int(candidates[np.argmin(distances)]);self.hold=0
        self.potential=self._potential();self.done=term or trunc
        reward=.99*(0. if term else self.potential)-previous_potential-.002
        reward+=1. if newly_visited else 0.
        reward+=-5. if failure else (5. if self.visited.all() else 0.)
        return self._observe(m),float(reward),term,trunc,self._info(failure)

    def render(self):return self.base.render()
    def close(self):self.base.close();self.done=True
