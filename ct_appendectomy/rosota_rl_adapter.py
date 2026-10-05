"""Normalized PPO interface for the two-arm, trocar-constrained Rosota CT env.

This is a privileged-state commissioning interface for safe reaching and scope
exploration. The wrapped environment's current reward is a reach objective; it
does not define successful grasp, dissection, or appendectomy.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


ACTION_DIM = 14
OBSERVATION_DIM = 53
TRACTION_OBSERVATION_DIM = 65
TIP_DELTA_LIMIT_M = 0.00012  # Matches the runtime commissioning cap per 30 Hz action.
ROLL_SPEED_LIMIT_RAD_S = 0.4  # Matches InstrumentLimits.rotation_speed_rad_s.
CONTROL_HZ = 30.0
ROLL_DELTA_LIMIT_RAD = ROLL_SPEED_LIMIT_RAD_S / CONTROL_HZ
WRENCH_FORCE_SCALE_N = 0.15   # Sensor stop scale; not a calibrated tissue-force scale.
WRENCH_TORQUE_SCALE_NM = 0.02
JAW_EFFORT_SCALE_NM = 0.001
SCOPE_ROTATION_SCALE_RAD = np.deg2rad(15.0)
WAYPOINT_TOLERANCE_M = 0.0005
WAYPOINT_HOLD_STEPS = 10
WAYPOINT_SUCCESS_BONUS = 5.0


def normalized_action_to_rosota(action):
    """Map 14 normalized actions to two Rosota tool rows and four scope rates.

    Per hand: translation xyz, shaft-axis roll, jaw aperture (0 closed, 100 open). Tool tilt
    is intentionally fixed by the strict aperture profile. Scope channels are
    normalized rates passed to the scope's geometric RCM controller.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError(f"Expected {ACTION_DIM} finite normalized actions in [-1, 1]")
    tools = np.zeros((2, 1, 7), dtype=np.float32)
    for hand in range(2):
        offset = hand * 5
        tools[hand, 0, :3] = (action[offset:offset + 3] * TIP_DELTA_LIMIT_M).astype(np.float32)
        # Rows 3:5 are currently rejected by the strict Rosota controller.
        # Keep them exactly zero instead of giving PPO two ineffective controls.
        tools[hand, 0, 5] = np.float32(action[offset + 3] * ROLL_DELTA_LIMIT_RAD)
        tools[hand, 0, 6] = np.float32((action[offset + 4] + 1.0) * 50.0)
    return {"tools": tools, "scope": action[10:14].astype(np.float32)}


def encode_rosota_observation(observation, initial_scope_pose, step_fraction):
    """Encode measured robot, tool, scope and local sensor state for an MLP.

    `contact_residual` and `jaw_effort` are simulator-local sensor signals.
    They are not calibrated external tissue force measurements.
    """
    required = {"joint_position", "tool_pose", "scope_pose", "contact_residual",
                "jaw_position", "jaw_effort", "jaw_contact_hold", "contact_stop",
                "tissue_held", "goal_position"}
    if set(observation) != required:
        raise ValueError("Rosota observation keys do not match the versioned RL contract")
    q = np.asarray(observation["joint_position"], dtype=np.float64)
    tools = np.asarray(observation["tool_pose"], dtype=np.float64)
    scope = np.asarray(observation["scope_pose"], dtype=np.float64)
    wrench = np.asarray(observation["contact_residual"], dtype=np.float64)
    jaw = np.asarray(observation["jaw_position"], dtype=np.float64)
    effort = np.asarray(observation["jaw_effort"], dtype=np.float64)
    goal = np.asarray(observation["goal_position"], dtype=np.float64)
    initial = np.asarray(initial_scope_pose, dtype=np.float64)
    holds = np.asarray(observation["jaw_contact_hold"], dtype=np.float64)
    stops = np.asarray(observation["contact_stop"], dtype=np.float64)
    held = np.asarray(observation["tissue_held"], dtype=np.float64)
    shapes = ((q, (2, 7)), (tools, (2, 4, 4)), (scope, (4, 4)),
              (wrench, (2, 6)), (jaw, (2, 2)), (effort, (2, 2)),
              (goal, (2, 3)), (initial, (4, 4)), (holds, (2,)),
              (stops, (2,)), (held, (2,)))
    if any(value.shape != shape for value, shape in shapes):
        raise ValueError("Rosota observation array shape mismatch")
    if any(not np.isfinite(value).all() for value, _ in shapes) or not np.isfinite(step_fraction):
        raise ValueError("Rosota observation contains non-finite values")
    from scipy.spatial.transform import Rotation
    goal_error = (goal - tools[:, :3, 3]) / 0.02
    scope_translation = initial[:3, :3].T @ (scope[:3, 3] - initial[:3, 3]) / 0.03
    scope_rotation = Rotation.from_matrix(initial[:3, :3].T @ scope[:3, :3]).as_rotvec() / SCOPE_ROTATION_SCALE_RAD
    state = np.concatenate([
        (q / np.pi).ravel(),
        goal_error.ravel(),
        (jaw / (np.pi / 4)).ravel(),
        (effort / JAW_EFFORT_SCALE_NM).ravel(),
        (wrench[:, :3] / WRENCH_FORCE_SCALE_N).ravel(),
        (wrench[:, 3:] / WRENCH_TORQUE_SCALE_NM).ravel(),
        holds, stops, held,
        scope_translation, scope_rotation,
        [np.clip(step_fraction, 0.0, 1.0)],
    ])
    if state.shape != (OBSERVATION_DIM,):
        raise RuntimeError(f"Internal observation dimension drift: {state.shape}")
    return np.clip(state, -10.0, 10.0).astype(np.float32)


class RosotaRLAdapter(gym.Wrapper):
    """Expose a two-arm, scope-controlled waypoint benchmark with privileged state."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (OBSERVATION_DIM,), dtype=np.float32)
        self.initial_scope_pose = None
        self.steps = 0
        self.max_steps = int(getattr(env, "max_steps", 300))
        self.success_hold = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.initial_scope_pose = np.asarray(obs["scope_pose"], dtype=np.float64).copy()
        self.steps = 0
        self.success_hold = 0
        encoded = encode_rosota_observation(obs, self.initial_scope_pose, 0.0)
        return encoded, dict(info, rl_observation="privileged_state_v1",
                             rl_action="bimanual_trocar_scope_v1", vision_policy=False,
                             appendectomy_completed=False)

    def step(self, action):
        if self.initial_scope_pose is None:
            raise RuntimeError("Reset adapter before stepping")
        native = normalized_action_to_rosota(action)
        obs, reward, terminated, truncated, info = self.env.step(native)
        self.steps += 1
        distances = np.linalg.norm(np.asarray(obs["goal_position"]) -
                                   np.asarray(obs["tool_pose"])[:, :3, 3], axis=1)
        within_waypoint = bool(np.all(distances <= WAYPOINT_TOLERANCE_M))
        self.success_hold = self.success_hold + 1 if within_waypoint and not info.get("failure") else 0
        waypoint_success = self.success_hold >= WAYPOINT_HOLD_STEPS
        if waypoint_success and not terminated:
            reward = float(reward) + WAYPOINT_SUCCESS_BONUS
            terminated = True
        truncated = bool(truncated and not terminated)
        encoded = encode_rosota_observation(obs, self.initial_scope_pose,
                                            self.steps / max(self.max_steps, 1))
        return encoded, reward, terminated, truncated, dict(
            info, rl_observation="privileged_state_v1", rl_action="bimanual_trocar_scope_v1",
            vision_policy=False, tissue_force_calibrated=False,
            task_scope="two-arm trocar-constrained waypoint reach; scope RCM actions exposed but not rewarded; no grasp or appendectomy task",
            waypoint_distance_m=distances.tolist(), waypoint_hold_steps=self.success_hold,
            waypoint_is_success=waypoint_success, waypoint_tolerance_m=WAYPOINT_TOLERANCE_M,
            appendectomy_completed=False)

    def scripted_waypoint_action(self):
        """Geometric comparison baseline using measured FR3 FK; never a learned policy."""
        if self.initial_scope_pose is None:
            raise RuntimeError("Reset adapter before requesting a scripted action")
        if not all(hasattr(self.env, name) for name in ("controllers", "specs", "goal", "q")):
            raise RuntimeError("Wrapped environment does not expose the measured Rosota controller")
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for index, hand in enumerate(self.env.specs):
            position, rotation, _ = self.env.controllers[hand].pose(self.env.q(hand))
            delta_local = rotation.T @ (np.asarray(self.env.goal[index]) - position)
            distance = float(np.linalg.norm(delta_local))
            if distance > 0:
                step = min(distance, TIP_DELTA_LIMIT_M)
                action[index * 5:index * 5 + 3] = delta_local / distance * (step / TIP_DELTA_LIMIT_M)
            action[index * 5 + 4] = 1.0  # Native command 100 means full configured jaw aperture.
        return action


def encode_rosota_traction_observation(observation,info,initial_scope_pose,step_fraction):
    """Extend privileged state with both target offsets and measured patch motion."""
    state=encode_rosota_observation(observation,initial_scope_pose,step_fraction)
    geometry=info.get("grasp_geometry",{})
    if set(geometry)!={"left","right"}:
        raise ValueError("Bimanual traction observation requires both target-patch measurements")
    relative=np.asarray([geometry[h]["centre_in_shaft_frame_m"] for h in ("left","right")],float)
    displacement=np.asarray([geometry[h]["patch_displacement_world_m"] for h in ("left","right")],float)
    if relative.shape!=(2,3) or displacement.shape!=(2,3) or not np.isfinite(relative).all() or not np.isfinite(displacement).all():
        raise ValueError("Invalid bimanual patch geometry")
    encoded=np.concatenate([state,(relative/.01).ravel(),(displacement/.01).ravel()])
    if encoded.shape!=(TRACTION_OBSERVATION_DIM,) or not np.isfinite(encoded).all():
        raise RuntimeError("Internal bimanual traction observation dimension drift")
    return np.clip(encoded,-10.,10.).astype(np.float32)


def traction_task_metrics(observation,info,proximity_m=.005,target_lift_m=.005):
    """Compute a transparent unassisted two-patch contact-and-traction gate."""
    if not np.isfinite([proximity_m,target_lift_m]).all() or min(proximity_m,target_lift_m)<=0:
        raise ValueError("Positive finite contact and lift thresholds required")
    geometry=info.get("grasp_geometry",{})
    if set(geometry)!={"left","right"}:
        raise ValueError("Bimanual traction metrics require both target patches")
    distances=np.asarray([geometry[h]["distance_m"] for h in ("left","right")],float)
    lifts=np.asarray([geometry[h]["patch_lift_m"] for h in ("left","right")],float)
    contacts=np.asarray(observation["jaw_contact_hold"],bool)
    if distances.shape!=(2,) or lifts.shape!=(2,) or contacts.shape!=(2,) or not np.isfinite([distances,lifts]).all():
        raise ValueError("Invalid measured bimanual traction state")
    contact_valid=contacts&(distances<=proximity_m)
    lift_fraction=np.clip(lifts/target_lift_m,0.,1.)
    potential=float(-.25*np.mean(distances/.01)+.25*np.mean(contact_valid)+np.mean(contact_valid*lift_fraction))
    success=bool(np.all(contact_valid)&np.all(lifts>=target_lift_m))
    return dict(potential=potential,distances_m=distances.tolist(),lifts_m=lifts.tolist(),
        contact_valid=contact_valid.tolist(),success=success,
        proximity_threshold_m=float(proximity_m),target_lift_m=float(target_lift_m))


class RosotaTractionRLAdapter(RosotaRLAdapter):
    """Unassisted bimanual target-patch contact and traction benchmark."""
    def __init__(self,env,*,proximity_m=.005,target_lift_m=.005):
        super().__init__(env)
        self.observation_space=spaces.Box(-10.,10.,(TRACTION_OBSERVATION_DIM,),dtype=np.float32)
        self.proximity_m=float(proximity_m);self.target_lift_m=float(target_lift_m)
        self.previous_potential=0.;self._last_obs=None;self._last_info=None
        self.task="unassisted-bimanual-target-patch-traction-v1"

    def _encode(self,observation,info):
        return encode_rosota_traction_observation(observation,info,self.initial_scope_pose,
            self.steps/max(self.max_steps,1))

    def reset(self,*,seed=None,options=None):
        obs,info=self.env.reset(seed=seed,options=options)
        self.initial_scope_pose=np.asarray(obs["scope_pose"],dtype=np.float64).copy()
        self.steps=0;self.success_hold=0
        metrics=traction_task_metrics(obs,info,self.proximity_m,self.target_lift_m)
        self.previous_potential=metrics["potential"]
        self._last_obs=obs;self._last_info=info
        encoded=self._encode(obs,info)
        return encoded,dict(info,rl_observation="privileged_traction_state_v1",
            rl_action="bimanual_trocar_scope_v1",vision_policy=False,
            task_scope="unassisted bimanual patch contact and upward traction; not appendectomy",
            appendectomy_completed=False)

    def step(self,action):
        if self.initial_scope_pose is None:raise RuntimeError("Reset adapter before stepping")
        native=normalized_action_to_rosota(action)
        obs,_,terminated,truncated,info=self.env.step(native)
        self.steps+=1
        metrics=traction_task_metrics(obs,info,self.proximity_m,self.target_lift_m)
        reward=.99*metrics["potential"]-self.previous_potential-.002
        self.previous_potential=metrics["potential"]
        self.success_hold=self.success_hold+1 if metrics["success"] and not info.get("failure") else 0
        task_success=self.success_hold>=WAYPOINT_HOLD_STEPS
        if info.get("failure"):
            reward=-5.;terminated=True
        elif task_success and not terminated:
            reward+=5.;terminated=True
        truncated=bool(truncated and not terminated)
        encoded=self._encode(obs,info)
        self._last_obs=obs;self._last_info=info
        return encoded,reward,bool(terminated),truncated,dict(info,
            rl_observation="privileged_traction_state_v1",rl_action="bimanual_trocar_scope_v1",
            vision_policy=False,tissue_force_calibrated=False,
            task_scope="unassisted bimanual patch contact and upward traction; not appendectomy",
            task_success=task_success,task_hold_steps=self.success_hold,
            task_distance_m=metrics["distances_m"],task_lift_m=metrics["lifts_m"],
            task_contact_valid=metrics["contact_valid"],task_thresholds={
                "patch_proximity_m":metrics["proximity_threshold_m"],
                "target_lift_m":metrics["target_lift_m"],"hold_steps":WAYPOINT_HOLD_STEPS},
            appendectomy_completed=False)

    def scripted_task_action(self):
        """Geometric unassisted baseline: approach each patch, close, then lift."""
        if self._last_obs is None or self._last_info is None:
            raise RuntimeError("Reset traction adapter before requesting a scripted action")
        action=np.zeros(ACTION_DIM,dtype=np.float32)
        from scipy.spatial.transform import Rotation
        geometry=self._last_info["grasp_geometry"]
        contacts=np.asarray(self._last_obs["jaw_contact_hold"],bool)
        for index,hand in enumerate(("left","right")):
            offset=index*5;target=geometry[hand]
            distance=float(target["distance_m"]);lift=float(target["patch_lift_m"])
            rotation=Rotation.from_matrix(self._last_obs["tool_pose"][index,:3,:3]).as_matrix()
            if distance>self.proximity_m:
                relative=np.asarray(target["centre_in_shaft_frame_m"],float)
                action[offset:offset+3]=relative/max(distance,1e-12)
                action[offset+4]=1.0
            else:
                action[offset+4]=-1.0
                if contacts[index] and lift<self.target_lift_m:
                    local_up=rotation.T@np.array([0.,1.,0.])
                    action[offset:offset+3]=local_up/max(np.linalg.norm(local_up),1e-12)
        return np.clip(action,-1.,1.)
