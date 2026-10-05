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
