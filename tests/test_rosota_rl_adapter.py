import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from rosota_rl_adapter import (ACTION_DIM, OBSERVATION_DIM, RosotaRLAdapter,
                               encode_rosota_observation, normalized_action_to_rosota)


def observation():
    return {
        "joint_position": np.zeros((2, 7), np.float32),
        "tool_pose": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "scope_pose": np.eye(4, dtype=np.float32),
        "contact_residual": np.zeros((2, 6), np.float32),
        "jaw_position": np.zeros((2, 2), np.float32),
        "jaw_effort": np.zeros((2, 2), np.float32),
        "jaw_contact_hold": np.zeros(2, np.int8),
        "contact_stop": np.zeros(2, np.int8),
        "tissue_held": np.zeros(2, np.int8),
        "goal_position": np.zeros((2, 3), np.float32),
    }


class DummyRosotaEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})
        self.max_steps = 12
        self.received = None

    def reset(self, *, seed=None, options=None):
        return observation(), {"trocar_ok": True}

    def step(self, action):
        self.received = action
        return observation(), 0.0, False, False, {"trocar_ok": True}


class FakeController:
    def pose(self, q):
        return np.zeros(3), np.eye(3), np.array([0, 0, 1.])


class ScriptableDummyEnv(DummyRosotaEnv):
    def __init__(self):
        super().__init__()
        self.specs = {"left": {}, "right": {}}
        self.controllers = {hand: FakeController() for hand in self.specs}
        self.goal = np.array([[.001, 0, 0], [0, .001, 0]])

    def q(self, hand):
        return np.zeros(7)


class RosotaRLAdapterTests(unittest.TestCase):
    def test_action_mapping_uses_safe_translation_roll_jaw_and_scope_channels(self):
        action = np.zeros(ACTION_DIM, np.float32)
        action[0:5] = [1, -1, .5, 1, -1]
        action[5:10] = [-1, 1, -.5, -1, 1]
        action[10:14] = [.1, .2, .3, .4]

        native = normalized_action_to_rosota(action)

        self.assertEqual(native["tools"].shape, (2, 1, 7))
        np.testing.assert_allclose(native["tools"][0, 0, :3], [0.00012, -0.00012, 0.00006])
        np.testing.assert_allclose(native["tools"][1, 0, :3], [-0.00012, 0.00012, -0.00006])
        np.testing.assert_array_equal(native["tools"][:, 0, 3:5], 0)
        self.assertAlmostEqual(float(native["tools"][0, 0, 5]), .4 / 30)
        self.assertAlmostEqual(float(native["tools"][1, 0, 5]), -.4 / 30)
        np.testing.assert_allclose(native["tools"][:, 0, 6], [0, 100])
        np.testing.assert_allclose(native["scope"], [.1, .2, .3, .4])

    def test_rejects_nonfinite_and_out_of_range_actions(self):
        for bad in (np.zeros(ACTION_DIM - 1), np.full(ACTION_DIM, np.nan),
                    np.full(ACTION_DIM, 1.01)):
            with self.assertRaises(ValueError):
                normalized_action_to_rosota(bad)

    def test_observation_encoding_and_wrapper_contract(self):
        core = DummyRosotaEnv()
        wrapped = RosotaRLAdapter(core)
        state, info = wrapped.reset(seed=17)
        self.assertEqual(state.shape, (OBSERVATION_DIM,))
        self.assertTrue(wrapped.observation_space.contains(state))
        self.assertEqual(info["rl_observation"], "privileged_state_v1")
        self.assertFalse(info["vision_policy"])

        action = np.zeros(ACTION_DIM, np.float32)
        next_state, reward, terminated, truncated, step_info = wrapped.step(action)
        self.assertTrue(wrapped.observation_space.contains(next_state))
        self.assertEqual(core.received["tools"].shape, (2, 1, 7))
        self.assertEqual(core.received["scope"].shape, (4,))
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertFalse(step_info["tissue_force_calibrated"])
        self.assertEqual(step_info["task_scope"],
                         "two-arm trocar-constrained waypoint reach; scope RCM actions exposed but not rewarded; no grasp or appendectomy task")
        for _ in range(8):
            _, reward, terminated, truncated, _ = wrapped.step(action)
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        _, reward, terminated, truncated, step_info = wrapped.step(action)
        self.assertEqual((reward, terminated, truncated), (5.0, True, False))
        self.assertTrue(step_info["waypoint_is_success"])
        self.assertFalse(step_info["appendectomy_completed"])

    def test_geometric_waypoint_baseline_uses_fk_and_keeps_jaws_open(self):
        wrapped = RosotaRLAdapter(ScriptableDummyEnv())
        wrapped.reset(seed=4)
        action = wrapped.scripted_waypoint_action()
        np.testing.assert_allclose(action[:3], [1, 0, 0])
        np.testing.assert_allclose(action[5:8], [0, 1, 0])
        self.assertEqual(action[4], 1.0)
        self.assertEqual(action[9], 1.0)
        np.testing.assert_array_equal(action[10:14], 0)

    def test_observation_shape_and_finiteness_are_fail_closed(self):
        obs = observation()
        encoded = encode_rosota_observation(obs, np.eye(4), .5)
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded.shape, (OBSERVATION_DIM,))
        self.assertAlmostEqual(float(encoded[-1]), .5)
        obs["jaw_effort"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            encode_rosota_observation(obs, np.eye(4), 0)

    def test_ppo_rollout_uses_adapter_contract_without_simulator_import(self):
        import torch
        from ppo_agent import ActorCritic
        from run_rosota_rl import train_ppo

        torch.set_num_threads(1)
        env = RosotaRLAdapter(DummyRosotaEnv())
        model = ActorCritic(observations=OBSERVATION_DIM, actions=ACTION_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        result = train_ppo(env, model, optimizer, total_steps=64, seed=11,
                           rollout_steps=64)

        self.assertEqual(result["environment_steps"], 64)
        self.assertEqual(len(result["updates"]), 1)
        self.assertEqual(result["updates"][0]["batch_size"], 64)
        self.assertTrue(result["episodes"])
        self.assertTrue(all(np.isfinite(update["loss_mean"]) for update in result["updates"]))


if __name__ == "__main__":
    unittest.main()
