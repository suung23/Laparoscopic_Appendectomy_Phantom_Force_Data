import sys
import tempfile
import json
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from rosota_rl_adapter import (TRACTION_OBSERVATION_DIM,
                               RosotaTractionRLAdapter,
                               encode_rosota_traction_observation,
                               traction_task_metrics)
from tissue_metrics import target_patch_motion
from run_rosota_rl import asset_contract


def observation(contacts=(0, 0)):
    return {
        "joint_position": np.zeros((2, 7), np.float32),
        "tool_pose": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "scope_pose": np.eye(4, dtype=np.float32),
        "contact_residual": np.zeros((2, 6), np.float32),
        "jaw_position": np.zeros((2, 2), np.float32),
        "jaw_effort": np.zeros((2, 2), np.float32),
        "jaw_contact_hold": np.asarray(contacts, np.int8),
        "contact_stop": np.zeros(2, np.int8),
        "tissue_held": np.zeros(2, np.int8),
        "goal_position": np.zeros((2, 3), np.float32),
    }


def geometry(distances=(.002, .002), lifts=(0., 0.)):
    return {
        hand: {"distance_m": distance, "patch_lift_m": lift,
               "centre_in_shaft_frame_m": [distance, 0., 0.],
               "patch_displacement_world_m": [0., lift, 0.]}
        for hand, distance, lift in zip(("left", "right"), distances, lifts)
    }


class DummyTractionEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})
        self.max_steps = 20
        self.count = 0

    def reset(self, *, seed=None, options=None):
        self.count = 0
        return observation(), {"grasp_geometry": geometry()}

    def step(self, action):
        self.count += 1
        lift = min(self.count * .0006, .006)
        return observation((1, 1)), 0., False, False, {
            "grasp_geometry": geometry(lifts=(lift, lift))}


class RosotaTractionAdapterTests(unittest.TestCase):
    def test_task_contract_versions_traction_observation_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            (assets / "attached.usda").write_text("scene", encoding="utf-8")
            (assets / "bimanual_config.json").write_text(json.dumps({}), encoding="utf-8")
            urdf = root / "fr3.urdf"
            urdf.write_text("urdf", encoding="utf-8")
            waypoint = asset_contract(assets, urdf, "waypoint")
            traction = asset_contract(assets, urdf, "unassisted-traction")
        self.assertEqual(waypoint["observation_dim"], 53)
        self.assertEqual(traction["observation_dim"], TRACTION_OBSERVATION_DIM)
        self.assertNotEqual(waypoint["environment"], traction["environment"])
        self.assertFalse(traction["appendectomy_completed"])

    def test_patch_measurement_uses_reset_reference_and_world_up(self):
        reference = np.array([[0., 0., 0.], [.001, 0., 0.], [0., 0., .001]])
        current = reference + np.array([.002, .006, -.001])
        measured = target_patch_motion(reference, current, [0, 1, 2],
                                       np.array([.002333333333, .003, -.000666666667]), np.eye(3),
                                       np.array([0., 1., 0.]))
        self.assertAlmostEqual(measured["patch_lift_m"], .006)
        self.assertAlmostEqual(measured["distance_m"], .003)
        np.testing.assert_allclose(measured["patch_displacement_world_m"],
                                   [.002, .006, -.001])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            target_patch_motion(reference, current, [0, 1, 2], [0, 0, 0],
                                np.diag([2., 1., 1.]), [0, 1, 0])

    def test_patch_motion_features_append_to_privileged_state(self):
        obs = observation()
        info = {"grasp_geometry": geometry(distances=(.003, .004), lifts=(.001, .002))}
        encoded = encode_rosota_traction_observation(obs, info, np.eye(4), .25)
        self.assertEqual(encoded.shape, (TRACTION_OBSERVATION_DIM,))
        self.assertTrue(np.isfinite(encoded).all())
        np.testing.assert_allclose(encoded[-12:-6], [.3, 0, 0, .4, 0, 0])
        np.testing.assert_allclose(encoded[-6:], [0, .1, 0, 0, .2, 0])

    def test_task_success_requires_both_nearby_contact_latches_and_lift(self):
        obs = observation((1, 1))
        good = traction_task_metrics(obs, {"grasp_geometry": geometry(
            distances=(.004, .005), lifts=(.005, .006))})
        self.assertTrue(good["success"])
        missing_contact = observation((1, 0))
        self.assertFalse(traction_task_metrics(missing_contact, {"grasp_geometry": geometry(
            lifts=(.01, .01))})["success"])
        too_far = traction_task_metrics(obs, {"grasp_geometry": geometry(
            distances=(.006, .002), lifts=(.01, .01))})
        self.assertFalse(too_far["success"])
        too_low = traction_task_metrics(obs, {"grasp_geometry": geometry(
            lifts=(.004, .01))})
        self.assertFalse(too_low["success"])

    def test_adapter_rewards_unassisted_task_and_holds_success_ten_steps(self):
        env = RosotaTractionRLAdapter(DummyTractionEnv())
        state, info = env.reset(seed=2)
        self.assertEqual(state.shape, (TRACTION_OBSERVATION_DIM,))
        self.assertEqual(info["rl_observation"], "privileged_traction_state_v1")
        self.assertFalse(info["appendectomy_completed"])
        successes = []
        for _ in range(20):
            state, reward, terminated, truncated, info = env.step(np.zeros(14, np.float32))
            self.assertTrue(env.observation_space.contains(state))
            successes.append(info["task_success"])
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(successes[-1])
        self.assertEqual(info["task_hold_steps"], 10)
        self.assertEqual(info["task_thresholds"]["patch_proximity_m"], .005)


if __name__ == "__main__":
    unittest.main()
