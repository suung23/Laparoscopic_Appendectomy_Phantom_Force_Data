import sys
import tempfile
import json
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from rosota_rl_adapter import (TRACTION_OBSERVATION_DIM,
                               VISION_IMAGE_SHAPE,
                               VISION_PROPRIOCEPTION_DIM,
                               RosotaTractionRLAdapter,
                               RosotaVisionTractionRLAdapter,
                               encode_rosota_proprioception,
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

    def render(self):
        frame=np.zeros((360,640,3),dtype=np.uint8)
        frame[120:240,200:440]=(80,120,160)
        return frame


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
            vision = asset_contract(assets, urdf, "vision-traction")
        self.assertEqual(waypoint["observation_dim"], 53)
        self.assertEqual(traction["observation_dim"], TRACTION_OBSERVATION_DIM)
        self.assertNotEqual(waypoint["environment"], traction["environment"])
        self.assertFalse(traction["appendectomy_completed"])
        self.assertEqual(vision["rgb_observation_shape"], [90, 160, 3])
        self.assertTrue(vision["policy_observes_rgb"])
        self.assertEqual(vision["policy_architecture"], "cnn_endoscope_rgb_proprio_v1")

    def test_vision_checkpoint_load_requires_matching_policy_contract(self):
        import torch
        from ppo_agent import VisionActorCritic, load_checkpoint, save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            model=VisionActorCritic(proprioception=VISION_PROPRIOCEPTION_DIM,actions=14)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            checkpoint=Path(tmp)/"vision.pt"
            contract={"task":"vision-traction","policy_architecture":"cnn_endoscope_rgb_proprio_v1"}
            save_checkpoint(checkpoint,model,optimizer,contract)
            self.assertEqual(load_checkpoint(checkpoint,model,expected=contract)["task"],
                             "vision-traction")
            with self.assertRaisesRegex(ValueError,"policy_architecture"):
                load_checkpoint(checkpoint,model,expected={"policy_architecture":"mlp_privileged_v1"})

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

    def test_rgb_policy_receives_endoscope_image_and_only_proprioceptive_state(self):
        core=DummyTractionEnv()
        env=RosotaVisionTractionRLAdapter(core)
        obs,info=env.reset(seed=4)
        self.assertEqual(obs["rgb"].shape,VISION_IMAGE_SHAPE)
        self.assertEqual(obs["rgb"].dtype,np.uint8)
        self.assertEqual(obs["proprio"].shape,(VISION_PROPRIOCEPTION_DIM,))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(info["vision_policy"])
        self.assertEqual(info["rl_observation"],"endoscope_rgb_proprioception_v1")
        next_obs,_,_,_,step_info=env.step(np.zeros(14,np.float32))
        self.assertTrue(env.observation_space.contains(next_obs))
        self.assertTrue(step_info["vision_policy"])

    def test_proprioception_encoder_does_not_include_privileged_goal_geometry(self):
        obs=observation()
        obs.update({"tool_pose":np.tile(np.eye(4),(2,1,1)),
                    "tissue_held":np.zeros(2,np.int8),"goal_position":np.ones((2,3))})
        encoded=encode_rosota_proprioception(obs,np.eye(4),.5)
        self.assertEqual(encoded.shape,(VISION_PROPRIOCEPTION_DIM,))
        self.assertAlmostEqual(float(encoded[-1]),.5)

    def test_rgb_ppo_rollout_updates_cnn_policy_on_mock_endoscope_frames(self):
        import torch
        from ppo_agent import VisionActorCritic
        from run_rosota_rl import train_ppo

        torch.set_num_threads(1)
        env=RosotaVisionTractionRLAdapter(DummyTractionEnv())
        model=VisionActorCritic(proprioception=VISION_PROPRIOCEPTION_DIM,actions=14)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        result=train_ppo(env,model,optimizer,total_steps=64,seed=19,rollout_steps=64)
        self.assertEqual(result["environment_steps"],64)
        self.assertEqual(len(result["updates"]),1)
        self.assertTrue(np.isfinite(result["updates"][0]["loss_mean"]))


if __name__ == "__main__":
    unittest.main()
