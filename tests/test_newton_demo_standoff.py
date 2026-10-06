import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from newton_cell.vla_env import DemoPolicy


def observation(tip, target, grasped=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(tip, dtype=np.float32)
    return {
        "left": {"tip_pos_rig": pose[:3, 3], "tip_pose_rig": pose,
                 "grasped": grasped},
        "task": {
            "handles_rig": np.asarray([target], dtype=np.float32),
            "handle_to_pin_m": np.asarray([0.05], dtype=np.float32),
            "handle_free": np.asarray([True]),
            "appendix_lift_m": 0.0,
        },
    }


class NewtonDemoStandoffTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(chunk=8, cfg=SimpleNamespace(
            grasp_band=(0.30, 0.70)))
        self.policy = DemoPolicy(self.env)

    def test_port_axis_rotation_tracks_candidate_tip_without_changing_path_contract(self):
        current = np.array([0.0, 0.0, 1.0])
        desired = np.array([0.0, 1.0, 0.0])
        rotation = self.policy._rotation_between(current, desired)
        np.testing.assert_allclose(rotation @ current, desired, atol=1e-7)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        opposite = self.policy._rotation_between(current, -current)
        np.testing.assert_allclose(opposite @ current, -current, atol=1e-7)
        np.testing.assert_allclose(opposite.T @ opposite, np.eye(3), atol=1e-7)

    def test_swept_mesh_clearance_prefers_route_away_from_cecum(self):
        tip = np.array([0.0, 0.0, 0.06])
        obstacles = np.array([[0.0, 0.0, 0.04]])
        empty_tool_point = np.zeros((1, 3))
        port = np.array([0.0, 0.0, -0.1])
        shaft = np.array([0.0, 0.0, 1.0])
        near = self.policy._swept_route_clearance(
            np.array([0.0, 0.0, 0.02]), tip, np.eye(3), shaft, port,
            obstacles, empty_tool_point)
        clear = self.policy._swept_route_clearance(
            np.array([0.05, 0.0, 0.02]), tip, np.eye(3), shaft, port,
            obstacles, empty_tool_point)
        self.assertLess(near, clear)

    def test_cecum_surface_uses_captured_mesh_to_fem_node_binding(self):
        # Display face IDs intentionally differ from FEM node IDs. The old
        # implementation indexed the FEM block directly with display IDs.
        particles = np.arange(36, dtype=np.float64).reshape(12, 3)
        faces = np.array([[0, 1, 2], [1, 2, 3]], np.int32)
        render_nodes = np.array([7, 2, 9, 5], np.int32)
        renderer = SimpleNamespace(
            _organ_faces={"cecum": faces},
            _organ_render_nodes={"cecum": render_nodes},
            _organ_render_stencils={"cecum": (
                render_nodes[:, None], np.ones((4, 1), np.float32))})
        cell = SimpleNamespace(
            scene=SimpleNamespace(organ_span={"cecum": (3, 15)}),
            state_0=SimpleNamespace(particle_q=SimpleNamespace(
                numpy=lambda: np.vstack([np.zeros((3, 3)), particles]))))
        env = SimpleNamespace(cell=cell, renderer=renderer,
                              _to_rig=lambda points: np.asarray(points))
        self.policy.env = env
        actual = self.policy._cecum_surface_rig()
        np.testing.assert_array_equal(actual, particles[render_nodes[:4]])

    def test_cecum_surface_composes_live_subdivision_stencils(self):
        particles = np.arange(30, dtype=np.float64).reshape(10, 3)
        faces = np.array([[0, 1, 2]], np.int32)
        indices = np.array([[1, 2], [4, 5], [7, 8]], np.int32)
        weights = np.full((3, 2), 0.5, np.float32)
        renderer = SimpleNamespace(
            _organ_faces={"cecum": faces},
            _organ_render_nodes={"cecum": np.arange(3, dtype=np.int32)},
            _organ_render_stencils={"cecum": (indices, weights)})
        cell = SimpleNamespace(
            scene=SimpleNamespace(organ_span={"cecum": (2, 12)}),
            state_0=SimpleNamespace(particle_q=SimpleNamespace(
                numpy=lambda: np.vstack([np.zeros((2, 3)), particles]))))
        env = SimpleNamespace(cell=cell, renderer=renderer,
                              _to_rig=lambda points: np.asarray(points))
        self.policy.env = env
        expected = np.stack([
            particles[[1, 2]].mean(axis=0),
            particles[[4, 5]].mean(axis=0),
            particles[[7, 8]].mean(axis=0),
        ])
        np.testing.assert_array_equal(self.policy._cecum_surface_rig(), expected)

    def test_cecum_clearance_samples_include_triangle_interiors(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], np.int32)
        policy = DemoPolicy(SimpleNamespace(
            cell=SimpleNamespace(
                scene=SimpleNamespace(organ_span={"cecum": (0, 3)}),
                state_0=SimpleNamespace(particle_q=SimpleNamespace(
                    numpy=lambda: vertices.copy()))),
            renderer=SimpleNamespace(
                _organ_faces={"cecum": faces},
                _organ_render_nodes={"cecum": np.arange(3, dtype=np.int32)},
                _organ_render_stencils={"cecum": (
                    np.arange(3, dtype=np.int32)[:, None],
                    np.ones((3, 1), np.float32))}),
            _to_rig=lambda points: np.asarray(points)))
        samples = policy._cecum_clearance_samples_rig()
        self.assertEqual(samples.shape, (7, 3))
        self.assertTrue(any(np.allclose(p, [0.5, 0.5, 0.0]) for p in samples))

    def test_target_handle_stays_locked_while_tissue_clearance_scores_change(self):
        selected = self.policy._locked_or_best_handle([1, 4, 5], [0.1, 0.9, 0.2])
        self.assertEqual(selected, 4)
        # A moving tissue cloud may change route scores; preserve the current
        # grasp target until it is no longer in the candidate set.
        selected = self.policy._locked_or_best_handle([1, 4, 5], [0.99, 0.1, 0.9])
        self.assertEqual(selected, 4)
        selected = self.policy._locked_or_best_handle([1, 5], [0.1, 0.9])
        self.assertEqual(selected, 5)

    def test_approach_targets_hover_point_before_tissue(self):
        tip = np.zeros(3)
        target = np.array([0.02, 0.0, 0.02])
        action = self.policy(observation(tip, target))
        expected = target + np.array([0.0, 0.0, self.policy.STANDOFF_M])
        expected_direction = (expected - tip) / np.linalg.norm(expected - tip)
        actual_step = action[0, 0, :3]
        self.assertEqual(self.policy.approach_phase, "standoff")
        self.assertAlmostEqual(float(np.dot(actual_step, expected_direction)),
                               self.policy.STEP_M, places=7)
        direct_direction = (target - tip) / np.linalg.norm(target - tip)
        self.assertGreater(np.linalg.norm(actual_step -
                                          self.policy.STEP_M * direct_direction),
                           1e-5)

    def test_scripted_approach_speed_override_converts_mm_per_second(self):
        env = SimpleNamespace(chunk=8, cfg=SimpleNamespace(
            control_hz=30.0, demo_approach_mm_s=6.0,
            grasp_band=(0.30, 0.70)))
        policy = DemoPolicy(env)
        self.assertAlmostEqual(policy.step_m, 0.0002, places=10)

    def test_scripted_approach_speed_must_be_positive_and_finite(self):
        for speed in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(speed=speed):
                env = SimpleNamespace(chunk=8, cfg=SimpleNamespace(
                    control_hz=30.0, demo_approach_mm_s=speed,
                    grasp_band=(0.30, 0.70)))
                with self.assertRaisesRegex(ValueError, "approach speed"):
                    DemoPolicy(env)

    def test_descends_then_stops_and_closes_inside_approach_band(self):
        target = np.array([0.02, 0.0, 0.02])
        hover = target + np.array([0.0, 0.0, self.policy.STANDOFF_M])
        action = self.policy(observation(hover, target))
        self.assertEqual(self.policy.approach_phase, "descend")
        self.assertTrue(np.allclose(action[0, :, :3], 0.0))

        near = target + np.array([0.0, 0.0,
                                  (self.policy.APPROACH_MM - 1.0) / 1000.0])
        action = self.policy(observation(near, target))
        self.assertEqual(self.policy.approach_phase, "close")
        self.assertTrue(np.allclose(action[0, :, :3], 0.0))
        self.assertTrue(np.all(action[0, :, 6] == 0.0))


if __name__ == "__main__":
    unittest.main()
