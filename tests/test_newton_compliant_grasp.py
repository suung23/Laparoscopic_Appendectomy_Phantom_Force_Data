import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from grasp import GraspAssist, compliant_grasp_forces
sys.path.insert(0, str(ROOT / "isaac"))
import rollout as newton_rollout


class Array:
    def __init__(self, value):
        self.value = np.asarray(value, np.float64).copy()

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value[...] = np.asarray(value)


class Arm:
    def __init__(self, position):
        self.grip_cmd = 0.0
        self.T = np.eye(4)
        self.T[:3, 3] = position

    def link_frames(self):
        return {"tool_tip": self.T.copy()}


class State:
    def __init__(self, positions, velocities):
        self.particle_q = Array(positions)
        self.particle_qd = Array(velocities)
        self.particle_f = Array(np.zeros_like(positions))


class Model:
    def __init__(self, inverse_masses):
        self.particle_inv_mass = Array(inverse_masses)


class NewtonCompliantGraspTests(unittest.TestCase):
    def test_spring_direction_and_total_force_cap(self):
        q = np.zeros((2, 3))
        qd = np.zeros_like(q)
        target = np.tile([0.001, 0.0, 0.0], (2, 1))
        force, total = compliant_grasp_forces(
            q, qd, target, np.zeros_like(q),
            stiffness_n_m=30.0, damping_n_s_m=0.0, force_limit_n=0.02)
        self.assertTrue((force[:, 0] > 0.0).all())
        self.assertAlmostEqual(total, 0.02, places=8)
        self.assertAlmostEqual(float(np.linalg.norm(force, axis=1).sum()), 0.02,
                               places=8)

    def test_damping_opposes_relative_node_velocity(self):
        q = np.zeros((1, 3))
        qd = np.array([[0.1, 0.0, 0.0]])
        force, total = compliant_grasp_forces(
            q, qd, q, np.zeros_like(q),
            stiffness_n_m=30.0, damping_n_s_m=0.5, force_limit_n=0.03)
        self.assertLess(float(force[0, 0]), 0.0)
        self.assertGreater(total, 0.0)

    def test_latched_cluster_receives_force_without_position_teleport(self):
        q = np.array([[-0.002, 0.0, 0.0], [0.002, 0.0, 0.0]])
        qd = np.zeros_like(q)
        state = State(q, qd)
        model = Model([1000.0, 1000.0])
        arm = Arm([0.0, 0.0, 0.0])
        scene = type("Scene", (), {"handles": [{"idx": np.array([0, 1])}]})()
        grasp = GraspAssist(scene, {"left": arm}, model=model, dt=1 / 480,
                            force_limit_n=0.03, stiffness_n_m=30.0)

        grasp.update(state)
        self.assertTrue(grasp.grasped("left"))
        arm.T[0, 3] += 0.001
        grasp.apply_forces(state)

        np.testing.assert_allclose(state.particle_q.numpy(), q)
        np.testing.assert_allclose(state.particle_qd.numpy(), qd)
        forces = state.particle_f.numpy()
        self.assertTrue((forces[:, 0] > 0.0).all())
        self.assertLessEqual(float(np.linalg.norm(forces, axis=1).sum()), 0.03000001)
        self.assertGreater(grasp.info()["deflection_mm"]["left"], 0.0)

    def test_two_arms_can_latch_distinct_handles(self):
        q = np.array([[-0.002, 0.0, 0.0], [0.002, 0.0, 0.0],
                      [0.098, 0.0, 0.0], [0.102, 0.0, 0.0]])
        state = State(q, np.zeros_like(q))
        scene = type("Scene", (), {"handles": [
            {"idx": np.array([0, 1])}, {"idx": np.array([2, 3])}]})()
        arms = {"left": Arm([0.0, 0.0, 0.0]),
                "right": Arm([0.1, 0.0, 0.0])}
        grasp = GraspAssist(scene, arms, model=Model([1000.0] * 4),
                            dt=1 / 480)

        grasp.update(state)
        self.assertEqual({x["handle"] for x in grasp.latched.values()}, {0, 1})
        self.assertTrue(grasp.grasped("left"))
        self.assertTrue(grasp.grasped("right"))

    def test_episode_npz_recorder_captures_model_force_separately(self):
        newton_rollout.install_newton_rollout_metrics()
        rec = newton_rollout.vla_rollout.Episode(0, keep_images=False)
        rec.add({"T_world_rig": np.eye(4, dtype=np.float32),
                 "grasp_assist": {
            "left_model_force_n": 0.012,
            "right_model_force_n": 0.0,
            "left_deflection_m": 0.0004,
            "right_deflection_m": 0.0,
        }})
        self.assertEqual(rec.rows["left_grasp_assist_model_force_n"], [0.012])
        self.assertEqual(rec.rows["right_grasp_assist_model_force_n"], [0.0])
        self.assertEqual(rec.rows["left_grasp_assist_deflection_m"], [0.0004])


if __name__ == "__main__":
    unittest.main()
