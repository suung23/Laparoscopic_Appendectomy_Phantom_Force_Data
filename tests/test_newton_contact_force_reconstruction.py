import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from forces import ContactForces, contact_node_diagnostics


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def numpy(self):
        return self.values


class NewtonContactForceTests(unittest.TestCase):
    def test_failure_contact_diagnostics_attribute_link_and_worst_tet_nodes(self):
        result = contact_node_diagnostics(
            shape_ids=[2, 7, 99],
            contact_indices=[[10, 11, -1], [12, 13, 14], [15, 16, -1]],
            shape_to_link={2: "left/jaw", 7: "right/shaft"},
            organ_spans={"cecum": (10, 20), "appendix": (30, 40)},
            worst_tet_nodes={"cecum": [10, 11, 12, 13],
                             "appendix": [30, 31, 32, 33]})

        self.assertEqual(result["soft_contact_records"], 3)
        self.assertEqual(result["by_link"]["left/jaw"]["soft_contact_records"], 1)
        self.assertEqual(result["by_link"]["left/jaw"]["particle_counts_by_organ"]["cecum"], 2)
        self.assertEqual(result["by_link"]["left/jaw"]["worst_tet_contacts_by_organ"]["cecum"], 1)
        self.assertEqual(result["by_link"]["left/jaw"][
            "worst_tet_contact_particle_ids_by_organ"]["cecum"], [10, 11])
        self.assertEqual(result["by_link"]["right/shaft"]["particle_ids_by_organ"]["cecum"],
                         [12, 13, 14])
        self.assertEqual(set(result["by_link"]), {"left/jaw", "right/shaft"})

    def test_edge_contact_uses_barycentric_nodes_not_minus_one_particle_sentinel(self):
        model = SimpleNamespace(
            shape_count=1, body_count=1, particle_count=3,
            shape_body=Array([0]), particle_radius=Array([.001, .001, .001]),
            shape_margin=Array([0.]), shape_material_ke=Array([1000.]),
            shape_material_kd=Array([0.]), shape_material_mu=Array([0.]),
            soft_contact_ke=1000., soft_contact_kd=0., soft_contact_mu=0.)
        tool_pose = np.eye(4)
        tool_pose[:3, 3] = [0., .001, 0.]
        arms = {"left": SimpleNamespace(
            bodies={"tool": 0},
            link_frames=lambda: {"tool_tip": tool_pose})}
        identity_q = np.array([[0., 0., 0., 0., 0., 0., 1.]])
        state = SimpleNamespace(
            particle_q=Array([[.0002, .002, 0.], [.0002, .002, 0.],
                              [20., 20., 20.]]),
            particle_qd=Array(np.zeros((3, 3))), body_q=Array(identity_q))
        contacts = SimpleNamespace(
            soft_contact_count=Array([1]), soft_contact_max=1,
            # Edge record; particle-only view is the -1 sentinel by contract.
            soft_contact_particle=Array([-1]),
            soft_contact_indices=Array([[0, 1, -1]]),
            soft_contact_barycentric=Array([[.5, .5, 0.]]),
            soft_contact_shape=Array([0]),
            soft_contact_normal=Array([[1., 0., 0.]]),
            soft_contact_body_pos=Array([[0., .002, 0.]]),
            soft_contact_body_vel=Array([[0., 0., 0.]]))
        solver = SimpleNamespace(
            particle_q_prev=Array(state.particle_q.values.copy()),
            body_q_prev=Array(identity_q.copy()),
            # The dual-updated post-step penalty is not the force coefficient
            # used during this step and must not be mistaken for a sensor gain.
            body_particle_contact_penalty_k=Array([1e12]),
            body_particle_contact_material_kd=Array([0.]),
            body_particle_contact_material_mu=Array([0.]),
            friction_epsilon=1e-3)

        wrench = ContactForces(model, arms, solver=solver, dt=1 / 480).tool_wrench(
            state, contacts, "left")

        np.testing.assert_allclose(wrench[:3], [-.8, 0., 0.], atol=1e-6)
        np.testing.assert_allclose(wrench[3:], [0., 0., .0008], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
