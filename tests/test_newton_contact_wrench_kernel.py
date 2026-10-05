import sys
import unittest
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from newton_contact_probe import _sum_tool_contact_wrenches


class NewtonContactWrenchKernelTests(unittest.TestCase):
    def test_known_penalty_contact_force_and_moment(self):
        """A 1 cm overlap at 1000 N/m must yield a 10 N reaction and known moment."""
        wp.init()
        device = "cpu"
        contact_count = wp.array([1], dtype=int, device=device)
        contacts = wp.array([wp.vec3i(0, -1, -1)], dtype=wp.vec3i, device=device)
        barycentric = wp.array([wp.vec3(1., 0., 0.)], dtype=wp.vec3, device=device)
        particle_q = wp.array([wp.vec3(.5, .09, 0.)], dtype=wp.vec3, device=device)
        particle_q_prev = wp.array([wp.vec3(.5, .09, 0.)], dtype=wp.vec3, device=device)
        particle_radius = wp.array([.1], dtype=float, device=device)
        contact_ke = wp.array([1000.], dtype=float, device=device)
        contact_kd = wp.array([0.], dtype=float, device=device)
        contact_mu = wp.array([0.], dtype=float, device=device)
        shape_body = wp.array([0], dtype=int, device=device)
        body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_q_prev = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        body_qd = wp.array([wp.spatial_vector()], dtype=wp.spatial_vector, device=device)
        body_com = wp.array([wp.vec3(0., 0., 0.)], dtype=wp.vec3, device=device)
        contact_shape = wp.array([0], dtype=int, device=device)
        contact_body_pos = wp.array([wp.vec3(.5, 0., 0.)], dtype=wp.vec3, device=device)
        contact_body_vel = wp.array([wp.vec3(0., 0., 0.)], dtype=wp.vec3, device=device)
        contact_normal = wp.array([wp.vec3(0., 1., 0.)], dtype=wp.vec3, device=device)
        shape_margin = wp.array([0.], dtype=float, device=device)
        tool_shape_ids = wp.array([0, 1], dtype=int, device=device)
        force_out = wp.zeros(2, dtype=wp.vec3, device=device)
        torque_out = wp.zeros(2, dtype=wp.vec3, device=device)

        wp.launch(
            _sum_tool_contact_wrenches,
            dim=1,
            inputs=[.01, contact_count, contacts, barycentric, particle_q,
                    particle_q_prev, particle_radius, contact_ke, contact_kd,
                    contact_mu, 1e-3, shape_body, body_q, body_q_prev, body_qd,
                    body_com, contact_shape, contact_body_pos, contact_body_vel,
                    contact_normal, shape_margin, tool_shape_ids],
            outputs=[force_out, torque_out],
            device=device,
        )
        np.testing.assert_allclose(force_out.numpy()[0], [0., -10., 0.], atol=1e-5)
        np.testing.assert_allclose(torque_out.numpy()[0], [0., 0., -5.], atol=1e-5)
        np.testing.assert_allclose(force_out.numpy()[1], [0., 0., 0.], atol=1e-8)
        np.testing.assert_allclose(torque_out.numpy()[1], [0., 0., 0.], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
