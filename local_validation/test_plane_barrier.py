import unittest
import numpy as np
from ct_appendectomy.plane_barrier import PlaneBarrier
from ct_appendectomy.constrained_elasticity import ElasticReference


class PlaneTests(unittest.TestCase):
    def test_translation_minimum_preserves_shape_and_balances_normal_load(self):
        barrier = PlaneBarrier([0,1,0], 0, .001, 100., [1.,2.])
        x = np.array([[0.,.001,0],[.002,.0012,0]])
        loads = np.array([[0.,-.03,0],[0.,-.02,0]])
        shifted, report = barrier.normal_translation_minimum(x, loads)
        np.testing.assert_allclose(shifted[1]-shifted[0], x[1]-x[0], atol=1e-15)
        self.assertLess(abs(report['normal_residual_n']), 1e-12)
        self.assertGreater(barrier.gaps(shifted).min(), 0)
        before = barrier.energy_gradient(x)[0]
        after = barrier.energy_gradient(shifted)[0]-np.sum(loads*(shifted-x))
        self.assertLess(after, before)

    def test_force_derivative_and_separation(self):
        barrier = PlaneBarrier([0,1,0], 0, .001, 100., [1.,2.])
        x = np.array([[0.,.0004,0],[0,.002,0]])
        _, gradient = barrier.energy_gradient(x)
        epsilon = 1e-9
        plus, minus = x.copy(), x.copy()
        plus[0,1] += epsilon; minus[0,1] -= epsilon
        numerical = (barrier.energy_gradient(plus)[0]-barrier.energy_gradient(minus)[0])/(2*epsilon)
        self.assertAlmostEqual(gradient[0,1], numerical, places=8)
        self.assertLess(gradient[0,1], 0)
        np.testing.assert_array_equal(gradient[1], 0)

    def test_feasible_step_and_invalid_initial(self):
        barrier = PlaneBarrier([0,1,0], 0, .001, 100., [1.])
        x = np.array([[0.,.0004,0]])
        direction = np.array([[0.,-.002,0]])
        self.assertAlmostEqual(barrier.feasible_step(x, direction), .18)
        with self.assertRaises(ValueError):
            barrier.energy_gradient(-x)

    def test_loaded_tetrahedron_plane_equilibrium(self):
        rest = .01*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        elastic = ElasticReference(rest, [[0,1,2,3]], 12000, .3)
        barrier = PlaneBarrier([0,0,-1], -.0105, .0005, 100., [0,0,0,1])
        fixed = np.ones_like(rest, bool); fixed[3,2] = False
        loads = np.zeros_like(rest); loads[3,2] = .1
        result = elastic.solve(rest, loads, fixed, method='lbfgs', contact=barrier)
        self.assertEqual(result['status'], 'converged')
        self.assertGreater(result['positions'][3,2], rest[3,2])
        self.assertLess(result['positions'][3,2], .0105)
        _, contact_gradient = barrier.energy_gradient(result['positions'])
        self.assertGreater(contact_gradient[3,2], 0)
        total = result['constraint_reaction_n'].sum(0)+loads.sum(0)-contact_gradient.sum(0)
        np.testing.assert_allclose(total, 0, atol=2e-7)
        self.assertTrue(np.all(np.diff([v['energy_j'] for v in result['history']]) <= 1e-15))


if __name__ == '__main__':
    unittest.main()
