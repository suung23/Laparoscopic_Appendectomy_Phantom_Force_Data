import unittest
import numpy as np
from ct_appendectomy.constrained_elasticity import ElasticReference


class ElasticTests(unittest.TestCase):
    def test_small_strain_energy_retains_quadratic_limit(self):
        rest = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        model = ElasticReference(rest,[[0,1,2,3]],12000,.3)
        for epsilon in (1e-5,1e-7,1e-9):
            x = rest.copy(); x[3,2] += epsilon
            energy,_ = model.energy_gradient(x,np.zeros_like(x))
            actual_strain = x[3,2]-1
            expected = .5*(model.lam+2*model.mu)*actual_strain**2/6
            self.assertGreater(energy,0)
            self.assertLess(abs(energy/expected-1),2e-5)

    def setUp(self):
        self.x = .01*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.model = ElasticReference(self.x, [[0,1,2,3]], 12000, .3)

    def test_analytic_force_matches_energy_derivative(self):
        x = self.x @ np.array([[1.1,.05,0],[0,.9,.02],[0,0,1.05]])
        loads = np.full_like(x, .001)
        _, gradient = self.model.energy_gradient(x, loads)
        numerical = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[index] += 1e-8; minus[index] -= 1e-8
            numerical[index] = (self.model.energy_gradient(plus,loads)[0]-self.model.energy_gradient(minus,loads)[0])/2e-8
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-9)

    def test_rigid_motion_has_zero_internal_energy_and_force(self):
        rotation = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        energy, gradient = self.model.energy_gradient(self.x@rotation + .05, self.x*0)
        self.assertAlmostEqual(energy, 0, places=12)
        np.testing.assert_allclose(gradient, 0, atol=1e-12)

    def test_compression_equilibrium_and_unloading(self):
        fixed = np.ones_like(self.x, dtype=bool); fixed[3,2] = False
        loads = self.x*0; loads[3,2] = -.01
        result = self.model.solve(self.x, loads, fixed)
        self.assertEqual(result['status'], 'converged')
        self.assertLess(result['positions'][3,2], self.x[3,2])
        self.assertGreater(result['positions'][3,2], 0)
        np.testing.assert_allclose(result['constraint_reaction_n'].sum(axis=0)+loads.sum(axis=0), 0, atol=2e-7)
        energy = [row['energy_j'] for row in result['history']]
        self.assertTrue(np.all(np.diff(energy) <= 1e-15))
        unloaded = self.model.solve(result['positions'], loads*0, fixed)
        self.assertEqual(unloaded['status'], 'converged')
        np.testing.assert_allclose(unloaded['positions'], self.x, atol=1e-8)

    def test_inversion_rejected(self):
        x = self.x.copy(); x[3,2] *= -1
        with self.assertRaises(ValueError):
            self.model.energy_gradient(x, x*0)

    def test_lbfgs_agrees_with_gradient_equilibrium(self):
        fixed = np.ones_like(self.x, dtype=bool); fixed[3,2] = False
        loads = self.x*0; loads[3,2] = -.03
        reference = self.model.solve(self.x, loads, fixed)
        result = self.model.solve(self.x, loads, fixed, method='lbfgs')
        self.assertEqual(reference['status'], 'converged')
        self.assertEqual(result['status'], 'converged')
        np.testing.assert_allclose(result['positions'], reference['positions'], atol=1e-8)
        self.assertTrue(np.all(np.diff([v['energy_j'] for v in result['history']]) <= 1e-15))

    def test_multicell_affine_patch_interior_equilibrium(self):
        from local_validation.regular_tet_control import build
        rest, tets = build(n=3)
        model = ElasticReference(rest, tets, 12000, .3)
        target = rest @ np.array([[1.02,.03,0],[0,.98,0],[0,0,.9]])
        boundary = np.any((rest == 0) | (rest == rest.max()), axis=1)
        fixed = np.repeat(boundary[:,None], 3, axis=1)
        initial = target.copy(); initial[~boundary] += [.0001,-.0001,.0001]
        result = model.solve(initial, rest*0, fixed)
        self.assertEqual(result['status'], 'converged')
        np.testing.assert_allclose(result['positions'], target, atol=1e-8)
        np.testing.assert_allclose(result['positions'][boundary], initial[boundary], atol=0)


if __name__ == '__main__':
    unittest.main()
