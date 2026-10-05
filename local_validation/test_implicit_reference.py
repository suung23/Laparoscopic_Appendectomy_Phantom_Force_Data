import unittest
from unittest.mock import patch
import numpy as np
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.implicit_reference import InertialPotential, implicit_step


class DynamicsTests(unittest.TestCase):
    def test_near_equilibrium_newton_refinement_reaches_original_tolerance(self):
        x = self.x.copy(); dt = .002
        candidate = x.copy(); candidate[3,2] += 1e-13
        potential = InertialPotential(x,self.mass,dt)
        _,g = self.model.energy_gradient(candidate,x*0)
        _,pg = potential.energy_gradient(candidate)
        failed = dict(status='iteration_limit',positions=candidate,
            residual_n=float(np.abs(g+pg).max()),history=[],constraint_reaction_n=x*0)
        with patch.object(self.model,'solve',return_value=failed):
            result = implicit_step(self.model,x,x*0,self.mass,x*0,dt)
        self.assertEqual(result['status'],'converged')
        self.assertLessEqual(result['residual_n'],1e-11)
        self.assertEqual(len(result['newton_refinements']),1)
        np.testing.assert_array_equal(x,self.x)

    def setUp(self):
        self.x = .01*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.model = ElasticReference(self.x,[[0,1,2,3]],12000,.3)
        self.mass = np.full(4,1050*self.model.volume[0]/4)

    def test_freefall_preserves_shape_and_matches_backward_euler(self):
        x = self.x.copy(); v = x*0; gravity = np.array([0.,-9.81,0]); dt = .002
        for _ in range(5):
            result = implicit_step(self.model,x,v,self.mass,self.mass[:,None]*gravity,dt,tolerance=1e-10)
            expected_v = v+dt*gravity
            np.testing.assert_allclose(result['velocity'],expected_v,atol=1e-8)
            np.testing.assert_allclose(result['positions'],x+dt*expected_v,atol=1e-10)
            x,v = result['positions'],result['velocity']

    def test_failed_step_does_not_modify_state(self):
        x = self.x.copy(); v = x*0
        with self.assertRaises(RuntimeError):
            implicit_step(self.model,x,v,self.mass,np.ones_like(x),.01,max_iterations=1,tolerance=1e-14)
        np.testing.assert_array_equal(x,self.x)
        np.testing.assert_array_equal(v,0)

    def test_inertia_hessian(self):
        potential = InertialPotential(self.x,self.mass,.01)
        d = np.random.default_rng(43).normal(size=self.x.shape)
        h = 1e-8
        numerical = (potential.energy_gradient(self.x+h*d)[1]-potential.energy_gradient(self.x-h*d)[1])/(2*h)
        np.testing.assert_allclose((potential.hessian(self.x)@d.ravel()).reshape(d.shape),numerical,rtol=1e-8)

if __name__ == '__main__': unittest.main()
