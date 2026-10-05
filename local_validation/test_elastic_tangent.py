import unittest
import numpy as np
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.elastic_preconditioner import rest_stiffness


class TangentTests(unittest.TestCase):
    def setUp(self):
        self.x = .01*np.array([[0.,0,0],[1,.1,0],[.1,1,.1],[0,.1,1]])
        self.model = ElasticReference(self.x,[[0,1,2,3]],12000,.45)
        self.rng = np.random.default_rng(153)

    def test_deformed_gradient_derivative(self):
        x = self.x@np.array([[.8,.2,0],[0,1.2,.1],[.1,0,.9]])
        direction = self.rng.normal(size=x.shape)
        h = 1e-8
        numerical = (self.model.energy_gradient(x+h*direction,x*0)[1]-self.model.energy_gradient(x-h*direction,x*0)[1])/(2*h)
        np.testing.assert_allclose(self.model.hessian_vector(x,direction),numerical,rtol=1e-7,atol=1e-6)

    def test_tangent_symmetry_and_translation_nullspace(self):
        x = self.x@np.diag([.8,1.1,.9])
        a,b = self.rng.normal(size=(2,*x.shape))
        self.assertAlmostEqual(float(np.sum(a*self.model.hessian_vector(x,b))),float(np.sum(b*self.model.hessian_vector(x,a))),places=10)
        np.testing.assert_allclose(self.model.hessian_vector(x,np.ones_like(x)),0,atol=1e-12)

    def test_rest_matches_assembled_matrix(self):
        direction = self.rng.normal(size=self.x.shape)
        np.testing.assert_allclose(self.model.hessian_vector(self.x,direction).ravel(),rest_stiffness(self.model)@direction.ravel(),rtol=1e-12,atol=1e-10)

if __name__ == '__main__': unittest.main()
