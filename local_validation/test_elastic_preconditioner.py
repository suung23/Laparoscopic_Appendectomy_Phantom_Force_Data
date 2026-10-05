import unittest
import numpy as np
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.elastic_preconditioner import rest_stiffness
from ct_appendectomy.plane_barrier import PlaneBarrier


class HessianTests(unittest.TestCase):
    def test_rest_hessian_matches_gradient_derivative(self):
        x = .01*np.array([[0.,0,0],[1,.1,0],[.1,1,.1],[0,.1,1]])
        model = ElasticReference(x, [[0,1,2,3]], 12000, .45)
        direction = np.random.default_rng(42).normal(size=x.shape)
        h = 1e-8
        numerical = (model.energy_gradient(x+h*direction,x*0)[1]-model.energy_gradient(x-h*direction,x*0)[1])/(2*h)
        matrix = rest_stiffness(model)
        np.testing.assert_allclose((matrix@direction.ravel()).reshape(x.shape),numerical,rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(matrix.toarray(),matrix.toarray().T,atol=1e-12)
        np.testing.assert_allclose(matrix@np.ones(x.size),0,atol=1e-12)

    def test_contact_hessian_matches_gradient_derivative(self):
        barrier = PlaneBarrier([0,1,0],0,.001,100.,[1.])
        x = np.array([[0.,.0004,0]]); d = np.array([[.1,.2,.3]])
        h = 1e-9
        numerical = (barrier.energy_gradient(x+h*d)[1]-barrier.energy_gradient(x-h*d)[1])/(2*h)
        np.testing.assert_allclose((barrier.hessian(x)@d.ravel()).reshape(x.shape),numerical,rtol=1e-7,atol=1e-7)

    def test_preconditioned_solve(self):
        x = .01*np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        model = ElasticReference(x,[[0,1,2,3]],12000,.3)
        fixed = np.ones_like(x,bool); fixed[3,2] = False
        loads = x*0; loads[3,2] = -.03
        result = model.solve(x,loads,fixed,method='rest_preconditioned')
        self.assertEqual(result['status'],'converged')
        np.testing.assert_array_equal(result['positions'][fixed],x[fixed])

if __name__ == '__main__': unittest.main()
