import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from independent_batch_lbfgs import minimize

class IndependentTests(unittest.TestCase):
    def test_quadratic_solution_and_batch_permutation(self):
        target=torch.tensor([[1.,2.],[-3.,.5],[.01,7.]],dtype=torch.float64)
        weights=torch.tensor([[1.,2.],[3.,4.],[5.,1.]],dtype=torch.float64)
        def solve(order):
            t=target[order];w=weights[order]
            def objective(x):
                d=x-t;return .5*(w*d*d).sum(1),w*d
            return minimize(objective,torch.zeros_like(t),1e-9,100)[0]
        answer=solve([0,1,2]);torch.testing.assert_close(answer,target,atol=1e-8,rtol=0)
        torch.testing.assert_close(solve([2,0,1]),answer[[2,0,1]],atol=1e-12,rtol=0)
        for i in range(3):torch.testing.assert_close(solve([i])[0],answer[i],atol=1e-8,rtol=0)

if __name__=='__main__':unittest.main()
