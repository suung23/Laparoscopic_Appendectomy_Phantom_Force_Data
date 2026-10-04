import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from reset_comparison import compare_states


class ResetComparison(unittest.TestCase):
    def setUp(self):
        p=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]])
        self.a=dict(tissue=p.copy(),rest=p.copy(),tets=np.array([[0,1,2,3]]),joints=np.zeros((2,7)))

    def test_node_order_only_is_not_physical_drift(self):
        order=np.array([2,0,3,1]);inverse=np.argsort(order)
        b={k:v.copy() for k,v in self.a.items()}
        b.update(tissue=b['tissue'][order],rest=b['rest'][order],tets=inverse[b['tets']])
        self.assertEqual(compare_states(self.a,b)['status'],'matched')

    def test_translation_is_not_fitted_away(self):
        b={k:v.copy() for k,v in self.a.items()};b['tissue'][:,0]+=.001
        self.assertEqual(compare_states(self.a,b)['status'],'mismatched')

    def test_nonfinite_state_rejected(self):
        b={k:v.copy() for k,v in self.a.items()};b['joints'][0,0]=np.nan
        with self.assertRaises(ValueError):compare_states(self.a,b)


if __name__=='__main__':unittest.main()
