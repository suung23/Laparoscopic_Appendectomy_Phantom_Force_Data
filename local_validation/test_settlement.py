import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from settlement import SettlementGate


class SettlementTests(unittest.TestCase):
    def test_rigid_drift_is_not_equilibrium(self):
        p=np.zeros((20,3));g=SettlementGate(p)
        for i in range(1,7):self.assertFalse(g.update(p+[i*.001,0,0]))

    def test_requires_consecutive_stable_windows(self):
        p=np.zeros((20,3));g=SettlementGate(p)
        for _ in range(3):self.assertFalse(g.update(p))
        self.assertTrue(g.update(p))
        self.assertFalse(g.update(p+.001))
        for _ in range(3):self.assertFalse(g.update(p+.001))
        self.assertTrue(g.update(p+.001))

    def test_invalid_state_rejected(self):
        g=SettlementGate(np.zeros((20,3)))
        with self.assertRaises(ValueError):g.update(np.full((20,3),np.nan))


if __name__=='__main__':unittest.main()
