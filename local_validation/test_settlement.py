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

    def test_return_to_same_position_does_not_hide_intermediate_motion(self):
        p=np.zeros((20,3));g=SettlementGate(p,require_velocity_peak=True)
        for _ in range(6):self.assertFalse(g.update(p,np.full(20,.01)))
        self.assertEqual(g.rows[-1]['rms_speed_m_s'],0)
        self.assertEqual(g.rows[-1]['stable_windows'],0)
        for _ in range(3):self.assertFalse(g.update(p,np.zeros(20)))
        self.assertTrue(g.update(p,np.zeros(20)))

    def test_missing_or_invalid_velocity_evidence_rejected(self):
        p=np.zeros((20,3));g=SettlementGate(p,require_velocity_peak=True)
        for peak in [None,np.zeros(19),np.full(20,np.nan),np.full(20,-1.)]:
            with self.assertRaises(ValueError):g.update(p,peak)
        self.assertEqual(g.streak,0)


if __name__=='__main__':unittest.main()
