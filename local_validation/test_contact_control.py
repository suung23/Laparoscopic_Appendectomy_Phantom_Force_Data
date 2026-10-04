import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from contact_control import ContactGuard


class ContactFeedback(unittest.TestCase):
    def test_gravity_subtraction_preserves_external_overload(self):
        g=ContactGuard(np.zeros(6),0)
        gravity=np.array([.0002,-.00031])
        g.update(np.zeros(6),gravity,jaw_gravity=gravity)
        self.assertFalse(g.pinch_latched)
        g.update(np.zeros(6),gravity+[0,-.0004],jaw_gravity=gravity)
        self.assertTrue(g.pinch_latched)
        np.testing.assert_allclose(g.last['jaw_corrected_effort_nm'],[0,-.0004])
        np.testing.assert_allclose(g.last['jaw_effort_nm'],gravity+[0,-.0004])

    def test_overload_latches_and_sensor_loss_fails_closed(self):
        g=ContactGuard(np.zeros(6),.001)
        for _ in range(40):g.update([.3,0,0,0,0,0],[0,0])
        self.assertTrue(g.latched)
        for _ in range(40):g.update(np.zeros(6),[0,0])
        self.assertTrue(g.latched)
        with self.assertRaises(RuntimeError):g.update([np.nan]*6,[0,0])

    def test_resisted_close_cannot_wind_up_and_open_remains_available(self):
        g=ContactGuard(np.zeros(6),.001)
        g.update(np.zeros(6),[.003,-.003])
        self.assertEqual(g.jaw_target(0,20,30,1/480),30.25)
        g.update(np.zeros(6),[0,0])
        self.assertEqual(g.jaw_target(0,30.25,30,1/480),30.25)
        self.assertGreater(g.jaw_target(74,30,30,1/480),30)
        g.update(np.zeros(6),[0,0])
        self.assertGreaterEqual(g.jaw_target(0,20,30,1/480),29)

    def test_cutter_limit_applies_even_when_load_guard_blocks_closure(self):
        g=ContactGuard(np.zeros(6),0,max_open_deg=50)
        g.update(np.zeros(6),[.001,0])
        self.assertEqual(g.jaw_target(0,50,50,1/480),50)
        self.assertEqual(g.jaw_target(74,50,50,1/480),50)

    def test_noise_floor_not_misreported_as_contact(self):
        g=ContactGuard(np.ones(6),.04)
        for _ in range(40):g.update([1.1,1,1,1,1,1],[0,0])
        self.assertFalse(g.latched)
        self.assertAlmostEqual(g.threshold,.32)


if __name__=='__main__':unittest.main()
