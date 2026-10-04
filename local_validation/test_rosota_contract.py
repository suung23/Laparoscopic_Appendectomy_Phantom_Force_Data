import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from rosota_controller import aperture_metrics, check_chunk, STRICT_CFG, trocar
from instrument_kinematics import PortInstrument


class TrocarContract(unittest.TestCase):
    def test_centered_rcm_is_not_aperture_clearance(self):
        axis=np.array([np.sin(np.pi/3),0,np.cos(np.pi/3)])
        m=aperture_metrics(np.zeros(3),[0,0,1],axis*.1,axis)
        self.assertLess(m['offset_m'],1e-12)
        self.assertAlmostEqual(m['bite_m'],.002,places=10)
        self.assertFalse(m['aperture_clear'])

    def test_strict_cone_reserves_geometry_budget(self):
        arm=PortInstrument([0,0,0])
        for _ in range(100): arm.step([1,1,0,1,0],.1)
        m=arm.trocar_metrics()
        self.assertTrue(m['aperture_clear'])
        self.assertLessEqual(m['bite_m'],-.00015+1e-9)
        self.assertEqual(STRICT_CFG.tilt_cap_deg,0)
        self.assertAlmostEqual(arm.limits.tilt_rad,trocar.max_tilt_rad(.00015))

    def test_lateral_error_and_outward_axis_are_detected(self):
        self.assertFalse(aperture_metrics([0,0,0],[0,0,1],[.0006,0,.1],[0,0,1])['aperture_clear'])
        self.assertFalse(aperture_metrics([0,0,0],[0,0,1],[0,0,-.1],[0,0,-1])['aperture_clear'])
        with self.assertRaises(ValueError): aperture_metrics([0,0,0],[0,0,0],[0,0,.1],[0,0,1])

    def test_chunk_validation_checks_both_arms_and_all_samples(self):
        good=np.zeros((2,8,7)); good[:,:,6]=100
        out=check_chunk(good); out[0,0,0]=1
        self.assertEqual(good[0,0,0],0)
        for bad in (np.zeros((2,0,7)), np.zeros((2,9,7)), np.zeros(16)):
            with self.assertRaises(ValueError): check_chunk(bad)
        for value in (float('nan'),-1,101):
            bad=good.copy();bad[1,7,6]=value
            with self.assertRaises(ValueError): check_chunk(bad)


if __name__=='__main__':unittest.main()
