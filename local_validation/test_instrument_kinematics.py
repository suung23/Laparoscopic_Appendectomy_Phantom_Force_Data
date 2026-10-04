import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from instrument_kinematics import PortInstrument


class InstrumentTests(unittest.TestCase):
    def test_pivot_and_cone_remain_valid_under_random_actions(self):
        instrument=PortInstrument([.1,.2,.3])
        rng=np.random.default_rng(7)
        for _ in range(1000):
            pose=instrument.step(rng.uniform(-1,1,5),.05)
            axis=pose['axes'][:,2]
            self.assertLess(np.linalg.norm(np.cross(pose['position_m']-instrument.port,axis)),1e-10)
            self.assertLessEqual(np.arccos(np.clip(axis[2],-1,1)),instrument.limits.tilt_rad+1e-8)
            self.assertAlmostEqual(np.linalg.det(pose['axes']),1)

    def test_invalid_action_does_not_mutate_state(self):
        instrument=PortInstrument([0,0,0])
        before=instrument.state.copy()
        with self.assertRaises(ValueError):instrument.step([0,0,float('nan'),0,0],.01)
        np.testing.assert_array_equal(before,instrument.state)

    def test_limits_and_reset(self):
        instrument=PortInstrument([0,0,0])
        for _ in range(300):instrument.step([1,1,1,1,-1],.1)
        self.assertEqual(instrument.state[2],instrument.limits.insertion_max_m)
        self.assertEqual(instrument.state[4],0)
        instrument.reset()
        self.assertEqual(instrument.state[2],instrument.limits.insertion_min_m)


if __name__=='__main__':unittest.main()
