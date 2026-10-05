import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from tissue_metrics import step_velocity_comparison

class VelocityComparisonTests(unittest.TestCase):
 def test_constant_velocity_matches(self):
  p=np.zeros((4,3));v=np.tile([.1,-.2,.3],(4,1));mean,gap=step_velocity_comparison(p,p+v*.01,v,.01)
  np.testing.assert_allclose(mean,v);self.assertAlmostEqual(gap,0)
 def test_stationary_positions_expose_nonzero_reported_velocity(self):
  p=np.zeros((4,3));mean,gap=step_velocity_comparison(p,p,np.tile([1,0,0],(4,1)),.01)
  self.assertEqual(gap,1);self.assertFalse(mean.any())
 def test_acceleration_is_a_gap_not_automatically_an_error(self):
  p=np.zeros((4,3));v=np.tile([.2,0,0],(4,1));mean,gap=step_velocity_comparison(p,p+v*.005,v,.01)
  self.assertAlmostEqual(gap,.1)
 def test_invalid_time_rejected(self):
  p=np.zeros((4,3))
  with self.assertRaises(ValueError):step_velocity_comparison(p,p,p,0)

if __name__=='__main__':unittest.main()
