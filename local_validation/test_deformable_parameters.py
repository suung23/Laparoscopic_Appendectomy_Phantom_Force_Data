import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from deformable_parameters import validate_solver_iterations

class ParameterBounds(unittest.TestCase):
 def test_supported_boundary_and_operating_values(self):
  for n in [1,16,64,128,255]:self.assertEqual(validate_solver_iterations(n),n)
 def test_invalid_values_are_not_clamped_or_silently_accepted(self):
  for n in [0,256,-1,1.5,True,None,float('nan')]:
   with self.subTest(value=n),self.assertRaises(ValueError):validate_solver_iterations(n)

if __name__=='__main__':unittest.main()
