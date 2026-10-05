import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from rosota_ct_env import RosotaCTEnv

class ClockPlant:
 def __init__(self,dt):self.dt=dt;self.time=0.;self.position=0.
 def step(self,render=False):self.time+=self.dt;self.position+=2*self.dt

class SubstepTests(unittest.TestCase):
 def test_action_duration_and_motion_preserved_at_every_resolution(self):
  for n in [1,2,4]:
   env=RosotaCTEnv.__new__(RosotaCTEnv);env.physics_substeps=n;env.sim=ClockPlant(1/(480*n));samples=[]
   for _ in range(16):env._physics_step(lambda:samples.append(env.sim.time))
   self.assertAlmostEqual(env.sim.time,1/30)
   self.assertAlmostEqual(env.sim.position,2/30)
   self.assertEqual(len(samples),16*n)
   self.assertAlmostEqual(samples[0],1/(480*n))
 def test_invalid_substep_counts_fail_before_scene_loading(self):
  for n in [0,3,True]:
   with self.assertRaises(ValueError):RosotaCTEnv(None,None,None,None,physics_substeps=n)

if __name__=='__main__':unittest.main()
