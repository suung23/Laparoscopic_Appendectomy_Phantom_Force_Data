import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from rosota_ct_env import RosotaCTEnv


class ActionValidation(unittest.TestCase):
    def test_invalid_second_hand_or_scope_is_rejected_before_any_plant_access(self):
        e=object.__new__(RosotaCTEnv);e.done=False
        # Deliberately no plant/controller: validation must finish before any access.
        for field,index,value in [('tools',(1,0,6),101),('tools',(1,0,2),np.nan),('scope',2,1.1)]:
            a={'tools':np.zeros((2,1,7)),'scope':np.zeros(4)}
            a[field][index]=value
            with self.assertRaises(ValueError):e.step(a)
            self.assertFalse(e.done)
        with self.assertRaises(ValueError):e.step({'tools':np.zeros((2,1,7))})

    def test_finished_env_rejects_action(self):
        e=object.__new__(RosotaCTEnv);e.done=True
        with self.assertRaises(RuntimeError):e.step({'tools':np.zeros((2,1,7)),'scope':np.zeros(4)})


if __name__=='__main__':unittest.main()
