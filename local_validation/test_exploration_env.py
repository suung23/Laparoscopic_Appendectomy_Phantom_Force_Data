import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from exploration_env import ExplorationEnv


class ExplorationContract(unittest.TestCase):
    def env(self):
        e=object.__new__(ExplorationEnv);e.done=False;e.calls=[]
        e.camera=np.zeros(6);e.steps=0;e.hold=0;e.index=0;e.max_steps=100
        e.visited=np.zeros(7,bool);e.goals=np.ones((7,2,3))*.01;e.goals[0]=0
        e.positions=lambda:np.zeros((2,3));e.initial_positions=e.positions()
        e.max_excursion=np.zeros(2);e.initial_camera=np.eye(4);e.camera_changed=False
        e.scope=SimpleNamespace(pose=np.eye(4),apply=lambda *args:e.calls.append('camera'))
        e.base=SimpleNamespace(control=SimpleNamespace(step_action=lambda *a,**kw:e.calls.append('tools'),
            arms={h:SimpleNamespace(trocar_metrics=lambda:{'aperture_clear':True}) for h in ('Left','Right')}),
            sim=SimpleNamespace(step=lambda **kw:e.calls.append('physics')),_metrics=lambda:{'failure':''})
        e._observe=lambda m:np.zeros(42,np.float32);e.potential=e._potential()
        return e

    def test_invalid_full_action_is_atomic(self):
        for action in (np.zeros(10),np.full(16,np.nan),np.full(16,1.1)):
            e=self.env()
            with self.assertRaises(ValueError):e.step(action)
            self.assertEqual(e.calls,[]);self.assertEqual(e.steps,0)

    def test_coverage_is_not_repeatable_reward_and_requires_both_tools(self):
        e=self.env()
        for _ in range(15):e.step(np.zeros(16))
        self.assertEqual(e.visited.sum(),1);self.assertNotEqual(e.index,0)
        for _ in range(15):e.step(np.zeros(16))
        self.assertEqual(e.visited.sum(),1)
        e=self.env();e.positions=lambda:np.array([[0,0,0],[.003,0,0]])
        for _ in range(20):e.step(np.zeros(16))
        self.assertFalse(e.visited.any())

    def test_camera_bounds_and_termination(self):
        e=self.env();e.camera=np.array([35.,30.,90.,.58,.58,.58])
        e.step(np.ones(16))
        self.assertLessEqual(np.linalg.norm(e.camera[3:]),1+1e-12)
        np.testing.assert_array_equal(e.camera[:3],[35,30,90])
        e.base._metrics=lambda:{'failure':'tet_inversion'}
        _,reward,term,trunc,info=e.step(np.zeros(16))
        self.assertTrue(term);self.assertFalse(trunc);self.assertLess(reward,0)
        self.assertFalse(info['success'])
        with self.assertRaises(RuntimeError):e.step(np.zeros(16))


if __name__=='__main__':unittest.main()
