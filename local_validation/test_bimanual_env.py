import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from bimanual_env import BimanualEnv


class EpisodeContract(unittest.TestCase):
    def make(self, achieved=False, failure='', steps=0):
        env = object.__new__(BimanualEnv)
        env.done=False; env.steps=steps; env.max_steps=40; env.hold=0
        env.active_task='retract'; env.patches=np.array([1,5]); env.lift=.006; env.potential=-1.
        env.calls=[]
        env.control=SimpleNamespace(step_action=lambda *a, **kw: env.calls.append('action'), observation=lambda: np.zeros(12))
        env.sim=SimpleNamespace(step=lambda **kw: env.calls.append('physics'))
        env._metrics=lambda: dict(error=np.zeros((2,3)), delta=np.array([[0,.006,0]]*2), owned=np.ones(2,bool),
            distance=np.zeros(2), achieved=achieved, failure=failure, pin_drift_m=0., health={})
        return env

    def test_invalid_action_does_not_mutate(self):
        for action in [np.zeros(9), np.full(10,np.nan), np.full(10,1.01)]:
            e=self.make()
            with self.assertRaises(ValueError): e.step(action)
            self.assertEqual(e.calls,[]); self.assertEqual(e.steps,0)

    def test_success_requires_consecutive_hold(self):
        e=self.make(True)
        for _ in range(29): self.assertFalse(e.step(np.zeros(10))[2])
        obs,reward,term,trunc,info=e.step(np.zeros(10))
        self.assertTrue(term and info['success']); self.assertFalse(trunc)
        self.assertEqual(obs.shape,(31,)); self.assertTrue(np.isfinite(reward))
        with self.assertRaises(RuntimeError): e.step(np.zeros(10))

    def test_timeout_bootstraps_and_failure_dominates_success(self):
        e=self.make(steps=39)
        _,_,term,trunc,info=e.step(np.zeros(10))
        self.assertTrue(trunc); self.assertFalse(term or info['success'])
        e=self.make(True,'tet_inversion'); e.hold=29
        _,reward,term,trunc,info=e.step(np.zeros(10))
        self.assertTrue(term); self.assertFalse(trunc or info['success']); self.assertLess(reward,0)

    def test_pregrasp_target_is_outside_tissue_and_grasp_requires_ownership(self):
        e=self.make()
        del e._metrics
        e.rest=np.array([[0,0,0],[.001,0,0],[0,.001,0],[0,0,.001]])
        e.reference=e.rest.copy(); e.tets=np.array([[0,1,2,3]])
        e.patches=np.array([0,1]); e.patch_reference=np.array([[0,0,0],[.02,0,0]])
        e.active_task='reach'
        arms={}
        for h,x in [('Left',0.),('Right',.02)]:
            position=np.array([x,.018,0])
            arms[h]=SimpleNamespace(port=np.array([x,.1,0]),pose=lambda pos=position: {'position_m':pos})
        e.control=SimpleNamespace(points=lambda:e.rest.copy(),centers=lambda:e.patch_reference.copy(),
                                  arms=arms,held={},data={'pin':np.array([0])})
        self.assertTrue(e._metrics()['achieved'])
        e.active_task='dual_grasp'
        self.assertFalse(e._metrics()['achieved'])


if __name__=='__main__': unittest.main()
