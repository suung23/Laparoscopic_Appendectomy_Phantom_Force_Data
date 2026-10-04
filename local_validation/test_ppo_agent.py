import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from ppo_agent import ActorCritic,advantages,update,save_checkpoint,load_checkpoint


class PPOTests(unittest.TestCase):
    def test_truncation_bootstraps_without_cross_episode_leak(self):
        adv,_=advantages([1.,100.],[.5,0.],[2.,0.],[False,True],[True,True])
        self.assertAlmostEqual(float(adv[0]),1+.99*2-.5,places=6)
        terminal,_=advantages([1.],[.5],[2.],[True],[True])
        self.assertAlmostEqual(float(terminal[0]),.5)

    def test_real_gradient_and_safe_checkpoint_roundtrip(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=ActorCritic();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        observations=torch.randn(32,14)
        with torch.no_grad():action,raw,logp,value,_=model.evaluate(observations)
        self.assertTrue(torch.all(torch.abs(action)<=1))
        before=torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        metrics=update(model,optimizer,{'obs':observations.numpy(),'raw':raw.numpy(),
            'log_prob':logp.numpy(),'advantage':np.linspace(-1,1,32),'return':np.ones(32)},epochs=2)
        after=torch.cat([p.detach().flatten() for p in model.parameters()])
        self.assertGreater(float(torch.linalg.vector_norm(after-before)),0)
        self.assertTrue(np.isfinite(metrics['loss_mean']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt'
            save_checkpoint(path,model,optimizer,{'env_id':'test','trained_steps':32})
            restored=ActorCritic()
            load_checkpoint(path,restored,expected={'env_id':'test'})
            for one,two in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(one,two)
            with self.assertRaises(ValueError):load_checkpoint(path,restored,expected={'env_id':'wrong'})


if __name__=='__main__':unittest.main()
