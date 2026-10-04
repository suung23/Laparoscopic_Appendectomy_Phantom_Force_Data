"""Small local PPO implementation for one sequential Isaac environment.

Clipped objective: https://arxiv.org/abs/1707.06347
Tanh Gaussian actions, GAE, distinct termination and truncation bootstrapping.
No network services, model downloads or external experiment trackers.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self,observations=14,actions=2):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(observations,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,actions))
        self.critic=nn.Sequential(nn.Linear(observations,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.log_std=nn.Parameter(torch.full((actions,),-.5))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight,np.sqrt(2));nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)

    def evaluate(self,obs,raw_action=None,deterministic=False):
        obs=torch.clamp(obs,-10,10)
        mean=self.actor(obs)
        distribution=Normal(mean,self.log_std.clamp(-5,1).exp())
        raw=mean if deterministic else (distribution.sample() if raw_action is None else raw_action)
        action=raw.tanh()
        log_prob=(distribution.log_prob(raw)-torch.log(1-action.square()+1e-6)).sum(-1)
        # Monte Carlo transformed-distribution entropy, not Gaussian entropy.
        entropy_sample=distribution.rsample()
        entropy=-(distribution.log_prob(entropy_sample)-torch.log(1-entropy_sample.tanh().square()+1e-6)).sum(-1)
        return action,raw,log_prob,self.critic(obs).squeeze(-1),entropy


def advantages(rewards,values,next_values,terminated,ended,gamma=.99,lam=.95):
    result=np.zeros(len(rewards),np.float32)
    carry=0.
    for i in range(len(rewards)-1,-1,-1):
        delta=rewards[i]+gamma*next_values[i]*(1-float(terminated[i]))-values[i]
        carry=delta+gamma*lam*(1-float(ended[i]))*carry
        result[i]=carry
    return result,result+np.asarray(values,np.float32)


def update(model,optimizer,batch,epochs=4,minibatch=64):
    obs,raw,old_log,adv,returns=[torch.as_tensor(batch[k],dtype=torch.float32) for k in ('obs','raw','log_prob','advantage','return')]
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    losses=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(obs)).split(minibatch):
            _,_,logp,value,entropy=model.evaluate(obs[ids],raw[ids])
            ratio=(logp-old_log[ids]).exp()
            policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
            loss=policy+.5*(value-returns[ids]).square().mean()-.001*entropy.mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite PPO loss')
            optimizer.zero_grad();loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(loss.detach()),float(norm)])
    return {'loss_mean':float(np.mean(losses,axis=0)[0]),'gradient_norm_max':float(np.max(losses,axis=0)[1])}


def save_checkpoint(path,model,optimizer,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'metadata':metadata},temporary)
    temporary.replace(path)


def load_checkpoint(path,model,optimizer=None,expected=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    metadata=checkpoint['metadata']
    for key,value in (expected or {}).items():
        if metadata.get(key)!=value:raise ValueError('Checkpoint contract mismatch: '+key)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:optimizer.load_state_dict(checkpoint['optimizer'])
    return metadata
