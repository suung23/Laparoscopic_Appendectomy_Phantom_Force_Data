"""Local PPO adapter for the versioned bimanual environment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ppo_agent import ActorCritic, advantages, update, save_checkpoint, load_checkpoint


def train(env, steps, seed, output, checkpoint=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = ActorCritic(observations=31, actions=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    contract = dict(env_id='RosotaBimanual-v1', observations=31, actions=10,
                    task=env.task, stage_sha256=env.config['stage_sha256'],
                    config_sha256=hashlib.sha256((env.assets/'bimanual_config.json').read_bytes()).hexdigest(),
                    environment_sha256=hashlib.sha256(Path(__file__).with_name('bimanual_env.py').read_bytes()).hexdigest(),
                    gamma=.99, grasp_assist=True)
    previous = 0
    if checkpoint:
        metadata = load_checkpoint(checkpoint, model, optimizer, expected=contract)
        previous = metadata['trained_steps']
    records=[]; episodes=[]; total=0; episode=0
    initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    if steps:
        obs, _ = env.reset(seed=seed)
        while total < steps:
            batch={key:[] for key in ('obs','raw','log_prob','value','next_value','reward','terminated','ended')}
            for _ in range(min(256, steps-total)):
                with torch.no_grad():
                    action,raw,logp,value,_=model.evaluate(torch.as_tensor(obs)[None])
                nxt,reward,term,trunc,info=env.step(action[0].numpy())
                with torch.no_grad(): nv=model.evaluate(torch.as_tensor(nxt)[None], deterministic=True)[3].item()
                values=(obs,raw[0].numpy(),logp.item(),value.item(),nv,reward,term,term or trunc)
                for key,v in zip(batch,values):batch[key].append(v)
                total+=1; obs=nxt
                if term or trunc:
                    episodes.append(dict(episode=episode,step=total,success=info['success'],failure=info['failure']))
                    episode+=1
                    obs,_=env.reset(seed=seed+episode)
            batch['advantage'],batch['return']=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminated'],batch['ended'])
            records.append(dict(step=total,**update(model,optimizer,batch)))
            save_checkpoint(output/'policy.pt',model,optimizer,dict(contract,trained_steps=previous+total))
            delta = float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in model.parameters()])-initial))
            if not np.isfinite(delta) or delta == 0.: raise RuntimeError('PPO did not update parameters')
            (output/'training.json').write_text(json.dumps(dict(contract,seed=seed,trained_steps=previous+total,
                parameter_change_l2=delta,updates=records,episodes=episodes),indent=2))
    return model
