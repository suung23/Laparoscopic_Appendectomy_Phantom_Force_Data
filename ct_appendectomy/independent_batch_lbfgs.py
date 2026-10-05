"""Batched L-BFGS with independent histories and Armijo steps per environment."""
import torch

def minimize(objective,initial,tolerance,maxiter=1000,history_size=10):
    x=initial.detach().clone();value,gradient=objective(x);history=[];calls=1
    dot=lambda a,b:(a*b).flatten(1).sum(1)
    expand=lambda a:a.reshape((-1,)+(1,)*(x.ndim-1))
    iterations=torch.zeros(len(x),device=x.device,dtype=torch.int64)
    for _ in range(maxiter):
        active=gradient.abs().flatten(1).amax(1)>tolerance
        if not active.any():break
        q=gradient.clone();alphas=[]
        for s,y,rho in reversed(history):
            alpha=rho*dot(s,q);alphas.append(alpha);q=q-expand(alpha)*y
        if history:
            s,y,rho=history[-1];yy=dot(y,y)
            gamma=torch.where(rho>0,dot(s,y)/yy.clamp_min(1e-30),torch.ones_like(yy))
            q=q*expand(gamma.clamp(1e-8,1e8))
        for (s,y,rho),alpha in zip(history,reversed(alphas)):
            beta=rho*dot(y,q);q=q+s*expand(alpha-beta)
        direction=-q;slope=dot(gradient,direction)
        fallback=slope>=0
        direction=torch.where(expand(fallback),-gradient,direction);slope=dot(gradient,direction)
        direction=torch.where(expand(active),direction,torch.zeros_like(direction))
        step=torch.ones(len(x),device=x.device,dtype=x.dtype);accepted=~active
        trial_x=x.clone();trial_value=value.clone();trial_gradient=gradient.clone()
        for _ in range(30):
            candidate=x+expand(step)*direction
            v,g=objective(candidate);calls+=1
            good=(v<=value+1e-4*step*slope)&torch.isfinite(v)&torch.isfinite(g).flatten(1).all(1)
            newly=good&~accepted
            trial_x=torch.where(expand(newly),candidate,trial_x)
            trial_value=torch.where(newly,v,trial_value);trial_gradient=torch.where(expand(newly),g,trial_gradient)
            accepted|=good
            if accepted.all():break
            step=torch.where(accepted,step,step*.5)
        if not accepted.all():break
        s=trial_x-x;y=trial_gradient-gradient;sy=dot(s,y)
        valid=(sy>1e-12*torch.sqrt(dot(s,s)*dot(y,y)))&active
        rho=torch.where(valid,1/sy.clamp_min(1e-30),torch.zeros_like(sy))
        history.append((s,y,rho));history=history[-history_size:]
        x,value,gradient=trial_x,trial_value,trial_gradient;iterations+=active
    return x,{'objective_calls':calls,'iterations_per_environment':iterations.cpu().tolist()}
