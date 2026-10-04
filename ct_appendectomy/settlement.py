"""Measured nodal-motion reset gate; a fixed wait is not evidence of equilibrium."""
import numpy as np


class SettlementGate:
    def __init__(self,initial,interval_s=.125,rms_limit_m_s=.0001,p99_limit_m_s=.0005,consecutive=4):
        self.previous=np.asarray(initial,float).copy()
        if self.previous.ndim!=2 or self.previous.shape[1]!=3 or not len(self.previous) or not np.isfinite(self.previous).all():
            raise ValueError('Finite XYZ node array required')
        if (not np.isfinite([interval_s,rms_limit_m_s,p99_limit_m_s]).all() or
            interval_s<=0 or min(rms_limit_m_s,p99_limit_m_s)<=0 or
            not isinstance(consecutive,int) or consecutive<1):
            raise ValueError('Positive settlement parameters required')
        self.dt=interval_s;self.rms_limit=rms_limit_m_s;self.p99_limit=p99_limit_m_s
        self.required=consecutive;self.streak=0;self.rows=[]

    def update(self,current):
        q=np.asarray(current,float)
        if q.shape!=self.previous.shape or not np.isfinite(q).all():raise ValueError('Invalid settlement state')
        speed=np.linalg.norm(q-self.previous,axis=1)/self.dt
        rms=float(np.sqrt(np.mean(speed**2)));p99=float(np.percentile(speed,99))
        stable=rms<=self.rms_limit and p99<=self.p99_limit
        self.streak=self.streak+1 if stable else 0
        self.previous=q.copy()
        self.rows.append(dict(rms_speed_m_s=rms,p99_speed_m_s=p99,max_speed_m_s=float(speed.max()),
                              stable_windows=self.streak))
        return self.streak>=self.required
