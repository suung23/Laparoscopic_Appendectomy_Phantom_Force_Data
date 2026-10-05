"""Probe exact Newton correction at retained failed step, without advancing it."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
from ct_appendectomy.constrained_elasticity import ElasticReference
from ct_appendectomy.implicit_reference import InertialPotential
from ct_appendectomy.plane_barrier import PlaneBarrier
from ct_appendectomy.newton_direction import newton_direction
from ct_appendectomy.determinant_step import positive_volume_step

folder=root/'out/ct_appendectomy/anatomical_implicit_contact_stable_energy'
state=np.load(folder/'last_accepted_state.npz'); candidate=folder/'unaccepted_iterate.npz'
rest,tets=state['rest'],state['tets']; x=np.load(candidate)['points']
elastic=ElasticReference(rest,tets,12000.,.45)
mass=np.zeros(len(rest)); weights=mass.copy()
for i in range(4): np.add.at(mass,tets[:,i],1050*elastic.volume/4)
faces=np.concatenate([tets[:,f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
unique,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
surface=unique[counts==1]; q=rest[surface]
area=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2
for i in range(3): np.add.at(weights,surface[:,i],area/3)
plane=PlaneBarrier([0,1,0],float(rest[:,1].min()-.0008),.0008,1e8,weights)
dt=1/480
potential=InertialPotential(state['points']+dt*state['velocity'],mass,dt,plane)
loads=np.zeros_like(rest); loads[:,1]=-9.81*mass
def evaluate(points):
    e,g=elastic.energy_gradient(points,loads); pe,pg=potential.energy_gradient(points)
    return e+pe,g+pg
energy,gradient=evaluate(x)
direction,diagnostic=newton_direction(elastic,x,gradient,potential)
alpha=min(positive_volume_step(rest,x,direction,tets)['alpha'],potential.feasible_step(x,direction))
trial,after=evaluate(x+alpha*direction)
report=dict(scope='Single unaccepted Newton correction diagnostic; does not advance dynamics',
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    initial_residual_n=float(np.abs(gradient).max()),corrected_residual_n=float(np.abs(after).max()),
    alpha=alpha,energy_change_j=trial-energy,diagnostic=diagnostic,
    minimum_gap_m=float(plane.gaps(x+alpha*direction).min()),
    force_tolerance_passed=bool(np.abs(after).max()<=1e-11))
(root/'validation/failed-step-newton-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
