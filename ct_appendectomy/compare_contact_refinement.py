"""Compare matched-time membrane states across completed reference runs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def compare(runs,out):
    records=[];states=[];pairs=[];configuration=None
    for root in runs:
        path=root/'report.json';r=json.loads(path.read_text());data=np.load(root/'trajectory.npz')
        if r['status']!='completed reference run':raise ValueError('Incomplete run: '+str(root))
        if len(data['points'])!=len(r['steps'])+1:raise ValueError('Trajectory/report length mismatch')
        if not np.isclose(float(data['dt']),r['dt'],rtol=0,atol=1e-12):raise ValueError('Trajectory/report timestep mismatch')
        settings={key:r.get(key) for key in ('friction_mu','full_tool_contact','contact_stiffness_N_per_m3','articulated_jaw_trajectory','artery_shape_regularization')}
        settings['solver_tolerance_N']=r.get('solver_tolerance_N',r['steps'][0]['tolerance_N'])
        settings['inertial_defect_tolerance_m']=r.get('inertial_defect_tolerance_m')
        if configuration is not None and settings!=configuration:raise ValueError('Reference contact settings differ')
        configuration=settings
        records.append({'dt':r['dt'],'duration':r['simulated_seconds'],'wall_seconds':r['wall_seconds'],
                        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'trajectory_sha256':hashlib.sha256((root/'trajectory.npz').read_bytes()).hexdigest(),
                        'peak_reaction_N':max(float(np.linalg.norm(s['contact_before']['reaction_N'])) for s in r['steps']),
                        'maximum_solver_residual_N':max(s['residual_N'] for s in r['steps']),
                        'peak_layer_motion_mm':max(s['layer_max_motion_mm'] for s in r['steps'])})
        states.append(data['points'])
    for i in range(len(runs)-1):
        a,b=records[i:i+2];ratio=a['dt']/b['dt'];stride=int(round(ratio))
        if stride<2 or not np.isclose(ratio,stride) or not np.isclose(a['duration'],b['duration']):raise ValueError('Runs must have integer timestep refinement and equal duration')
        aligned=states[i+1][::stride]
        if aligned.shape!=states[i].shape:raise ValueError('Matched-state count mismatch')
        distance=np.linalg.norm(aligned-states[i],axis=2)*1000
        pairs.append({'coarse_dt':a['dt'],'fine_dt':b['dt'],'max_difference_mm':float(distance.max()),
                      'rms_difference_mm':float(np.sqrt(np.mean(distance**2))),
                      'final_max_difference_mm':float(distance[-1].max())})
    result={'runs':records,'pairs':pairs,'contact_settings':configuration,'empirical_validation':False,'convergence_proven':False,
            'scope':'Same-model temporal sensitivity only; no clinical material or grasp validation'}
    if len(pairs)>1:
        result['successive_max_difference_ratio']=pairs[-1]['max_difference_mm']/pairs[-2]['max_difference_mm']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    compare(**vars(p.parse_args()))
