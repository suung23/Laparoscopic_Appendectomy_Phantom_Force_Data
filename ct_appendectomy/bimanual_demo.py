"""Scripted two-hand approach/grasp/lift/release acceptance test, not RL."""
import json
import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom
from tissue_metrics import tetra_health


def run(controller,simulation,rep,rgb,out,record=True):
    frames=out/'bimanual_frames';frames.mkdir(exist_ok=True)
    q0=controller.points().copy();targets={h:controller.arms[h].pose()['position_m'].copy() for h in controller.arms}
    camera=UsdGeom.Xformable(controller.stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0]
    camera_start=np.asarray(camera.Get()).copy()
    np.savez_compressed(out/'settled_geometry.npz',simulation=q0,
        collision=np.asarray(controller.stage.GetPrimAtPath('/World/Atlas/appendix/coll').GetAttribute('points').Get()))
    sample=0;frame_number=0;trajectory=[];worst_jump=0.;previous=q0.copy();latch_checks=[]
    def step(phase,count):
        nonlocal sample,frame_number,previous,worst_jump
        for _ in range(count):
            if phase=='close on appendix':
                cents=controller.centers()
                for h,index in [('Left',1),('Right',5)]:
                    if h not in controller.held:controller.target_position(h,cents[index],0.)
            before=set(controller.held)
            controller.tick();simulation.step(render=True)
            q=controller.points()
            if set(controller.held)-before:
                jump=float(np.linalg.norm(q-previous,axis=1).max());worst_jump=max(worst_jump,jump)
                latch_checks.append({'frame':frame_number,'all_node_step_m':jump,'patch_step_m':{
                    h:float(np.linalg.norm(q[controller.data['clusters'][controller.held[h]['cluster']]['indices']]-previous[controller.data['clusters'][controller.held[h]['cluster']]['indices']],axis=1).max())
                    for h in set(controller.held)-before}})
            previous=q.copy()
            if not np.isfinite(q).all():raise RuntimeError('Non-finite tissue state')
            if frame_number%8==0:
                item={'frame':frame_number,'phase':phase,'held':{h:v['cluster'] for h,v in controller.held.items()},
                      'positions':{h:controller.arms[h].pose()['position_m'].tolist() for h in controller.arms}}
                trajectory.append(item)
                if record:
                    simulation.pause();rep.orchestrator.step(rt_subframes=2,delta_time=0.)
                    pixels=np.asarray(rgb.get_data())[:,:,:3]
                    if pixels.shape!=(720,1280,3):raise RuntimeError('Missing demo frame')
                    image=Image.fromarray(pixels.copy());draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,1280,28),fill=(16,19,23))
                    draw.text((10,8),f'SCRIPTED BIMANUAL | {phase} | grasp assist: '+','.join(controller.held or ['none'])+' | fixed camera',(245,245,245))
                    image.save(frames/f'{sample:04d}.png');sample+=1
                    simulation.play()
            frame_number+=1
    # Empty-space closure must not acquire the nearest appendix remotely.
    for h in controller.arms:controller.targets[h][4]=0.
    step('empty-space closure',90)
    if controller.held:raise RuntimeError('Remote grasp: empty-space closure latched tissue')
    for h in controller.arms:controller.targets[h][4]=np.deg2rad(37)
    step('open before approach',70)
    cents=controller.centers()
    indices={'Left':1,'Right':5}
    target0={h:cents[i].copy() for h,i in indices.items()}
    # Use the preplanned view of both settled sites; do not move the camera.
    for h in controller.arms:controller.target_position(h,target0[h],np.deg2rad(37))
    step('two-hand approach',210)
    for h in controller.arms:controller.targets[h][4]=0.
    step('close on appendix',180)
    grasp={h:dict(v) for h,v in controller.held.items()}
    if len(grasp)!=2:
        debug={'events':controller.events,'centers':controller.centers().tolist(),
               'arms':{h:{'position':arm.pose()['position_m'].tolist(),'state':arm.state.tolist(),'axes':arm.pose()['axes'].tolist()} for h,arm in controller.arms.items()}}
        (out/'grasp_failure.json').write_text(json.dumps(debug,indent=2))
        raise RuntimeError('Both instruments must acquire distinct appendix patches: '+json.dumps(controller.events))
    if grasp['Left']['cluster']==grasp['Right']['cluster']:raise RuntimeError('Conflicting ownership')
    lift_start=controller.points().copy()
    tool_start={h:controller.arms[h].pose()['position_m'].copy() for h in controller.arms}
    for h in controller.arms:controller.target_position(h,tool_start[h]+np.array([0,.008,0]),0.)
    step('bilateral 8 mm lift',150)
    step('hold',60)
    lifted=controller.points().copy();measure={}
    for h,v in grasp.items():
        idx=controller.data['clusters'][v['cluster']]['indices']
        delta=(lifted[idx]-lift_start[idx]).mean(0)
        motion=controller.arms[h].pose()['position_m']-tool_start[h]
        measure[h]={'cluster':v['cluster'],'lift_m':float(delta[1]),'tracking_error_m':float(np.linalg.norm(delta-motion))}
    health=tetra_health(q0,lifted,np.asarray(controller.sim_mesh.GetAttribute('tetVertexIndices').Get(),int))
    # Always release before evaluating acceptance gates, including failed tests.
    for h in controller.arms:controller.targets[h][4]=np.deg2rad(37)
    step('release both jaws',90)
    for h in controller.arms:controller.target_position(h,targets[h],np.deg2rad(37))
    step('withdraw',180);step('settle after release',180)
    result={'scripted':True,'grasp_model':'geometry-gated FEM attachment assist','empty_space_rejected':True,
            'camera_unchanged':bool(np.array_equal(camera_start,np.asarray(camera.Get()))),
            'two_distinct_patches':True,'hands':measure,'lift_tissue_health':health,'max_latch_step_displacement_m':worst_jump,
            'released':not controller.held,'events':controller.events,'latch_checks':latch_checks,'recorded_frames':sample,'trajectory':trajectory}
    result['max_grasp_patch_latch_step_m']=max((v for item in latch_checks for v in item['patch_step_m'].values()),default=0.)
    (out/'bimanual_demo.json').write_text(json.dumps(result,indent=2))
    if any(v['lift_m']<.004 or v['tracking_error_m']>.003 for v in measure.values()):raise RuntimeError('Bimanual lift tracking gate failed')
    if health['inverted_tetrahedra']:raise RuntimeError('Bimanual lift inverted tissue')
    # Teleportation is measured on the nodes actually attached. Other nodes can
    # move in response to tool collision and are reported separately above.
    if result['max_grasp_patch_latch_step_m']>.001:raise RuntimeError('Grasp patch moved more than 1mm on attachment activation')
    if controller.held:raise RuntimeError('Jaw opening did not release')
    if not result['camera_unchanged']:raise RuntimeError('Camera changed during manipulation demo')
    return result
