"""Offline cut-topology benchmark on the cooked appendix volume mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from itertools import permutations
from tetra_cut import cut_plane,signed_volumes,boundary_faces


def components(cells,point_count):
    rows=np.repeat(cells[:,0],3)
    cols=cells[:,1:].ravel()
    graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(point_count,point_count)).tocsr()
    count,labels=connected_components(graph,directed=False)
    return int(count)


def run(source,output):
    output.mkdir(parents=True,exist_ok=True)
    if source.suffix.lower()=='.ply':
        mesh=trimesh.load(source,force='mesh')
        if not mesh.is_watertight:raise ValueError('Closed input surface required')
        grid=mesh.voxelized(.002).fill()
        lookup={};corners=[];cells=[]
        for index in grid.sparse_indices:
            ids={}
            for corner in np.ndindex(2,2,2):
                key=tuple(index+corner)
                if key not in lookup:lookup[key]=len(corners);corners.append(key)
                ids[corner]=lookup[key]
            for order in permutations(range(3)):
                one=np.zeros(3,int);one[order[0]]=1
                two=one.copy();two[order[1]]=1
                cells.append([ids[(0,0,0)],ids[tuple(one)],ids[tuple(two)],ids[(1,1,1)]])
        points=trimesh.transform_points(np.asarray(corners,float)-.5,grid.transform)
        cells=np.asarray(cells)
        reverse=signed_volumes(points,cells)<0
        swap=cells[reverse,0].copy();cells[reverse,0]=cells[reverse,1];cells[reverse,1]=swap
        preparation='2mm filled atlas voxels, conforming six-tet cube subdivision; block boundary approximation'
    else:
        data=np.load(source)
        points=data['rest'];cells=data['tetrahedra']
        preparation='unmodified input tetrahedra'
    try:
        boundary_faces(points,cells)
    except ValueError as exc:
        report={'status':'failed','scope':'input mesh topology gate','error':str(exc),
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'appendectomy_completed':False}
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    # Between grid planes, deliberately avoiding exact vertex coincidences.
    levels=np.unique(np.round(points[:,0],7))
    middle=len(levels)//2
    offset=float((levels[middle-1]+levels[middle])/2)
    result=cut_plane(points,cells,[1,0,0],offset)
    v0=float(np.abs(signed_volumes(points,cells)).sum())
    v1=float(signed_volumes(result['points'],result['tetrahedra']).sum())
    np.savez_compressed(output/'cut_mesh.npz',**result)
    halves=[]
    for side in (-1,1):
        selected=result['tetrahedra'][result['side']==side]
        faces=boundary_faces(result['points'],selected)
        mesh=trimesh.Trimesh(result['points'],faces,process=False)
        mesh.remove_unreferenced_vertices()
        mesh.export(output/('negative.ply' if side<0 else 'positive.ply'))
        halves.append({'side':side,'watertight':bool(mesh.is_watertight),'tetrahedra':len(selected)})
    before=components(cells,len(points))
    after=components(result['tetrahedra'],len(result['points']))
    report={'schema_version':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'offline planar split of solid appendix tetrahedra; not a live surgical cut',
            'mesh_preparation':preparation,
            'plane_normal':[1,0,0],'plane_offset_m':offset,
            'input_nodes':len(points),'output_nodes':len(result['points']),
            'input_tetrahedra':len(cells),'output_tetrahedra':len(result['tetrahedra']),
            'relative_volume_error':abs(v1-v0)/v0,'components_before':before,'components_after':after,
            'halves':halves,'live_solver_integrated':False,'finite_blade':False,
            'damage_law_validated':False,'appendectomy_completed':False}
    report['status']='passed' if report['relative_volume_error']<1e-8 and after>before and all(item['watertight'] for item in halves) else 'failed'
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise RuntimeError('Cut topology benchmark failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.out)
