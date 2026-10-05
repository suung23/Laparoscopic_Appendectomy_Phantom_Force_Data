"""Multi-location membrane penetration diagnostic; not exhaustive intersections."""
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from author_attachment_curve import mesh_at

def inspect(scene,surface,out,independent=False):
    stage=Usd.Stage.Open(str(scene.resolve()))
    data=np.load(surface);mesh=trimesh.Trimesh(data['points'],data['faces'],process=False)
    samples={'vertices':mesh.vertices,'edge_midpoints':mesh.vertices[mesh.edges_unique].mean(axis=1),'face_centroids':mesh.triangles_center}
    results={}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or prim.GetName()=='mesoappendix_FJ3397':continue
        other=mesh_at(stage,str(prim.GetPath()))
        if not other.is_watertight:
            results[str(prim.GetPath())]={'status':'open mesh; containment skipped'};continue
        origin=other.bounds.mean(axis=0);scale=1/np.ptp(other.vertices,axis=0).max()
        query=trimesh.Trimesh((other.vertices-origin)*scale,other.faces,process=False)
        counts={}
        for name,points in samples.items():
            local=(points-origin)*scale
            if independent:
                from independent_containment import contains
                inside=contains(query,local)
            else:
                inside=np.concatenate([query.contains(chunk) for chunk in np.array_split(local,max(1,len(local)//256+1))])
            if not inside.any():continue
            nearest=trimesh.proximity.closest_point_naive if independent else trimesh.proximity.closest_point
            batch=16 if independent else 256
            distances=np.concatenate([nearest(query,chunk)[1]/scale for chunk in np.array_split(local[inside],max(1,int(inside.sum())//batch+1))])
            deep=distances>1e-4
            if deep.any():counts[name]={'count_over_0_1mm':int(deep.sum()),'max_depth_mm':float(distances.max()*1000)}
        if counts:results[str(prim.GetPath())]=counts
    report={'status':'sampled diagnostic only','backend':'independent scipy ray parity / exhaustive closest point' if independent else 'trimesh rtree','surface':str(surface),'samples':{k:len(v) for k,v in samples.items()},'penetrations':results,'exhaustive_intersection_check':False,'clinical_validity':False}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--surface',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--independent',action='store_true')
    a=p.parse_args();inspect(a.scene,a.surface,a.out,a.independent)
