"""Conservative 2-to-3 internal face flips; exact boundary vertex preservation."""
import argparse
import json
from pathlib import Path
import numpy as np
from improve_interior_tets import quality


def boundary(tets):
    faces=np.concatenate([tets[:,f] for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    f,c=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    if c.max()>2:raise ValueError('Nonmanifold topology')
    return f[c==1]


def flip(points,tets):
    faces={}
    for i,t in enumerate(tets):
        for omit in range(4):
            f=tuple(sorted(np.delete(t,omit)))
            faces.setdefault(f,[]).append(i)
    initial_boundary=boundary(tets);used=set();new=[];records=[]
    existing_edges={tuple(sorted((int(t[i]),int(t[j])))) for t in tets for i in range(4) for j in range(i)}
    for face,ids in faces.items():
        if len(ids)!=2 or any(i in used for i in ids):continue
        old=tets[ids];q,d=quality(points[old])
        if q.min()>=.1:continue
        a,b=[next(int(v) for v in t if v not in face) for t in old]
        if tuple(sorted((a,b))) in existing_edges:continue
        x,y,z=face
        candidate=np.array([[a,b,x,y],[a,b,y,z],[a,b,z,x]])
        _,det=quality(points[candidate]);candidate[det<0]=candidate[det<0][:,[1,0,2,3]]
        nq,nd=quality(points[candidate])
        if nq.min()<=q.min()+1e-6 or np.any(nd<=0):continue
        if not np.isclose(nd.sum(),d.sum(),rtol=1e-10,atol=1e-18):continue
        if not np.array_equal(boundary(old),boundary(candidate)):continue
        used.update(ids);new.extend(candidate);existing_edges.add(tuple(sorted((a,b))))
        records.append(dict(old_min=float(q.min()),new_min=float(nq.min())))
    result=np.concatenate([tets[[i for i in range(len(tets)) if i not in used]],np.asarray(new).reshape(-1,4)]).astype(int)
    if not np.array_equal(initial_boundary,boundary(result)):raise RuntimeError('Boundary changed')
    if len(np.unique(np.sort(result,axis=1),axis=0))!=len(result):raise RuntimeError('Duplicate cells')
    return result,records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();z=np.load(a.mesh);t,records=flip(z['points'],z['tets']);q,d=quality(z['points'][t])
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'mesh.npz',points=z['points'],tets=t)
    r=dict(flips=len(records),minimum_quality=float(q.min()),below_0_1=int((q<.1).sum()),boundary_exact=True,positive=bool(np.all(d>0)),scope='Geometric diagnostic only, no mechanical acceptance',records=records)
    (a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}))
