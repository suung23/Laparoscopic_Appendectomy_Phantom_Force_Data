"""Boundary-preserving fairing for an authored membrane; not tissue relaxation."""
import argparse,json,os
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix,diags,eye
from scipy.sparse.linalg import spsolve
from pxr import Usd
from build_scene import mesh_prim

def fair(mesh,strength=1.):
    if not np.isfinite(strength) or strength<0:raise ValueError("Finite nonnegative strength required")
    edges=mesh.edges_unique;n=len(mesh.vertices)
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    adjacency=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    lap=eye(n)-diags(1/np.asarray(adjacency.sum(axis=1)).ravel())@adjacency
    counts=np.bincount(mesh.edges_unique_inverse);fixed=np.unique(edges[counts==1])
    # Keep one neighbor ring as well to preserve the authored departure.
    fixed=np.unique(np.r_[fixed,cols[np.isin(rows,fixed)]])
    free=np.setdiff1d(np.arange(n),fixed)
    system=eye(n)+strength*(lap.T@lap)
    system=system.tocsr();result=mesh.vertices.copy()
    rhs=mesh.vertices[free]-system[free][:,fixed]@mesh.vertices[fixed]
    result[free]=spsolve(system[free][:,free],rhs)
    before=float(np.sum((lap@mesh.vertices)**2));after=float(np.sum((lap@result)**2))
    np.testing.assert_array_equal(result[fixed],mesh.vertices[fixed])
    return trimesh.Trimesh(result,mesh.faces,process=False),dict(fixed_vertices=len(fixed),bending_proxy_before=before,bending_proxy_after=after,max_motion_mm=float(np.linalg.norm(result-mesh.vertices,axis=1).max()*1000),strength=strength)

def build(scene,surface,out,strength):
    d=np.load(surface);old=trimesh.Trimesh(d['points'],d['faces'],process=False)
    mesh,report=fair(old,strength)
    out.mkdir(parents=True,exist_ok=True);stage=Usd.Stage.CreateNew(str((out/'review.usda').resolve()))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(scene.resolve(),out.resolve()).replace('\\','/'))
    mesh_prim(stage,'/SurgicalRegion/mesoappendix_FJ3397',mesh,(.72,.56,.32));stage.GetRootLayer().Save()
    np.savez_compressed(out/'surface.npz',points=mesh.vertices,faces=mesh.faces)
    report.update(status='unreviewed fairing candidate',max_dihedral_before_deg=float(np.degrees(old.face_adjacency_angles).max()),max_dihedral_after_deg=float(np.degrees(mesh.face_adjacency_angles).max()),physics_relaxation=False,collision_checked=False,source_surface=str(surface))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--surface',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--strength',type=float,default=10.)
    a=p.parse_args();build(a.scene,a.surface,a.out,a.strength)
