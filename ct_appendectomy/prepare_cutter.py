"""Build private Rosota cutter CAD in metres, with a documented hinge frame."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdShade
from build_scene import mesh_prim

ROOT=Path(__file__).resolve().parents[1]


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    target=out/'rosota_cutter.usda'
    if target.exists():raise ValueError('Use a fresh output directory')
    source=ROOT/'isaac/build_surgical_cell.py'
    names=['CUTTER_VIS_XYZ','CUTTER_VIS_RPY','CUTTER_TIP_XYZ','CUTTER_TIP_RPY','BLADE_PIVOT_XYZ','BLADE_PIN_AXIS','BLADE_OPEN_TOTAL_DEG']
    values={}
    for n in ast.parse(source.read_text(encoding='utf-8-sig')).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names:
            values[n.targets[0].id]=ast.literal_eval(n.value)
    if set(values)!=set(names):raise ValueError('Missing upstream cutter geometry constants')
    visual=Rotation.from_euler('xyz',values['CUTTER_VIS_RPY']).as_matrix()
    basis=Rotation.from_euler('xyz',values['CUTTER_TIP_RPY']).as_matrix()@np.diag([-1.,1.,-1.])
    axis=basis.T@values['BLADE_PIN_AXIS']
    if not np.allclose(axis,[0,1,0],atol=1e-6):raise ValueError('Unexpected hinge axis')
    manifest=json.loads((ROOT/'validation/rosota-reference-assets.json').read_text())
    expected={f['path']:f['sha'] for f in manifest['verified_files']}
    folder=Path('laparo_endeffector/motorized_laparo_cutter_urdf/motorized_laparo_cutter_urdf/meshes')
    stage=Usd.Stage.CreateNew(str(target));stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/Tool').GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.)
    records=[]
    for part in ['cutter_tip_left','cutter_tip_right','laparo_shaft','part_13','part_15','gripper_shaft_inner','gripper_shaft_outer']:
        path=folder/(part+'.stl');raw=(ROOT/path).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if expected.get(path.as_posix())!=blob:raise ValueError('Unverified private CAD '+part)
        mesh=trimesh.load(ROOT/path,force='mesh')
        mesh.vertices=(mesh.vertices*.001@visual.T+values['CUTTER_VIS_XYZ']-np.asarray(values['BLADE_PIVOT_XYZ']))@basis
        UsdGeom.Xform.Define(stage,'/Tool/'+part);mesh_prim(stage,'/Tool/'+part+'/mesh',mesh,(.55,.58,.61),.2)
        UsdShade.Shader(stage.GetPrimAtPath('/Tool/'+part+'/mesh_material/surface')).GetInput('metallic').Set(.85)
        records.append(dict(part=part,git_blob=blob,bounds_m=mesh.bounds.tolist(),triangles=len(mesh.faces)))
    stage.GetRootLayer().Save()
    tip=basis.T@(np.asarray(values['CUTTER_TIP_XYZ'])-values['BLADE_PIVOT_XYZ'])
    result=dict(upstream_commit=manifest['commit'],private_no_redistribution=True,units='metres',hinge_axis=axis.tolist(),
        hinge_to_tip_m=tip.tolist(),open_total_deg=values['BLADE_OPEN_TOTAL_DEG'],parts=records,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),usd_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        scope='CAD/frame asset only; no collision, articulation, cutting or task validation')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ['units','hinge_axis','hinge_to_tip_m','open_total_deg','scope']}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out)
