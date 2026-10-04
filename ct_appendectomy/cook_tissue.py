"""Fresh source-to-tetrahedra diagnostic; no dynamics or cached attached stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--hex',action='store_true')
p.add_argument('--mesh',type=Path,help='Explicit validated NPZ points/tets; disables auto cooking')
p.add_argument('--prepare-bimanual',action='store_true')
a=p.parse_args()
if a.mesh and a.hex:p.error('Explicit mesh cannot use hex generation')
if a.prepare_bimanual and not a.mesh:p.error('Bimanual preparation requires explicit mesh')
a.out.mkdir(parents=True,exist_ok=True)
report=dict(status='starting',hex_simulation=a.hex,source_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
            scope='fresh cooking topology only; no mechanics validation')
from isaacsim import SimulationApp
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('WARP_CACHE_PATH',str(root/'.cache/isaac-warp'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(root/'.cache/optix'))
tokens={n:root/'.cache/kit'/n for n in ('shared_documents','app_documents','omni_documents','omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
for path in tokens.values():path.mkdir(parents=True,exist_ok=True)
app=SimulationApp({'headless':True,'extra_args':['--/app/vulkan=false','--/app/d3d12=true',
    '--/telemetry/enableNVDF=false','--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false',
    '--/privacy/extraDiagnosticDataOptIn=false','--/crashreporter/enabled=false']+
    [f'--/app/tokens/{k}={v.as_posix()}' for k,v in tokens.items()]})
try:
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from omni.physx.scripts import deformableUtils as du
    from pxr import Sdf,Vt,UsdGeom,UsdShade,PhysxSchema
    omni.usd.get_context().open_stage(str(a.scene.resolve()))
    for _ in range(20):app.update()
    stage=omni.usd.get_context().get_stage();root='/World/Atlas/appendix'
    collision=root+('/coll' if a.hex else '/sim')
    if not du.create_auto_volume_deformable_hierarchy(stage,root,root+'/sim',collision,root+'/render',
        simulation_hex_mesh_enabled=a.hex,cooking_src_simplification_enabled=False,set_visibility_with_guide_purpose=True):
        raise RuntimeError('Authoring failed')
    stage.GetPrimAtPath(root).CreateAttribute('physxDeformableBody:resolution',Sdf.ValueTypeNames.UInt).Set(64)
    if a.mesh:
        du.remove_auto_deformable_body(stage,root)
        data=np.load(a.mesh);vertices=np.asarray(data['points'],np.float32);indices=np.asarray(data['tets'],np.int32)
        explicit=UsdGeom.TetMesh(stage.GetPrimAtPath(root+'/sim'))
        explicit.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        explicit.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(indices))
        explicit.GetPrim().GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(vertices))
        explicit.GetPrim().GetAttribute('omniphysics:restTetVtxIndices').Set(Vt.Vec4iArray.FromNumpy(indices))
        explicit.GetPrim().GetAttribute('deformablePose:default:omniphysics:points').Set(Vt.Vec3fArray.FromNumpy(vertices))
        explicit.GetSurfaceFaceVertexIndicesAttr().Set(UsdGeom.TetMesh.ComputeSurfaceFaces(explicit,0))
        report['explicit_mesh_sha256']=hashlib.sha256(a.mesh.read_bytes()).hexdigest()
    elif not get_physx_cooking_interface().cook_auto_deformable_body(root):raise RuntimeError('Cooking failed')
    mesh=stage.GetPrimAtPath(root+'/sim')
    points=np.asarray(mesh.GetAttribute('omniphysics:restShapePoints').Get(),float)
    cells=np.asarray(mesh.GetAttribute('tetVertexIndices').Get(),int).reshape(-1,4)
    faces=np.concatenate([cells[:,f] for f in ((1,2,3),(0,3,2),(0,1,3),(0,2,1))])
    _,count=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    duplicates=len(cells)-len(np.unique(np.sort(cells,axis=1),axis=0))
    report.update(nodes=len(points),tetrahedra=len(cells),duplicate_tetrahedra=duplicates,
        nonmanifold_faces=int(np.sum(count>2)),status='failed' if duplicates or np.any(count>2) else 'topology_passed')
    np.savez_compressed(a.out/'mesh.npz',points=points,tets=cells)
    if a.prepare_bimanual:
        if report['status']!='topology_passed':raise RuntimeError('Reject invalid explicit mesh')
        du.add_deformable_material(stage,'/World/TissueMaterial',density=1050.,youngs_modulus=12000.,poissons_ratio=.45,dynamic_friction=.5,static_friction=.5)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(UsdShade.Material.Get(stage,'/World/TissueMaterial'),UsdShade.Tokens.weakerThanDescendants,'physics')
        collision_api=PhysxSchema.PhysxCollisionAPI.Apply(mesh)
        collision_api.CreateContactOffsetAttr(.0005);collision_api.CreateRestOffsetAttr(0.)
        body=stage.GetPrimAtPath(root)
        body.CreateAttribute('physxDeformableBody:solverPositionIterationCount',Sdf.ValueTypeNames.UInt).Set(64)
        body.CreateAttribute('physxDeformableBody:linearDamping',Sdf.ValueTypeNames.Float).Set(.1)
        import bimanual_tools
        from bimanual_env import save_config
        manifest=json.loads((a.scene.parent/'scene_manifest.json').read_text())
        config=bimanual_tools.author(stage,root+'/sim',points,np.asarray(manifest['ct_anchor_m']),a.scene.parent/'rosota_distal.usda')
        attached=a.out/'attached.usda';stage.GetRootLayer().Export(str(attached))
        save_config(a.out/'bimanual_config.json',attached,config)
    stage.GetRootLayer().Export(str(a.out/'cooked.usda'))
except Exception as exc:
    report.update(status='failed',error=str(exc));raise
finally:
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    app.close()
