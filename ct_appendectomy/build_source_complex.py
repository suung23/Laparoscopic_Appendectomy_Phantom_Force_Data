"""Native atlas surgical-region assembly, without cross-subject graft or smoothing.

This is a structural reference asset, not reconstructed Appendix300 anatomy.
No physical inter-organ attachments are implied by proximity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom
from build_scene import mesh_prim


def build(source,out):
    metadata=json.loads((source/'anatomy.json').read_text(encoding='utf-8'))
    display=trimesh.load(source/'anatomy.glb',force='scene')
    keep={'appendix','ascending_colon','mesoappendix','appendicular_artery','ileum','mesentery','peritoneum'}
    raw_to_display=np.asarray(metadata['transform_from_source'])
    display_to_ras=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
    transform=display_to_ras@raw_to_display
    transform[:3,:]*=.001
    out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str((out/'surgical_region.usda').resolve()))
    root=UsdGeom.Xform.Define(stage,'/SurgicalRegion')
    stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,'Z')
    root.GetPrim().SetCustomDataByKey('status','native atlas structural reference; not production surgical asset')
    meshes={};records=[]
    for item in metadata['structures']:
        if item['name'] not in keep:continue
        path=source/item['source_file']
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=item['source_sha256']:raise ValueError('Source hash mismatch: '+path.name)
        mesh=trimesh.load(path,force='mesh',process=False)
        original_faces=len(mesh.faces)
        mesh.merge_vertices();mesh.remove_unreferenced_vertices()
        if len(mesh.faces)!=original_faces:raise ValueError('Welding changed source face count')
        raw_volume=float(mesh.volume)
        smooth=display.geometry[item['id']]
        ratio=float(smooth.volume/raw_volume) if raw_volume else None
        mesh.apply_transform(transform)
        color=(.70,.43,.40)
        if item['name'] in ('mesentery','mesoappendix'):color=(.72,.56,.32)
        if item['name']=='appendicular_artery':color=(.60,.10,.12)
        obj=mesh_prim(stage,'/SurgicalRegion/'+item['id'],mesh,color)
        obj.GetPrim().SetCustomDataByKey('source_file',path.name)
        obj.GetPrim().SetCustomDataByKey('source_sha256',sha)
        meshes[item['id']]=mesh
        records.append(dict(id=item['id'],source_file=path.name,sha256=sha,
            source_triangles=original_faces,previous_display_triangles=len(smooth.faces),
            previous_display_to_source_volume_ratio=ratio,source_surface_watertight_after_welding=bool(mesh.is_watertight)))
    stage.GetRootLayer().Save()
    appendix=meshes['appendix_FJ2565']
    proximity={}
    for name in ('ascending_colon_FJ2566','mesoappendix_FJ3397'):
        _,distance,_=trimesh.proximity.closest_point(meshes[name],appendix.vertices)
        proximity[name]=dict(appendix_vertex_to_surface_mm_min_p05_median=(np.quantile(distance,[0,.05,.5])*1000).tolist(),
            scope='Unsigned source-surface proximity, not attachment or nonintersection validation')
    report=dict(status='unreviewed native anatomical assembly',license=metadata['license'],
        attribution='BodyParts3D, Database Center for Life Science',
        source_metadata_sha256=hashlib.sha256((source/'anatomy.json').read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256((out/'surgical_region.usda').read_bytes()).hexdigest(),
        raw_source_to_asset_matrix=transform.tolist(),structures=records,proximity=proximity,
        smoothing_applied=False,cross_subject_registration=False,physical_attachments=False,
        clinical_geometry_validated=False,
        limitations=['Source atlas is coarse and lacks surgical tissue layers and fatty lobules',
            'No new geometric information is introduced by this assembly',
            'No patient-specific CT or Appendix300 reconstruction',
            'Proximity does not establish tissue junction topology or mechanical coupling',
            'Not substituted into the production simulator before visual and contact validation'])
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(structures=len(records),proximity=proximity),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('../appendectomy_lab/assets'))
    p.add_argument('--out',type=Path,default=Path('out/ct_appendectomy/native_source_complex'))
    a=p.parse_args();build(a.source,a.out)
