"""Inspect the captured rabbit meshes without registering them to human CT."""
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Usd, UsdGeom
from build_scene import ROOT, mesh_prim


def inspect():
    output=ROOT/'out/rosota_anatomy';output.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    expected={v['path']:v['sha'] for v in source['files']}
    meshes={};metrics={}
    rotation=np.diag([1.,-1.,-1.])
    for file in sorted((ROOT/'mesh').glob('organ_*.ply')):
        raw=file.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==expected[file.relative_to(ROOT).as_posix()]
        mesh=trimesh.load(file);name=file.stem.removeprefix('organ_')
        metrics[name]={'source_blob':blob,'vertices':len(mesh.vertices),'faces':len(mesh.faces),
                       'axis_aligned_extent_mm':mesh.extents.tolist(),'watertight':bool(mesh.is_watertight)}
        mesh.vertices=mesh.vertices@rotation.T
        meshes[name]=mesh
    stack=np.vstack([m.vertices for m in meshes.values()]);centre=(stack.min(0)+stack.max(0))/2
    for m in meshes.values():m.vertices-=centre
    a,c=meshes['appendix'],meshes['cecum']
    distances,nearest=cKDTree(c.vertices).query(a.vertices)
    pair=[a.vertices[distances.argmin()].tolist(),c.vertices[nearest[distances.argmin()]].tolist()]
    # The CT compiler swaps the viewer axes back to RAS; verify, don't infer from USD upAxis.
    ct=json.loads((ROOT.parent/'appendectomy_lab/assets/ct.json').read_text())
    viewer_to_scene=np.array([[1,0,0],[0,0,1],[0,1,0.]])
    ras_to_scene=viewer_to_scene@np.array(ct['world_from_RAS'])[:3,:3]
    assert np.allclose(ras_to_scene,np.eye(3))
    report={'repository':source['repository'],'commit':source['commit'],'species':'rabbit according to upstream reconstruction documentation',
        'meshes':metrics,'appendix_cecum_closest_sampled_vertex_distance_mm':float(distances.min()),
        'closest_pair_centered_world_mm':pair,
        'rabbit_camera_to_world':rotation.tolist(),'rabbit_gravity_world_m_s2':[0,0,-9.81],
        'human_ras_to_scene':ras_to_scene.tolist(),'human_gravity_scene_m_s2':[0,-9.81,0],
        'human_posture_assumption':'supine; not a patient-position measurement',
        'limitations':['Captured reconstructed surfaces, not complete rabbit anatomy','Bounding-box extent is not anatomical organ length',
                       'Closest mesh vertices are not a reviewed anatomical junction','Rabbit meshes must not substitute for human CT anatomy']}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    stage=Usd.Stage.CreateNew(str(output/'rabbit_reference.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/RabbitReference').GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    colors={'cecum':(.92,.53,.22),'appendix':(.84,.19,.28)}
    for name,m in meshes.items():
        scaled=m.copy();scaled.apply_scale(.001)
        mesh_prim(stage,'/RabbitReference/'+name,scaled,colors.get(name,(.52,.59,.63)))
    stage.GetRootLayer().Save()
    fig=plt.figure(figsize=(14,6),facecolor='#101820')
    for index,(azimuth,elevation,title) in enumerate([(-90,90,'Captured anatomy: top view'),(-55,24,'Captured anatomy: oblique view')],1):
        ax=fig.add_subplot(1,2,index,projection='3d',facecolor='#101820')
        for name,m in meshes.items():
            artist=Poly3DCollection(m.triangles,facecolors=colors.get(name,(.4,.46,.5)),linewidths=0,
                                    alpha=1 if name in colors else .22)
            ax.add_collection3d(artist)
            if name in colors:
                p=m.vertices.mean(0);ax.text(*p,name.upper(),color='white',fontsize=10)
        ax.view_init(elev=elevation,azim=azimuth)
        radius=np.max(np.ptp(stack,axis=0))*.56
        ax.set(xlim=(-radius,radius),ylim=(-radius,radius),zlim=(-radius,radius))
        ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(title,color='white')
    fig.suptitle('Rosota rabbit reconstruction | orange: cecum | red: appendix',color='white',fontsize=15)
    fig.text(.5,.04,'Source poses preserved with one shared rigid transform. These are captured surfaces, not human CT anatomy.',ha='center',color='#c7d1da',fontsize=10)
    fig.savefig(output/'rabbit_cecum_appendix.png',dpi=150,facecolor=fig.get_facecolor(),bbox_inches='tight');plt.close(fig)
    print(json.dumps({'meshes':len(meshes),'nearest_mesh_vertices_mm':float(distances.min()),'ras_to_scene':ras_to_scene.tolist()}))


if __name__=='__main__':inspect()
