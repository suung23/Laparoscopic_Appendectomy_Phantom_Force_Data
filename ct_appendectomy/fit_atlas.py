"""Constrained rigid atlas initialization; never patient-specific registration.

Keeps the authored base fixed and optimizes rotation against CT label distance
fields. No CT organ is hidden, displaced or deleted to improve the objective.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def fit(source, case, manifest, output):
    metadata = json.loads((source/'ct.json').read_text())
    previous = json.loads(manifest.read_text())
    image = nib.load(str(case/'ct.nii.gz'))
    if hashlib.sha256((case/'ct.nii.gz').read_bytes()).hexdigest() != metadata['ct_sha256']:
        raise ValueError('CT hash differs from provenance')
    hu = np.asarray(image.dataobj, dtype=np.float32)[::2, ::2, ::2]
    occupied = np.zeros(hu.shape, dtype=bool)
    mask_hashes = {}
    for path in sorted((case/'segmentations').glob('*.nii.gz')):
        label = nib.load(str(path))
        if label.shape != image.shape or not np.allclose(label.affine, image.affine):
            raise ValueError('CT/mask coordinate mismatch')
        occupied |= np.asarray(label.dataobj)[::2, ::2, ::2] > 0
        mask_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not mask_hashes:
        raise ValueError('No organ masks')
    body = np.zeros_like(occupied)
    for k in range(hu.shape[2]):
        labels, count = ndimage.label(hu[:, :, k] > -500)
        if count:
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            body[:, :, k] = ndimage.binary_fill_holes(ndimage.binary_closing(labels == sizes.argmax(), iterations=2))
    spacing = np.asarray(image.header.get_zooms()[:3])*2
    organ_sdf = (ndimage.distance_transform_edt(~occupied, sampling=spacing)
                 - ndimage.distance_transform_edt(occupied, sampling=spacing)).astype(np.float32)
    body_depth = ndimage.distance_transform_edt(body, sampling=spacing).astype(np.float32)
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], float)
    scale = np.diag([.001,.001,.001,1])
    voxel_from_world = np.linalg.inv(scale @ swap @ np.asarray(metadata['world_from_RAS']) @ image.affine @ np.diag([2,2,2,1]))
    atlas = trimesh.load(source/'anatomy.glb', force='scene')
    atlas_transform = np.array([[.001,0,0,0],[0,0,-.001,0],[0,.001,0,0],[0,0,0,1]])
    for mesh in atlas.geometry.values():
        mesh.apply_transform(atlas_transform)
    appendix = atlas.geometry['appendix_FJ2565']
    distance, _ = cKDTree(atlas.geometry['ascending_colon_FJ2566'].vertices).query(appendix.vertices)
    base = appendix.vertices[np.argmin(distance)]
    anchor = np.asarray(previous['ct_anchor_m'])
    samples = []
    for name in ('appendix_FJ2565', 'mesoappendix_FJ3397', 'appendicular_artery_FJ3410'):
        vertices = atlas.geometry[name].vertices
        samples.append(vertices[np.unique(np.linspace(0,len(vertices)-1,min(1024,len(vertices)),dtype=int))])
    relative = np.concatenate(samples)-base
    proximal = np.linalg.norm(relative,axis=1) < .004

    def evaluate(angles, details=False):
        rotation = Rotation.from_euler('xyz', angles).as_matrix()
        points = relative @ rotation.T + anchor
        voxels = trimesh.transform_points(points, voxel_from_world).T
        clearance = ndimage.map_coordinates(organ_sdf, voxels, order=1, mode='constant', cval=-100)
        depth = ndimage.map_coordinates(body_depth, voxels, order=1, mode='constant', cval=0)
        # Small base contact tolerance; distal tissue must clear labelled organs.
        allowance = np.where(proximal, 3., .5)
        penalty = np.maximum(-clearance-allowance, 0)**2 + np.maximum(12-depth, 0)**2
        objective = float(penalty.mean() + .1*penalty.max())
        if details:
            return rotation, {'objective':objective, 'sample_count':len(points),
                'distal_penetration_over_0_5mm_fraction':float(np.mean(clearance[~proximal] < -.5)),
                'body_depth_below_12mm_fraction':float(np.mean(depth < 12)),
                'minimum_organ_clearance_mm':float(clearance.min())}
        return objective
    result = differential_evolution(evaluate, [(-np.pi,np.pi)]*3, seed=42,
                                    popsize=18, maxiter=160, polish=True, tol=1e-6)
    rotation, metrics = evaluate(result.x, True)
    report = {'schema_version':1, 'method':'rigid rotation about fixed heuristic colon anchor using downsampled CT mask SDF',
              'registration_validated':False, 'appendix_patient_specific':False,
              'ct_sha256':metadata['ct_sha256'], 'mask_sha256':mask_hashes,
              'atlas_sha256':hashlib.sha256((source/'anatomy.glb').read_bytes()).hexdigest(),
              'atlas_base_m':base.tolist(), 'ct_anchor_m':anchor.tolist(),
              'rotation':rotation.tolist(), 'translation_m':(anchor-rotation@base).tolist(),
              'optimizer_converged':bool(result.success), 'metrics':metrics,
              'limitations':['3mm distance-field sampling; mesh parity audit still required',
                             'Base is heuristic, not a reviewed cecal landmark',
                             'No insufflation or organ displacement modeled']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'metrics':metrics, 'optimizer_converged':bool(result.success)}))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=root.parent/'appendectomy_lab/assets')
    parser.add_argument('--case',type=Path,default=root.parent/'appendectomy_lab/public_ct/s0042')
    parser.add_argument('--manifest',type=Path,default=root/'out/ct_appendectomy/scene_manifest.json')
    parser.add_argument('--out',type=Path,default=root/'out/ct_appendectomy/placement.json')
    args=parser.parse_args()
    fit(args.source,args.case,args.manifest,args.out)
