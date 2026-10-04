"""CT body contour with a 10 mm inset: visual envelope, not peritoneum segmentation."""
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh

def build_envelope(ct_path,viewer_from_ras):
    image=nib.load(str(ct_path))
    hu=np.asarray(image.dataobj,dtype=np.float32)[::2,::2,::2]
    mask=np.zeros_like(hu,dtype=bool)
    for z in range(hu.shape[2]):
        labels,count=ndimage.label(hu[:,:,z]>-500)
        if count:
            sizes=np.bincount(labels.ravel());sizes[0]=0
            mask[:,:,z]=ndimage.binary_fill_holes(ndimage.binary_closing(labels==sizes.argmax(),iterations=2))
    spacing=np.asarray(image.header.get_zooms()[:3])*2
    inside=ndimage.distance_transform_edt(mask,sampling=spacing)>10.
    inside[[0,-1],:,:]=False;inside[:,[0,-1],:]=False;inside[:,:,[0,-1]]=False
    vertices,faces,_,_=marching_cubes(inside.astype(np.float32),.5,step_size=2)
    ras=nib.affines.apply_affine(image.affine,vertices*2)
    viewer=trimesh.transform_points(ras,np.asarray(viewer_from_ras))
    points=viewer[:,[0,2,1]]*.001
    mesh=trimesh.Trimesh(points,faces,process=True)
    trimesh.smoothing.filter_taubin(mesh,iterations=8)
    return mesh
