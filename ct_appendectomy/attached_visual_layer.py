"""USD visual follower for authored tissue layers; no collision or force coupling."""
from pathlib import Path
import numpy as np
import trimesh
from pxr import UsdGeom,Vt
from build_fat_layer import evaluate

class AttachedVisualLayer:
    def __init__(self,stage,parent_path,layer_path,parent_rest,layer_data):
        self.parent=UsdGeom.Mesh(stage.GetPrimAtPath(parent_path));self.layer=UsdGeom.Mesh(stage.GetPrimAtPath(layer_path))
        rest=np.load(parent_rest);data=np.load(layer_data)
        self.parent_faces=rest['faces'].copy();self.parent_count=len(rest['points'])
        self.faces=data['faces'].copy();self.nodes=data['nodes'].copy();self.weights=data['weights'].copy();self.offset=data['frame_offsets'].copy()
        self.layer_rest=data['points'].copy();self.follow=data['follow_weight'].copy() if 'follow_weight' in data else np.ones(len(self.layer_rest))
        if self.follow.shape!=(len(self.layer_rest),) or not np.isfinite(self.follow).all() or (self.follow<0).any() or (self.follow>1).any():raise ValueError('Invalid follow weights')
        for mesh,points,faces in [(self.parent,rest['points'],self.parent_faces),(self.layer,data['points'],self.faces)]:
            if not mesh:raise ValueError('Mesh prim missing')
            if not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),faces.ravel()):raise ValueError('Topology mismatch')
            if not np.allclose(mesh.GetPointsAttr().Get(),points,rtol=0,atol=1e-8):raise ValueError('Rest shape mismatch')
        self.shape_regularization=False
        self.check_frames()
    def check_frames(self):
        a=UsdGeom.Xformable(self.parent).ComputeLocalToWorldTransform(0)
        b=UsdGeom.Xformable(self.layer).ComputeLocalToWorldTransform(0)
        if not np.allclose(np.asarray(a),np.asarray(b),rtol=0,atol=1e-10):raise ValueError('Parent and layer must share coordinate frame')
    def update(self):
        self.check_frames()
        points=np.asarray(self.parent.GetPointsAttr().Get(),float)
        if points.shape!=(self.parent_count,3) or not np.isfinite(points).all():raise ValueError('Invalid parent coordinates')
        if not np.array_equal(self.parent.GetFaceVertexIndicesAttr().Get(),self.parent_faces.ravel()):raise ValueError('Parent topology changed')
        positions=evaluate(points,self.nodes,self.weights,self.offset)
        positions=self.layer_rest+self.follow[:,None]*(positions-self.layer_rest)
        if self.shape_regularization:
            from visual_shape_regularization import regularize
            edges=trimesh.Trimesh(self.layer_rest,self.faces,process=False).edges_unique
            positions=regularize(self.layer_rest,edges,positions,np.flatnonzero(self.follow==0))
        if not np.isfinite(positions).all():raise ValueError('Degenerate parent frame')
        mesh=trimesh.Trimesh(positions,self.faces,process=False)
        self.layer.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
        self.layer.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
        self.layer.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.bounds.astype(np.float32)))
        return positions

    def parent_loads(self,layer_forces):
        """Return mapped loads for a solver caller; does not apply them."""
        if self.shape_regularization or not np.all(self.follow==1):raise ValueError("Partial visual following has no integrated support-force accounting")
        from layer_force_mapping import pullback_forces
        self.check_frames()
        if not np.array_equal(self.parent.GetFaceVertexIndicesAttr().Get(),self.parent_faces.ravel()):raise ValueError('Parent topology changed')
        points=np.asarray(self.parent.GetPointsAttr().Get(),float)
        if points.shape!=(self.parent_count,3):raise ValueError('Parent vertex count changed')
        return pullback_forces(points,self.nodes,self.weights,self.offset,layer_forces)
