"""Read-only live deformable observation with exact material-node correspondence.

Requires the Warp GPU tensor pipeline. USD points are never a runtime fallback.
Positions and velocities returned by PhysX are world-space stage units; outputs
are SI and retain the authored node order used by attachment and damage indices.
"""
import numpy as np
from node_correspondence import node_permutation


class TissueTensorObserver:
    def __init__(self,body,authored_rest,authored_tets,meters_per_unit,stage_id,owner=None):
        if not np.isfinite(meters_per_unit) or meters_per_unit<=0:
            raise ValueError('Positive finite meters_per_unit required')
        if body.count!=1:raise ValueError('Exactly one deformable body required')
        self._body=body;self._owner=owner;self._closed=False
        self.stage_id=int(stage_id);self.meters_per_unit=float(meters_per_unit)
        rest=self._array(body.get_rest_nodal_positions())[0]
        tets=self._array(body.get_simulation_element_indices())[0]
        self.permutation=node_permutation(authored_rest,authored_tets,rest,tets)
        self.node_count=len(self.permutation)

    @staticmethod
    def _array(value):
        return np.asarray(value.numpy() if hasattr(value,'numpy') else value)

    @classmethod
    def bind(cls,stage,body_path,mesh_path,stage_id):
        import omni.physics.tensors as tensors
        from pxr import UsdGeom
        prim=stage.GetPrimAtPath(mesh_path)
        if not prim:raise ValueError('Missing simulation mesh')
        owner=tensors.create_simulation_view('warp',stage_id=stage_id)
        body=owner.create_volume_deformable_body_view(body_path)
        return cls(body,np.asarray(prim.GetAttribute('omniphysics:restShapePoints').Get()),
            np.asarray(prim.GetAttribute('tetVertexIndices').Get()),
            UsdGeom.GetStageMetersPerUnit(stage),stage_id,owner)

    def snapshot(self,stage_id):
        if self._closed:raise RuntimeError('Observer is closed; rebind after reset')
        if int(stage_id)!=self.stage_id:raise RuntimeError('Stage changed; rebind after reset')
        values={}
        for name,getter in [('position_m',self._body.get_simulation_nodal_positions),
                            ('velocity_m_s',self._body.get_simulation_nodal_velocities)]:
            raw=self._array(getter())
            if raw.shape!=(1,self.node_count,3) or not np.isfinite(raw).all():
                raise RuntimeError('Invalid live tissue buffer: '+name)
            values[name]=raw[0,self.permutation].astype(float,copy=True)*self.meters_per_unit
        return values

    def close(self):
        self._closed=True;self._body=None;self._owner=None

    def collision_snapshot(self,stage_id):
        """Raw collision topology and SI world points; no simulation-node mapping."""
        if self._closed:raise RuntimeError('Observer is closed; rebind after reset')
        if int(stage_id)!=self.stage_id:raise RuntimeError('Stage changed; rebind after reset')
        points=self._array(self._body.get_collision_nodal_positions())
        elements=self._array(self._body.get_collision_element_indices())
        if points.ndim!=3 or points.shape[0]!=1 or points.shape[2]!=3 or not np.isfinite(points).all():
            raise RuntimeError('Invalid collision position buffer')
        if elements.ndim!=3 or elements.shape[0]!=1 or elements.shape[2]!=4 or elements.dtype.kind not in 'iu':
            raise RuntimeError('Expected tetrahedral collision topology')
        if elements.size==0 or elements.min()<0 or elements.max()>=points.shape[1]:
            raise RuntimeError('Invalid collision indices')
        return dict(collision_points_m=points[0].astype(float,copy=True)*self.meters_per_unit,
                    collision_tets=elements[0].copy())
