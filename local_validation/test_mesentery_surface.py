import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from mesentery_surface import extract
from multilayer_mesentery import extract_multilayer,split_vertex_fans


class MidsurfaceTests(unittest.TestCase):
    def test_point_touching_triangles_are_separated_without_moving_geometry(self):
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],float)
        faces=np.array([[0,1,2],[0,3,4]])
        vertices,new_faces,parent=split_vertex_fans(points,faces)
        self.assertEqual(len(vertices),6)
        self.assertEqual(len(set(new_faces[0])&set(new_faces[1])),0)
        np.testing.assert_array_equal(vertices[new_faces],points[faces])

    def test_rotated_slab_recovers_midplane_and_envelope_chord(self):
        mesh=trimesh.creation.box(extents=[.03,.02,.002])
        rotation=trimesh.transformations.rotation_matrix(.6,[1,2,3])
        mesh.apply_transform(rotation)
        center=np.array([.12,-.07,.04]);mesh.apply_translation(center)
        surface,arrays,report=extract(mesh,.001)
        normal=rotation[:3,:3]@np.array([0,0,1.])
        np.testing.assert_allclose((surface.vertices-center)@normal,0.,atol=1e-10)
        np.testing.assert_allclose(arrays['source_axis_chord_m'],.002,atol=1e-10)
        self.assertEqual(report['multiple_intervals'],0)

    def test_two_overlapping_projected_layers_are_not_bridged(self):
        first=trimesh.creation.box(extents=[.03,.02,.002]);first.apply_translation([0,0,-.002])
        second=trimesh.creation.box(extents=[.03,.02,.002]);second.apply_translation([0,0,.002])
        with self.assertRaisesRegex(ValueError,'No unambiguous'):
            extract(trimesh.util.concatenate([first,second]),.001)

    def test_multilayer_retains_both_slabs_without_cross_gap_faces(self):
        first=trimesh.creation.box(extents=[.03,.02,.002]);first.apply_translation([0,0,-.002])
        second=trimesh.creation.box(extents=[.03,.02,.002]);second.apply_translation([0,0,.002])
        surface,arrays,report=extract_multilayer(trimesh.util.concatenate([first,second]),.001)
        self.assertEqual(report['connected_surface_components'],2)
        self.assertEqual(report['represented_multiple_interval_rays'],report['multiple_interval_rays'])
        np.testing.assert_allclose(np.abs(surface.vertices[:,2]),.002,atol=1e-10)
        np.testing.assert_allclose(np.ptp(surface.vertices[surface.faces,2],axis=1),0,atol=1e-10)
        np.testing.assert_allclose(arrays['source_axis_chord_m'],.002,atol=1e-10)


if __name__=='__main__':unittest.main()
