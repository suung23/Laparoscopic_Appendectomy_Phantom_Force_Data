import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from mesentery_surface import extract


class MidsurfaceTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
