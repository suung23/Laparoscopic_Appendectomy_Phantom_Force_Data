import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from anatomy_audit import contains


class AnatomyAuditTests(unittest.TestCase):
    def test_box_inside_outside_and_translation(self):
        box=trimesh.creation.box(extents=[.04,.02,.06])
        box.apply_translation([.1,.2,.3])
        inside,uncertain=contains(box,np.array([[.1,.2,.3],[.119,.2,.3],[.121,.2,.3],[0,0,0]]))
        np.testing.assert_array_equal(inside,[True,True,False,False])
        self.assertFalse(uncertain.any())


if __name__=='__main__': unittest.main()
