import tempfile
import unittest
from pathlib import Path
import numpy as np
from local_validation.audit_tissue_migration import audit


class MigrationAudit(unittest.TestCase):
    def test_reorder_is_not_mistaken_for_deformation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);points=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]])
            for phase in ['before_initialize','after_initialize','after_settlement']:
                q=points if phase=='before_initialize' else points[[1,0,2,3]]
                np.savez(root/('tissue_'+phase+'.npz'),points=q,rest=points,tets=[[0,1,2,3]],pin=[0],target_world=[[0.,0,0]],anchor_world=np.eye(4))
            row=audit(root)['phases'][1]
            self.assertEqual(row['nearest_original_distance_m'],0)
            self.assertEqual(row['indexed_displacement_m'],1)
            self.assertEqual(row['pin_target_error_m'],1)


if __name__=='__main__':unittest.main()
