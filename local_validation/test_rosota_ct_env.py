import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from rosota_ct_env import RosotaCTEnv


class ActionValidation(unittest.TestCase):
    def test_cutter_cannot_capture_assisted_tissue(self):
        with self.assertRaisesRegex(ValueError,'Cutter cannot'):
            RosotaCTEnv(None,None,None,None,grasp_mode='solver_attachment',cutter_cad='unused.usda')

    def test_changed_cutter_asset_is_rejected(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tool.usda';p.write_text('changed')
            p.with_name('manifest.json').write_text(json.dumps({'usd_sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                RosotaCTEnv(None,None,None,None,cutter_cad=p)

    def test_zero_gravity_requires_isolated_non_grasp_fixture(self):
        with self.assertRaises(ValueError):
            RosotaCTEnv(None,None,None,None,diagnostic_zero_gravity=True)
        with self.assertRaises(ValueError):
            RosotaCTEnv(None,None,None,None,diagnostic_zero_gravity=True,
                diagnostic_no_tool_collision=True,grasp_mode='solver_attachment')

    def test_invalid_second_hand_or_scope_is_rejected_before_any_plant_access(self):
        e=object.__new__(RosotaCTEnv);e.done=False
        # Deliberately no plant/controller: validation must finish before any access.
        for field,index,value in [('tools',(1,0,6),101),('tools',(1,0,2),np.nan),('scope',2,1.1)]:
            a={'tools':np.zeros((2,1,7)),'scope':np.zeros(4)}
            a[field][index]=value
            with self.assertRaises(ValueError):e.step(a)
            self.assertFalse(e.done)
        with self.assertRaises(ValueError):e.step({'tools':np.zeros((2,1,7))})

    def test_finished_env_rejects_action(self):
        e=object.__new__(RosotaCTEnv);e.done=True
        with self.assertRaises(RuntimeError):e.step({'tools':np.zeros((2,1,7)),'scope':np.zeros(4)})


if __name__=='__main__':unittest.main()
