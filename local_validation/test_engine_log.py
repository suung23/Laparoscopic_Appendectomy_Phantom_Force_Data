import tempfile
import unittest
from pathlib import Path
from audit_engine_log import audit


class EngineLogAudit(unittest.TestCase):
    def test_physics_errors_fail_even_with_successful_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'engine.log'
            path.write_text('[Error] [omni.physx.plugin] PhysX error: updateNodeInverseVolumes(): tetrahedron is degenerate or inverted\n[Error] [omni.physx.plugin] PhysX error: another error\nSimulation App Shutting Down\n')
            result=audit(path)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(sum(result['counts'].values()),2)
        self.assertNotIn('another error',str(result))

    def test_unrelated_extension_error_is_not_mislabeled_physics(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'engine.log';path.write_text('[Error] [other.extension] failure\n')
            result=audit(path)
        self.assertEqual(result['status'],'no_physx_errors_observed')


if __name__=='__main__':unittest.main()
