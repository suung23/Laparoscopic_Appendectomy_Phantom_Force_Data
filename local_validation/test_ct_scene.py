"""Checks the exported hybrid anatomy contract, not clinical correctness."""
import importlib.util
import sys
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pxr import Usd, UsdGeom

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ct_appendectomy'))
spec=importlib.util.spec_from_file_location('ct_builder',ROOT/'ct_appendectomy/build_scene.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class HybridSceneTests(unittest.TestCase):
    def test_geometry_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as temp:
            report=builder.build(ROOT.parent/'appendectomy_lab/assets',temp)
            stage=Usd.Stage.Open(str(Path(temp)/'ct_appendectomy.usda'))
            self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage),1.)
            self.assertEqual(UsdGeom.GetStageUpAxis(stage),'Z')
            self.assertFalse(report['appendix_is_patient_specific'])
            self.assertFalse(report['registration_validated'])
            self.assertEqual(len(report['inputs']),4)
            self.assertEqual(report['camera_inside_ct_organs'],[])
            self.assertTrue(report['body_envelope']['camera_inside'])
            self.assertFalse(report['body_envelope']['peritoneum_segmentation'])
            self.assertFalse(report['body_envelope']['physical_cavity'])
            meshes=[UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            self.assertEqual(len(meshes),15)
            for mesh in meshes:
                v=np.asarray(mesh.GetPointsAttr().Get())
                f=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                self.assertTrue(np.isfinite(v).all())
                self.assertLess(np.abs(v).max(),2.)
                self.assertTrue((f>=0).all() and (f<len(v)).all())
            self.assertGreater(report['ct_anchor_m'][0],.06)
            # Release USD file handles before Windows removes the temp directory.
            meshes.clear()
            stage=None

    def test_reject_nonfinite_anchor(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as temp:
            with self.assertRaises(ValueError):
                builder.build(ROOT.parent/'appendectomy_lab/assets',temp,[float('nan'),0,0])

    def test_checked_profile_and_source_hash(self):
        profile=ROOT/'ct_appendectomy/profiles/s0042_hybrid.json'
        with tempfile.TemporaryDirectory(dir=ROOT/'out') as temp:
            report=builder.build(ROOT.parent/'appendectomy_lab/assets',temp,placement=profile)
            self.assertEqual(report['camera_inside_ct_organs'],[])
            self.assertIsNotNone(report['placement_fit']['atlas_sha256'])
            self.assertFalse(report['registration_validated'])
            incorrect=json.loads(profile.read_text(encoding='utf-8'))
            incorrect['atlas_sha256']='incorrect'
            tampered=Path(temp)/'tampered.json'
            tampered.write_text(json.dumps(incorrect),encoding='utf-8')
            with self.assertRaises(ValueError):
                builder.build(ROOT.parent/'appendectomy_lab/assets',temp,placement=tampered)


if __name__=='__main__': unittest.main()
