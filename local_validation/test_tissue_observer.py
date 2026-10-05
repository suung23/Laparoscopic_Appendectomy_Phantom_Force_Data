import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from tissue_observer import TissueTensorObserver


class Body:
    count=1
    def get_rest_nodal_positions(self):return self.rest[None]
    def get_simulation_element_indices(self):return self.tets[None]
    def get_simulation_nodal_positions(self):return self.position[None]
    def get_simulation_nodal_velocities(self):return self.velocity[None]
    def get_collision_nodal_positions(self):return self.position[None]
    def get_collision_element_indices(self):return self.tets[None]


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
        self.tets=np.array([[0,1,2,3]])
        self.order=np.array([2,0,3,1]);self.body=Body()
        self.body.rest=self.rest[self.order];self.body.tets=np.argsort(self.order)[self.tets]
        self.body.position=self.body.rest+2;self.body.velocity=np.ones((4,3))*3
        self.observer=TissueTensorObserver(self.body,self.rest,self.tets,.01,42)

    def test_live_snapshot_preserves_material_indices_and_si_units(self):
        initial=self.observer.snapshot(42)
        np.testing.assert_allclose(initial['position_m'],(self.rest+2)*.01)
        np.testing.assert_allclose(initial['velocity_m_s'],.03)
        self.body.position+=1
        np.testing.assert_allclose(self.observer.snapshot(42)['position_m'],(self.rest+3)*.01)
        np.testing.assert_allclose(initial['position_m'],(self.rest+2)*.01)

    def test_reset_requires_rebinding(self):
        with self.assertRaises(RuntimeError):self.observer.snapshot(43)
        self.observer.close()
        with self.assertRaises(RuntimeError):self.observer.snapshot(42)

    def test_invalid_buffer_never_falls_back_to_authored_points(self):
        self.body.position[0,0]=np.nan
        with self.assertRaises(RuntimeError):self.observer.snapshot(42)

    def test_changed_geometry_rejected_before_observation(self):
        self.body.rest=self.body.rest.copy();self.body.rest[0,0]+=.01
        with self.assertRaises(ValueError):TissueTensorObserver(self.body,self.rest,self.tets,.01,42)

    def test_collision_snapshot_preserves_its_own_order_and_copies(self):
        result=self.observer.collision_snapshot(42)
        np.testing.assert_allclose(result['collision_points_m'],self.body.position*.01)
        np.testing.assert_array_equal(result['collision_tets'],self.body.tets)
        self.body.position+=10
        np.testing.assert_allclose(result['collision_points_m'],(self.body.rest+2)*.01)

    def test_invalid_collision_topology_and_stale_stage_rejected(self):
        with self.assertRaises(RuntimeError):self.observer.collision_snapshot(43)
        self.body.tets[0,0]=4
        with self.assertRaises(RuntimeError):self.observer.collision_snapshot(42)
        self.observer.close()
        with self.assertRaises(RuntimeError):self.observer.collision_snapshot(42)


if __name__=='__main__':unittest.main()
