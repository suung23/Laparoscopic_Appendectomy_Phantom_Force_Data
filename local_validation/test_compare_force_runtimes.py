import copy
import unittest
from local_validation.compare_force_runtimes import compare


class RuntimeComparison(unittest.TestCase):
    def setUp(self):
        self.a=dict(source_sha256='source',helper_sha256={'helper':'hash'},robot_scene_sha256='scene',
            load_protocol={'end_tick':2},gpu=True,gravity_m_s2=0.,robot_collisions=False,
            surrounding_scene_retained=True,excluded_component=None,external_forces_every_iteration=False,
            reset_seed=True,warmup_ticks=480,isaacsim_version='5.1.0',status='failed',physics_ticks=6,
            vector_error_n=[.04,.01,.05],release_residual_n=[0,0,0],
            trace=[{'axis':a,'tick':t} for a in range(3) for t in range(2)])
        self.b=copy.deepcopy(self.a)
        self.b.update(isaacsim_version='6.1.0.0',status='passed',vector_error_n=[1e-6]*3,
            projected_torque_check={'status':'passed','error_nm':[1e-6]*3,'tolerance_nm':[.0001]*3})

    def test_matching_full_scene_can_record_resolution(self):
        self.assertTrue(compare(self.a,self.b)['prior_failure_resolved'])

    def test_scene_or_helper_change_prevents_attribution(self):
        for key in ('robot_scene_sha256','helper_sha256'):
            b=copy.deepcopy(self.b);b[key]='changed'
            with self.assertRaises(ValueError):compare(self.a,b)

    def test_false_green_status_does_not_override_measured_failure(self):
        self.b['vector_error_n']=[.1]*3
        self.assertFalse(compare(self.a,self.b)['prior_failure_resolved'])

    def test_missing_trace_is_not_a_complete_run(self):
        self.b['trace'].pop()
        with self.assertRaises(ValueError):compare(self.a,self.b)


if __name__=='__main__':unittest.main()
