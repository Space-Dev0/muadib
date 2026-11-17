import math
import random
import unittest
from exposure.object_exposure_null import independent_object_null

class ObjectExposureNullTests(unittest.TestCase):
    def test_review_example_and_nonmonotonicity(self):
        def lift(k): return independent_object_null([.5,.5], {k:1.})['scene_log_lift_under_null']
        self.assertAlmostEqual(lift(1), 0.)
        self.assertAlmostEqual(lift(3), math.log(.578125/.765625))
        self.assertGreater(lift(10),lift(3))

    def test_variable_count_matches_monte_carlo(self):
        expected=independent_object_null([.4,.6],{1:.3,3:.7})
        rng=random.Random(1811); count=60000; singles=[0,0]; joint=0
        for _ in range(count):
            k=1 if rng.random()<.3 else 3
            objs=[(rng.random()<.4,rng.random()<.6) for _ in range(k)]
            singles[0]+=any(a for a,b in objs); singles[1]+=any(b for a,b in objs)
            joint+=any(a and b for a,b in objs)
        self.assertAlmostEqual(singles[0]/count,expected['scene_singletons'][0],delta=.008)
        self.assertAlmostEqual(singles[1]/count,expected['scene_singletons'][1],delta=.008)
        self.assertAlmostEqual(joint/count,expected['scene_joint_independent'],delta=.008)

    def test_boundaries_and_invalid_counts(self):
        x=independent_object_null([0.,1.],{0:.5,3:.5},observed_joint=0.)
        self.assertIsNone(x['scene_log_lift_under_null']); self.assertIsNone(x['observed_log_ratio_to_independent_object_null'])
        with self.assertRaises(ValueError): independent_object_null([.5],{1:.4})
        with self.assertRaises(ValueError): independent_object_null([.5],{True:1.})
        with self.assertRaises(ValueError): independent_object_null([float('nan')],{1:1.})

if __name__=='__main__': unittest.main()
