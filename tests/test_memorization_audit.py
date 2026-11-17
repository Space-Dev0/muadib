import unittest
import numpy as np
from exposure.memorization_audit import nearest_mse

class MemorizationAuditTests(unittest.TestCase):
    def test_exact_and_known_offset(self):
        zero=np.zeros((64,64,3),dtype=np.float32)
        refs=np.stack([np.full_like(zero,.5),zero,np.ones_like(zero)])
        i,d=nearest_mse(zero,refs)
        self.assertEqual(i,1);self.assertEqual(d,0.)
        i,d=nearest_mse(np.full_like(zero,.25),refs[:1])
        self.assertEqual(i,0);self.assertAlmostEqual(d,.0625)

    def test_invalid_input_and_deterministic_tie(self):
        x=np.zeros((64,64,3),dtype=np.float32)
        self.assertEqual(nearest_mse(x,np.stack([x,x]))[0],0)
        with self.assertRaises(ValueError): nearest_mse(x,np.empty((0,64,64,3)))
        with self.assertRaises(ValueError): nearest_mse(x,np.ones((1,64,64,3))*2)

if __name__=='__main__':unittest.main()
