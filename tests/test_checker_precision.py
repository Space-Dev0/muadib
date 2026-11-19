import unittest

from exposure.checker_precision import upper_error_bound, zero_error_sample_size


class CheckerPrecisionTests(unittest.TestCase):
    def test_zero_errors_require_correct_sample_size(self):
        for alpha, expected in ((.05, 149), (.025, 183)):
            with self.subTest(alpha=alpha):
                n = zero_error_sample_size(.02, alpha)
                self.assertEqual(n, expected)
                self.assertLessEqual(upper_error_bound(0, n, alpha), .02)
                self.assertGreater(upper_error_bound(0, n - 1, alpha), .02)

    def test_errors_widen_bound_and_all_errors_is_one(self):
        self.assertGreater(upper_error_bound(1, 183, .025), upper_error_bound(0, 183, .025))
        self.assertEqual(upper_error_bound(5, 5), 1.0)

    def test_invalid_inputs(self):
        for errors, accepted in ((0, 0), (-1, 10), (11, 10), (1.5, 10)):
            with self.assertRaises(ValueError):
                upper_error_bound(errors, accepted)
        with self.assertRaises(ValueError):
            zero_error_sample_size(0)


if __name__ == "__main__":
    unittest.main()
