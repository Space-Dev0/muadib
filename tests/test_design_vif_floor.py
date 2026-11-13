import unittest

from exposure.design_vif_floor import bound


def report():
    return {"rows": [{"rule": f"r{i}", "design": f"d{j}", "log_mean_marginal": float(i),
                      "log_lift": float(j), "k_attr": 1, "k_rel": i % 2}
                     for i in range(4) for j in range(2)]}


class VIFFloorTests(unittest.TestCase):
    def test_pairwise_floor_and_repeated_design_invariance(self):
        full = report()
        result = bound(full)
        self.assertAlmostEqual(result["correlation"], 1 / 5 ** 0.5)
        self.assertAlmostEqual(result["vif_lower_bound"], 1.25)
        single = {"rows": [row for row in full["rows"] if row["design"] == "d0"]}
        self.assertAlmostEqual(bound(single)["vif_lower_bound"], result["vif_lower_bound"])

    def test_rejects_missing_rows_and_varying_marginal(self):
        data = report()
        data["rows"].pop()
        with self.assertRaisesRegex(ValueError, "same design"):
            bound(data)
        data = report()
        data["rows"][0]["log_mean_marginal"] += 0.01
        with self.assertRaisesRegex(ValueError, "varies within"):
            bound(data)

    def test_perfect_collinearity_is_explicit(self):
        data = report()
        for row in data["rows"]:
            row["log_mean_marginal"] = float(row["k_rel"])
        result = bound(data)
        self.assertEqual(result["status"], "perfect_collinearity")
        self.assertIsNone(result["vif_lower_bound"])


if __name__ == "__main__":
    unittest.main()
