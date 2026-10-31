import json
import subprocess
import sys
import unittest

from exposure.grammar_audit import assess, canonical_templates
from exposure.render import parse_rule
from exposure.rules import Attribute, Relation, SceneObject, satisfies


class GrammarAuditTests(unittest.TestCase):
    def test_all_canonical_cells_have_valid_witnesses(self):
        templates = canonical_templates()
        self.assertEqual(len(templates), 12)
        self.assertEqual({(t["k_attr"], t["k_rel"]) for t in templates},
                         {(a, r) for a in (1, 2, 3) for r in (0, 1, 2)})
        for template in templates:
            self.assertTrue(template["feasible"])
            scene = tuple(SceneObject(**obj) for obj in template["witness"])
            self.assertEqual(len(scene), 3)
            # Every witness, including padding objects, uses separated grid cells.
            self.assertEqual(len({(obj.x, obj.y) for obj in scene}), 3)
            self.assertEqual(len({obj.x for obj in scene}), 3)
            self.assertTrue(satisfies(scene, parse_rule(template["predicates"])))

    def test_impossible_cycles_attributes_and_object_count(self):
        self.assertEqual(assess((Relation("a", "left_of", "b"), Relation("b", "left_of", "a")))["reason"],
                         "cyclic_relation")
        self.assertEqual(assess((Attribute("a", "color", "red"), Attribute("a", "color", "blue")))["reason"],
                         "conflicting_attributes")
        four = tuple(Attribute(name, "shape", "circle") for name in ("a", "b", "c", "d"))
        self.assertEqual(assess(four)["reason"], "more_than_three_variables")

    def test_above_and_left_independent_orders(self):
        predicates = (Relation("a", "left_of", "b"), Relation("b", "above", "a"))
        result = assess(predicates)
        self.assertTrue(result["feasible"])
        self.assertTrue(satisfies(tuple(SceneObject(**obj) for obj in result["witness"]), predicates))

    def test_cli_scopes_counts(self):
        run = subprocess.run([sys.executable, "-m", "exposure.grammar_audit", "--attrs", "2",
                              "--relations", "2"], check=True, capture_output=True, text=True)
        result = json.loads(run.stdout)
        self.assertEqual({t["layout"] for t in result["templates"]}, {"chain", "same_pair"})
        self.assertIn("not an exhaustive", result["scope"])


if __name__ == "__main__":
    unittest.main()
