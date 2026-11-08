import unittest

from exposure.measure import load_rules
from exposure.render import parse_rule
from exposure.rules import Attribute, Relation, SceneObject, satisfies, validate_rule


class RuleSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.scene = (
            SceneObject("red", "circle", 1, 2),
            SceneObject("blue", "square", 3, 1),
            SceneObject("green", "triangle", 5, 4),
        )

    def test_shared_binding_and_relation(self):
        rule = (Attribute("a", "color", "red"), Attribute("b", "shape", "square"),
                Relation("a", "left_of", "b"))
        self.assertTrue(satisfies(self.scene, rule))
        self.assertFalse(satisfies(self.scene, rule[:-1] + (Relation("b", "left_of", "a"),)))

    def test_distinct_variables_cannot_use_same_object(self):
        rule = (Attribute("a", "color", "red"), Attribute("b", "shape", "circle"))
        self.assertFalse(satisfies(self.scene, rule))
        self.assertTrue(satisfies(self.scene, (rule[0], Attribute("a", "shape", "circle"))))

    def test_incidental_and_subset_exposure(self):
        rule = (Attribute("a", "color", "red"), Attribute("b", "shape", "triangle"),
                Relation("b", "above", "a"))
        self.assertTrue(satisfies(self.scene, rule[:2]))
        self.assertFalse(satisfies(self.scene, rule))
        self.assertTrue(satisfies(self.scene, (Attribute("a", "color", "green"),)))

    def test_ties_and_too_few_objects(self):
        tied = (SceneObject("red", "circle", 1, 1), SceneObject("blue", "square", 1, 1))
        self.assertFalse(satisfies(tied, (Relation("a", "left_of", "b"),)))
        self.assertFalse(satisfies(tied[:1], (Relation("a", "above", "b"),)))

    def test_invalid_rules(self):
        with self.assertRaises(ValueError):
            satisfies(self.scene, ())
        with self.assertRaises(ValueError):
            Relation("a", "left_of", "a")
        with self.assertRaises(ValueError):
            Attribute("a", "texture", "striped")

    def test_validator_rejects_redundant_and_impossible_rules(self):
        red = Attribute("a", "color", "red")
        with self.assertRaisesRegex(ValueError, "Duplicate predicate"):
            validate_rule((red, red))
        with self.assertRaisesRegex(ValueError, "Contradictory"):
            validate_rule((red, Attribute("a", "color", "blue")))
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            validate_rule((Relation("a", "left_of", "b"),
                           Relation("b", "left_of", "a")))
        self.assertEqual(validate_rule((red,)), (red,))

    def test_parser_and_loader_inherit_validation(self):
        duplicate = [
            {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
            {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
        ]
        contradictory = [
            {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
            {"type": "attribute", "variable": "a", "field": "color", "value": "blue"},
        ]
        cycle = [
            {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"},
            {"type": "relation", "subject": "b", "kind": "left_of", "object": "a"},
        ]
        for data, message in ((duplicate, "Duplicate predicate"),
                              (contradictory, "Contradictory"),
                              (cycle, "Cyclic")):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                parse_rule(data)
            with self.assertRaisesRegex(ValueError, "Invalid rule"):
                load_rules({"bad": data})


if __name__ == "__main__":
    unittest.main()
