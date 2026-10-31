import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from exposure.render import parse_rule, render_conditioned
from exposure.rules import Attribute, Relation, satisfies


RULE = (Attribute("a", "color", "red"), Attribute("b", "shape", "square"),
        Relation("a", "left_of", "b"))


class ConditionedRendererTests(unittest.TestCase):
    def test_condition_and_reproducibility(self):
        for seed in range(20):
            first, attempts = render_conditioned(seed, RULE)
            second, attempts2 = render_conditioned(seed, RULE)
            self.assertTrue(satisfies(first.objects, RULE))
            self.assertEqual((first.objects, attempts), (second.objects, attempts2))
            self.assertEqual(first.image.tobytes(), second.image.tobytes())

    def test_failure_does_not_loop_forever(self):
        impossible = (Attribute("a", "color", "red"), Attribute("a", "color", "blue"))
        with self.assertRaisesRegex(ValueError, "rare or infeasible"):
            render_conditioned(1, impossible, max_attempts=20)
        with self.assertRaises(ValueError):
            render_conditioned(1, RULE, max_attempts=0)

    def test_rule_parser_rejects_bad_inputs(self):
        for data in ([], {}, [{"type": "attribute", "variable": "a", "field": "color", "value": "pink"}],
                     [{"type": "relation", "subject": "a", "kind": "above", "object": "a"}]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_rule(data)

    def test_cli_writes_condition_and_matching_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "scene"
            subprocess.run([sys.executable, "-m", "exposure.render", "--seed", "42", "--rule",
                            "examples/red_left_of_square.json", "--output", str(prefix)], check=True,
                           capture_output=True, text=True)
            data = json.loads(prefix.with_suffix(".json").read_text())
            self.assertGreaterEqual(data["attempts"], 1)
            from exposure.rules import SceneObject
            self.assertTrue(satisfies(tuple(SceneObject(**obj) for obj in data["objects"]),
                                      parse_rule(data["rule"])))
            self.assertTrue(prefix.with_suffix(".png").exists())


if __name__ == "__main__":
    unittest.main()
