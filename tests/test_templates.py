import copy
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from exposure.grammar_audit import assess
from exposure.measure import load_rules
from exposure.render import COLORS, SHAPES
from exposure.rules import Attribute, Relation, SceneObject, satisfies
from exposure.templates import build_templates, canonical_key, main, validate_templates


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules, cls.manifest = build_templates()
        cls.parsed = load_rules(cls.rules)
        cls.entries = {entry["name"]: entry for entry in cls.manifest["templates"]}

    def test_exactly_57_templates_and_expected_cell_counts(self):
        expected_cells = {
            "attr1_rel0": 7,
            "attr1_rel1": 7,
            "attr1_rel2": 7,
            "attr2_rel0": 6,
            "attr2_rel1": 6,
            "attr2_rel2": 6,
            "attr3_rel0": 6,
            "attr3_rel1": 6,
            "attr3_rel2": 6,
        }
        expected_layouts = {
            "attr1_rel0": {"attributes": 7},
            "attr1_rel1": {"single_relation": 7},
            "attr1_rel2": {"chain": 4, "same_pair": 3},
            "attr2_rel0": {"attributes": 6},
            "attr2_rel1": {"single_relation": 6},
            "attr2_rel2": {"chain": 3, "same_pair": 3},
            "attr3_rel0": {"attributes": 6},
            "attr3_rel1": {"single_relation": 6},
            "attr3_rel2": {"chain": 3, "same_pair": 3},
        }
        self.assertEqual(len(self.rules), 57)
        self.assertEqual(self.manifest["total_templates"], 57)
        self.assertEqual(len(self.manifest["templates"]), 57)
        self.assertEqual(self.manifest["counts"]["by_cell"], expected_cells)
        self.assertEqual(self.manifest["counts"]["by_cell_layout"], expected_layouts)
        self.assertEqual(self.manifest["counts"]["by_k_attr"],
                         {"1": 21, "2": 18, "3": 18})
        self.assertEqual(self.manifest["counts"]["by_k_rel"],
                         {"0": 19, "1": 19, "2": 19})
        self.assertEqual(self.manifest["counts"]["by_layout"],
                         {"attributes": 19, "single_relation": 19,
                          "chain": 10, "same_pair": 9})

    def test_krel2_cells_have_expected_chain_and_same_pair_splits(self):
        expected = {
            1: {"chain": 4, "same_pair": 3},
            2: {"chain": 3, "same_pair": 3},
            3: {"chain": 3, "same_pair": 3},
        }
        for k_attr, layouts in expected.items():
            cell = f"attr{k_attr}_rel2"
            self.assertEqual(self.manifest["counts"]["by_cell_layout"][cell], layouts)
            entries = [entry for entry in self.manifest["templates"]
                       if entry["k_attr"] == k_attr and entry["k_rel"] == 2]
            self.assertEqual(Counter(entry["layout"] for entry in entries),
                             Counter(layouts))
            for entry in entries:
                self.assertEqual(entry["relation_count"], 2)
                relations = [predicate for predicate in self.parsed[entry["name"]]
                             if isinstance(predicate, Relation)]
                self.assertEqual(len(relations), 2)
                variables = {variable for predicate in relations
                             for variable in (predicate.subject, predicate.object)}
                if entry["layout"] == "chain":
                    self.assertEqual(len(variables), 3)
                else:
                    self.assertEqual(len(variables), 2)

    def test_krel1_cells_have_expected_relation_kind_splits(self):
        expected = {
            1: {"left_of": 3, "above": 4},
            2: {"left_of": 3, "above": 3},
            3: {"left_of": 3, "above": 3},
        }
        for k_attr, counts in expected.items():
            entries = [entry for entry in self.manifest["templates"]
                       if entry["k_attr"] == k_attr and entry["k_rel"] == 1]
            self.assertEqual(len(entries), sum(counts.values()))
            actual = Counter(kind for entry in entries for kind in entry["relation_kinds"])
            self.assertEqual(actual, Counter(counts))

    def test_every_kattr1_cell_covers_all_colors_and_shapes(self):
        for k_rel in (0, 1, 2):
            entries = [entry for entry in self.manifest["templates"]
                       if entry["k_attr"] == 1 and entry["k_rel"] == k_rel]
            self.assertEqual(len(entries), 7)
            triangle_entries = [entry for entry in entries
                                if "triangle" in entry["shapes"]]
            self.assertEqual(len(triangle_entries), 1)
            colors = {color for entry in entries for color in entry["colors"]}
            shapes = {shape for entry in entries for shape in entry["shapes"]}
            self.assertEqual(colors, set(COLORS))
            self.assertEqual(shapes, set(SHAPES))

    def test_extra_triangle_candidates_cover_each_kattr1_cell(self):
        expected = {
            "attr1_rel0_attributes_7": ("attributes", []),
            "attr1_rel1_above_4": ("single_relation", ["above"]),
            "attr1_rel2_chain_4": ("chain", ["left_of", "above"]),
        }
        for name, (layout, relation_kinds) in expected.items():
            entry = self.entries[name]
            self.assertEqual(entry["k_attr"], 1)
            self.assertEqual(entry["layout"], layout)
            self.assertEqual(entry["colors"], [])
            self.assertEqual(entry["shapes"], ["triangle"])
            self.assertEqual(entry["relation_kinds"], relation_kinds)

    def test_rules_are_unique_under_variable_renaming_and_predicate_order(self):
        keys = [canonical_key(self.parsed[name]) for name in self.rules]
        self.assertEqual(len(set(keys)), 57)
        raw = {json.dumps(predicates, sort_keys=True) for predicates in self.rules.values()}
        self.assertEqual(len(raw), 57)

        predicates = self.parsed["attr3_rel2_chain_1"]
        mapping = {"a": "z", "b": "x", "c": "y"}
        renamed = []
        for predicate in predicates:
            if isinstance(predicate, Attribute):
                renamed.append(Attribute(mapping[predicate.variable],
                                         predicate.field, predicate.value))
            else:
                renamed.append(Relation(mapping[predicate.subject], predicate.kind,
                                        mapping[predicate.object]))
        reordered = tuple(reversed(renamed))
        self.assertEqual(canonical_key(predicates), canonical_key(reordered))
        self.assertNotEqual(canonical_key(predicates),
                            canonical_key(self.parsed["attr3_rel2_chain_2"]))

    def test_parser_compatibility_and_json_round_trip(self):
        parsed = load_rules(self.rules)
        self.assertEqual(list(parsed), list(self.rules))
        self.assertEqual(len(parsed), 57)
        for predicates in parsed.values():
            self.assertTrue(predicates)
        round_tripped = load_rules(json.loads(json.dumps(self.rules)))
        self.assertEqual(list(round_tripped), list(parsed))
        for name, predicates in round_tripped.items():
            self.assertEqual(predicates, parsed[name])

    def test_every_candidate_has_a_true_witness(self):
        for name, predicates in self.parsed.items():
            entry = self.entries[name]
            result = assess(predicates)
            self.assertTrue(result["feasible"], name)
            self.assertIsNone(result["reason"], name)
            self.assertEqual(result["witness"], entry["witness"], name)
            self.assertTrue(entry["witness_verified"], name)
            self.assertEqual(len(entry["witness"]), 3, name)
            scene = tuple(SceneObject(**obj) for obj in entry["witness"])
            self.assertTrue(satisfies(scene, predicates), name)
            self.assertEqual(len({(obj.x, obj.y) for obj in scene}), 3, name)

    def test_manifest_metadata_and_used_vocabulary(self):
        self.assertEqual(self.manifest["colors"], list(COLORS))
        self.assertEqual(self.manifest["shapes"], list(SHAPES))
        self.assertEqual(self.manifest["relation_kinds"], ["left_of", "above"])
        self.assertEqual(self.manifest["object_count"], 3)
        self.assertEqual(self.manifest["max_variables"], 3)
        self.assertEqual(self.manifest["used"]["colors"], list(COLORS))
        self.assertEqual(self.manifest["used"]["shapes"], list(SHAPES))
        self.assertEqual(self.manifest["used"]["relation_kinds"], ["left_of", "above"])
        self.assertEqual(self.manifest["used"]["layouts"],
                         ["attributes", "single_relation", "chain", "same_pair"])
        self.assertEqual(self.manifest["used"]["attribute_fields"], ["color", "shape"])
        self.assertEqual(self.manifest["witnesses_verified"], 57)
        self.assertTrue(self.manifest["provisional"])
        self.assertTrue(self.manifest["limitations"])
        for name, predicates in self.parsed.items():
            entry = self.entries[name]
            self.assertEqual(entry["predicate_count"], len(predicates), name)
            self.assertEqual(entry["attribute_count"],
                             sum(isinstance(p, Attribute) for p in predicates), name)
            self.assertEqual(entry["relation_count"],
                             sum(isinstance(p, Relation) for p in predicates), name)
            self.assertLessEqual(entry["variable_count"], 3, name)
            self.assertEqual(entry["relation_kinds"],
                             [kind for kind in ("left_of", "above")
                              if any(isinstance(p, Relation) and p.kind == kind
                                     for p in predicates)], name)
            self.assertEqual(entry["colors"],
                             [color for color in COLORS
                              if any(isinstance(p, Attribute) and p.field == "color"
                                     and p.value == color for p in predicates)], name)
            self.assertEqual(entry["shapes"],
                             [shape for shape in SHAPES
                              if any(isinstance(p, Attribute) and p.field == "shape"
                                     and p.value == shape for p in predicates)], name)

    def test_build_is_reproducible(self):
        rules, manifest = build_templates()
        self.assertEqual(rules, self.rules)
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(json.dumps(rules), json.dumps(self.rules))
        self.assertEqual(json.dumps(manifest), json.dumps(self.manifest))

    def test_validation_rejects_a_semantic_duplicate(self):
        rules = {name: list(predicates) for name, predicates in self.rules.items()}
        rules["attr1_rel0_attributes_2"] = list(rules["attr1_rel0_attributes_1"])
        with self.assertRaisesRegex(ValueError, "Duplicate template semantics"):
            validate_templates(rules, self.manifest)

    def test_layout_labels_must_match_relation_graph(self):
        for k_attr in (1, 2, 3):
            manifest = copy.deepcopy(self.manifest)
            entries = [entry for entry in manifest["templates"]
                       if entry["k_attr"] == k_attr and entry["k_rel"] == 2]
            chain = next(entry for entry in entries if entry["layout"] == "chain")
            same_pair = next(entry for entry in entries if entry["layout"] == "same_pair")
            chain["layout"], same_pair["layout"] = "same_pair", "chain"
            with self.subTest(k_attr=k_attr), self.assertRaisesRegex(ValueError, "layout"):
                validate_templates(self.rules, manifest)

        manifest = copy.deepcopy(self.manifest)
        single = next(entry for entry in manifest["templates"]
                      if entry["layout"] == "single_relation")
        single["layout"] = "attributes"
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_templates(self.rules, manifest)

        manifest = copy.deepcopy(self.manifest)
        attributes = next(entry for entry in manifest["templates"]
                          if entry["layout"] == "attributes")
        attributes["layout"] = "single_relation"
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_templates(self.rules, manifest)

        manifest = copy.deepcopy(self.manifest)
        chain = next(entry for entry in manifest["templates"]
                     if entry["layout"] == "chain")
        chain["layout"] = "single_relation"
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_templates(self.rules, manifest)

        manifest = copy.deepcopy(self.manifest)
        same_pair = next(entry for entry in manifest["templates"]
                         if entry["layout"] == "same_pair")
        same_pair["layout"] = "chain"
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_templates(self.rules, manifest)

        rules = {name: list(predicates) for name, predicates in self.rules.items()}
        rules["attr1_rel2_chain_1"] = [
            {"type": "attribute", "variable": "a", "field": "color", "value": "red"},
            {"type": "relation", "subject": "a", "kind": "left_of", "object": "b"},
            {"type": "relation", "subject": "c", "kind": "left_of", "object": "b"},
        ]
        with self.assertRaisesRegex(ValueError, "layout"):
            validate_templates(rules, self.manifest)

    def test_manifest_rejects_noncanonical_overlapping_or_out_of_bounds_witnesses(self):
        overlapping = copy.deepcopy(self.manifest)
        entry = next(entry for entry in overlapping["templates"]
                     if entry["name"] == "attr1_rel0_attributes_1")
        for obj in entry["witness"]:
            obj["x"] = 12
            obj["y"] = 12
        scene = tuple(SceneObject(**obj) for obj in entry["witness"])
        self.assertTrue(satisfies(scene, self.parsed[entry["name"]]))
        with self.assertRaisesRegex(ValueError, "witness"):
            validate_templates(self.rules, overlapping)

        out_of_bounds = copy.deepcopy(self.manifest)
        entry = next(entry for entry in out_of_bounds["templates"]
                     if entry["name"] == "attr1_rel0_attributes_1")
        entry["witness"][0]["x"] = 1000
        scene = tuple(SceneObject(**obj) for obj in entry["witness"])
        self.assertTrue(satisfies(scene, self.parsed[entry["name"]]))
        with self.assertRaisesRegex(ValueError, "witness"):
            validate_templates(self.rules, out_of_bounds)

    def test_main_does_not_write_when_build_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rules_path = directory / "rules.json"
            manifest_path = directory / "manifest.json"
            with patch("exposure.templates.build_templates", side_effect=ValueError("bad build")):
                with self.assertRaisesRegex(ValueError, "bad build"):
                    main(["--rules-output", str(rules_path),
                          "--manifest-output", str(manifest_path)])
            self.assertFalse(rules_path.exists())
            self.assertFalse(manifest_path.exists())

    def test_cli_rejects_same_or_aliased_output_paths_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            target = directory / "out.json"
            nested = directory / "nested"
            nested.mkdir()
            alias = nested / ".." / "out.json"
            link = directory / "link.json"
            link.symlink_to(target)
            for rules_path, manifest_path in (
                (target, target),
                (target, alias),
                (target, link),
            ):
                with self.subTest(rules=str(rules_path), manifest=str(manifest_path)):
                    result = subprocess.run(
                        [sys.executable, "-m", "exposure.templates",
                         "--rules-output", str(rules_path),
                         "--manifest-output", str(manifest_path)],
                        check=False, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("different", result.stderr)
                    self.assertFalse(target.exists())

    def test_cli_writes_deterministic_valid_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first_rules = directory / "first-rules.json"
            first_manifest = directory / "first-manifest.json"
            second_rules = directory / "second-rules.json"
            second_manifest = directory / "second-manifest.json"
            for rules_path, manifest_path in ((first_rules, first_manifest),
                                              (second_rules, second_manifest)):
                subprocess.run([sys.executable, "-m", "exposure.templates",
                                "--rules-output", str(rules_path),
                                "--manifest-output", str(manifest_path)],
                               check=True, capture_output=True, text=True)
            first_rules_text = first_rules.read_text(encoding="utf-8")
            second_rules_text = second_rules.read_text(encoding="utf-8")
            first_manifest_text = first_manifest.read_text(encoding="utf-8")
            second_manifest_text = second_manifest.read_text(encoding="utf-8")
            self.assertEqual(first_rules_text, second_rules_text)
            self.assertEqual(first_manifest_text, second_manifest_text)
            rules = load_rules(json.loads(first_rules_text))
            manifest = json.loads(first_manifest_text)
            self.assertEqual(len(rules), 57)
            self.assertEqual(manifest["total_templates"], 57)
            self.assertEqual(manifest["counts"]["by_cell"]["attr1_rel2"], 7)
            self.assertEqual(manifest["counts"]["by_cell"]["attr3_rel2"], 6)
            self.assertEqual(manifest["witnesses_verified"], 57)


if __name__ == "__main__":
    unittest.main()
