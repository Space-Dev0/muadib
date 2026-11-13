import unittest

from exposure.audit_followup import combine


def item(ident):
    return {"sample_id": ident, "image_path": f"{ident}.png", "rule_name": "red",
            "predicates": [{"type": "attribute", "variable": "a", "field": "color", "value": "red"}],
            "k_attr": 1, "k_rel": 0}


class FollowupTests(unittest.TestCase):
    def test_selected_disagreement_and_qualified_answer(self):
        manifest = {"samples": [item("a"), item("b"), item("c")]}
        reviews = [{"a": True, "b": False}, {"a": False, "c": True}]
        predictions = {"a": None, "b": True, "c": True}
        rows = [{"sample_id": "a", "decision": "yes but slightly", "notes": ""},
                {"sample_id": "b", "decision": "no", "notes": ""}]
        result = combine(manifest, reviews, predictions, rows)
        self.assertEqual(result["followup_ids"], ["a", "b"])
        self.assertEqual(result["overall"]["fp"], 1)
        self.assertEqual(result["overall"]["rejected_human_yes"], 1)
        self.assertEqual(reviews[0]["b"], False)
        for altered in (rows[:1], rows + [rows[0]], [dict(rows[0], decision="maybe"), rows[1]]):
            with self.subTest(rows=altered), self.assertRaises(ValueError):
                combine(manifest, reviews, predictions, altered)


if __name__ == "__main__":
    unittest.main()
