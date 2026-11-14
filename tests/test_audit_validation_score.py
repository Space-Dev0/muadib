import hashlib
import json
import unittest
from unittest.mock import patch

from exposure.audit_validation_score import evaluate
from tests import test_audit_pilot_score


class ValidationScoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_audit_pilot_score.AuditPilotScoreTests(methodName="test_coverage_overlap_and_agreement_without_consensus")
        self.fixture.setUp()
        self.pack, self.reviews = self.fixture.pack, self.fixture.reviews
        private = self.pack / "private"
        hashes = {}
        for kind, plural in (("rule", "rules"), ("predicate", "predicates")):
            path = private / f"variable-count-{plural}-predictions.json"
            path.write_text((private / f"{kind}-predictions.json").read_text())
            hashes[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.hash_patch = patch("exposure.audit_validation_score.CANDIDATE_SHA256", hashes)
        self.hash_patch.start()

    def tearDown(self):
        self.hash_patch.stop()
        self.fixture.tearDown()

    def test_disagreement_excluded_and_single_reviews_counted(self):
        result = evaluate(self.pack, self.reviews)
        rule = result["results"]["rule"]
        self.assertEqual(rule["unique_questions"], 24)
        self.assertEqual(rule["unanimous_or_single_review"], 23)
        self.assertEqual(rule["single_reviewer_questions"], 16)
        self.assertEqual(len(rule["disputed_ids_excluded"]), 1)
        self.assertEqual(rule["scores"]["candidate"]["reviewed"], 23)
        self.assertIn("by_rule", rule["scores"]["candidate"])
        self.assertIn("accepted_human_yes", rule["scores"]["candidate"]["overall"])

    def test_modified_candidate_is_rejected(self):
        (self.pack / "private" / "variable-count-rules-predictions.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "candidate hash mismatch"):
            evaluate(self.pack, self.reviews)

    def test_rejects_extra_candidate_id_even_if_digest_matches(self):
        path = self.pack / "private" / "variable-count-rules-predictions.json"
        data = json.loads(path.read_text())
        data["unexpected"] = True
        path.write_text(json.dumps(data))
        with patch.dict("exposure.audit_validation_score.CANDIDATE_SHA256", rule=hashlib.sha256(path.read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError, "Invalid candidate rule predictions"):
                evaluate(self.pack, self.reviews)


if __name__ == "__main__":
    unittest.main()
