import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from exposure import vlm_audit
from exposure.vlm_relation_audit import PROMPTS, prepare, run, score


class RelationAuditTests(unittest.TestCase):
    def test_prompts_preserve_distinct_center_semantics(self):
        for text in PROMPTS.values():
            self.assertIn('distinct', text)
            self.assertIn('center', text)
            self.assertIn('strictly', text)

    def test_frozen_run_resume_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; private = source / 'private'
            (private / 'images').mkdir(parents=True)
            buf = io.BytesIO(); Image.new('RGB', (64, 64), 'white').save(buf, 'PNG')
            raw = buf.getvalue(); digest = vlm_audit.sha(raw)
            items = {}; rules = []
            for index in range(150):
                name = f'item-{index:03d}'; rule = list(PROMPTS)[index // 75]
                (private / 'images' / f'{name}.png').write_bytes(raw)
                items[name] = {'sha256': digest}; rules.append({'sample_id': name, 'rule_name': rule})
            (private / 'provenance.json').write_text(json.dumps({'images': items}))
            (private / 'all-rules.json').write_text(json.dumps({'samples': rules}))
            output = root / 'out'; human_score = root / 'human.json'
            human_score.write_text(json.dumps({'review_sha256': {'reviewer-1': 'digest'}}))
            checked = {'review_sha256': {'reviewer-1': 'digest'}}
            with patch('exposure.vlm_relation_audit._verify_source', return_value=checked):
                plan = prepare(source, output)
                self.assertEqual(len(plan['items']), 150)
                self.assertNotIn('truth', json.dumps(plan))
                calls = []
                def fake_request(url, payload=None):
                    calls.append(payload)
                    if payload is None: return {'models': [{'name': 'model', 'digest': 'digest'}]}
                    self.assertNotIn('judgment', payload['messages'][0]['content'])
                    return {'message': {'content': '{"answer":"no"}'}, 'done_reason': 'stop'}
                with patch('exposure.vlm_audit.request_json', side_effect=fake_request):
                    with redirect_stdout(io.StringIO()):
                        run(source, output, url='http://127.0.0.1:11435', model='model')
                        run(source, output, url='http://127.0.0.1:11435', model='model')
                self.assertEqual(sum(value is not None for value in calls), 150)
                with patch('exposure.vlm_relation_audit.read_reviews',
                           side_effect=[{name: False for name in items}, {}, {}, {}]):
                    result = score(source, output, human_score)
                self.assertEqual(sum(x['no']['no'] for x in result['counts'].values()), 150)

    def test_remote_endpoint_refused(self):
        with self.assertRaises(ValueError):
            run(Path('.'), Path('.'), url='http://example.org', model='model')


if __name__ == '__main__':
    unittest.main()
