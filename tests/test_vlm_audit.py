import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from exposure.vlm_audit import parse_answer, prepare, run, score


class VisionAuditTests(unittest.TestCase):
    def test_parse_invalid_output_as_unsure(self):
        self.assertEqual(parse_answer('{"answer":"yes"}'), 'yes')
        for value in ('bad', '{}', '{"answer":"maybe"}', '[]'):
            self.assertEqual(parse_answer(value), 'unsure')

    def test_frozen_blind_run_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pack = root / 'pack'; private = pack / 'private'
            (private / 'images').mkdir(parents=True)
            image = Image.new('RGB', (64, 64), 'white'); buffer = io.BytesIO(); image.save(buffer, 'PNG')
            raw = buffer.getvalue(); digest = lambda b: hashlib.sha256(b).hexdigest()
            (private / 'images' / 'item.png').write_bytes(raw)
            meta = {'items': {'item': {'color': 'red', 'image_sha256': digest(raw),
                                      'pixel_sha256': digest(image.tobytes())}}}
            (private / 'provenance.json').write_text(json.dumps(meta))
            output = root / 'out'; plan = prepare(pack, output)
            self.assertNotIn('target_circle', json.dumps(plan))
            calls = []
            def fake_request(url, payload=None):
                calls.append((url, payload))
                if payload is None:
                    return {'models': [{'name': 'model', 'digest': 'digest'}]}
                self.assertEqual(payload['messages'][0]['content'].count('red'), 2)
                return {'message': {'content': '{"answer":"yes"}'}, 'done_reason': 'stop'}
            with patch('exposure.vlm_audit.request_json', side_effect=fake_request):
                run(pack, output, url='http://127.0.0.1:11435', model='model')
                run(pack, output, url='http://127.0.0.1:11435', model='model')
            self.assertEqual(sum(payload is not None for _, payload in calls), 1)
            truth = {'item': {'unanimous': {'target_circle': 'yes'}}}
            human = {'provenance_sha256': plan['provenance_sha256'], 'prediction_sha256': 'p',
                     'review_sha256': ['r'] * 4, 'per_image': truth}
            (pack / 'score.json').write_text(json.dumps(human))
            with patch('exposure.mosaic_matched_eval_score.score', return_value=human):
                self.assertEqual(score(pack, output)['counts']['yes']['yes'], 1)
            result = output / 'results' / 'item.json'; data = json.loads(result.read_text())
            data['decision'] = 'no'; result.write_text(json.dumps(data))
            with patch('exposure.mosaic_matched_eval_score.score', return_value=human):
                with self.assertRaises(ValueError): score(pack, output)

    def test_remote_endpoint_refused(self):
        with self.assertRaises(ValueError):
            run(Path('.'), Path('.'), url='http://example.org', model='model')


if __name__ == '__main__':
    unittest.main()
