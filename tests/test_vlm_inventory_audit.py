import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from exposure import vlm_audit
from exposure.vlm_inventory_audit import decide, prepare, run, score


class InventoryAuditTests(unittest.TestCase):
    def test_decision_requires_bound_object(self):
        data = {'objects': [{'color': 'red', 'shape': 'square'},
                            {'color': 'blue', 'shape': 'circle'}], 'uncertain': False}
        self.assertEqual(decide(json.dumps(data), 'red'), 'no')
        self.assertEqual(decide(json.dumps(data), 'blue'), 'yes')
        data['uncertain'] = True
        self.assertEqual(decide(json.dumps(data), 'red'), 'unsure')
        self.assertEqual(decide('{bad', 'red'), 'unsure')

    def test_blinded_inventory_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pack = root / 'pack'; private = pack / 'private'
            (private / 'images').mkdir(parents=True)
            image = Image.new('RGB', (64, 64), 'white'); buf = io.BytesIO(); image.save(buf, 'PNG')
            raw = buf.getvalue(); (private / 'images' / 'item.png').write_bytes(raw)
            (private / 'provenance.json').write_text(json.dumps({'items': {'item': {
                'color': 'red', 'image_sha256': vlm_audit.sha(raw), 'pixel_sha256': vlm_audit.sha(image.tobytes())}}}))
            output = root / 'out'; plan = prepare(pack, output)
            self.assertEqual(plan['inventory_options']['num_predict'], 256)
            calls = []
            def request(url, payload=None):
                calls.append(payload)
                if payload is None: return {'models': [{'name': 'model', 'digest': 'digest'}]}
                self.assertNotIn('red circle', payload['messages'][0]['content'])
                return {'message': {'content': '{"objects":[],"uncertain":false}'}}
            with patch('exposure.vlm_audit.request_json', side_effect=request):
                run(pack, output, url='http://127.0.0.1:11435', model='model')
                run(pack, output, url='http://127.0.0.1:11435', model='model')
            self.assertEqual(sum(value is not None for value in calls), 1)
            with patch('exposure.vlm_audit.score', return_value={'counts': {}}):
                self.assertEqual(score(pack, output), {'counts': {}})
            path = output / 'results' / 'item.json'; record = json.loads(path.read_text())
            record['decision'] = 'yes'; path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): score(pack, output)


if __name__ == '__main__':
    unittest.main()
