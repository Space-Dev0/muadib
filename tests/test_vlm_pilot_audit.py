import json
import unittest

from exposure.vlm_pilot_audit import decide


class PilotInventoryTests(unittest.TestCase):
    def test_attribute_only_questions(self):
        inventory = {'objects': [{'color': 'red', 'shape': 'triangle'},
                                 {'color': 'blue', 'shape': 'square'}], 'uncertain': True}
        self.assertEqual(decide(json.dumps(inventory), 'red'), 'yes')
        self.assertEqual(decide(json.dumps(inventory), 'square'), 'yes')
        inventory['objects'][0]['color'] = 'yellow'
        self.assertEqual(decide(json.dumps(inventory), 'red'), 'no')
        inventory['objects'][0]['color'] = 'unknown'
        self.assertEqual(decide(json.dumps(inventory), 'red'), 'unsure')

    def test_invalid_output_and_rule(self):
        self.assertEqual(decide('not json', 'red'), 'unsure')
        with self.assertRaises(ValueError):
            decide('{"objects":[],"uncertain":false}', 'relation')


if __name__ == '__main__':
    unittest.main()
