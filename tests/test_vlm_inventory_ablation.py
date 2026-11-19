import json
import unittest
from exposure.vlm_inventory_ablation import decision


class InventoryAblationTests(unittest.TestCase):
    def test_ignores_uncertainty_but_preserves_binding(self):
        data = {'objects': [{'color': 'red', 'shape': 'square'},
                            {'color': 'blue', 'shape': 'circle'}], 'uncertain': True}
        self.assertEqual(decision(json.dumps(data), 'red'), 'no')
        self.assertEqual(decision(json.dumps(data), 'blue'), 'yes')

    def test_unknown_or_malformed_without_witness_abstains(self):
        self.assertEqual(decision('invalid', 'red'), 'unsure')
        self.assertEqual(decision(json.dumps({'objects': [{'color': 'red', 'shape': 'unknown'}],
                                              'uncertain': False}), 'red'), 'unsure')


if __name__ == '__main__':
    unittest.main()
