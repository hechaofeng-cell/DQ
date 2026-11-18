"""Integrity checks for cross-annotator comparisons and unknown controls."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_cross_vlm_analysis_20261112 as analysis
from verify_cross_vlm_20261112 import validate_specific


class CrossVLMIntegrityTests(unittest.TestCase):
    def test_unknown_is_excluded_from_known_complement(self):
        matrix = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
        states = [('shared', 'visible', 'yes'), ('shared', 'visible', 'no')]
        controls = analysis.value_known_mask(matrix, states)
        np.testing.assert_array_equal(controls, [[0, 1], [1, 0], [0, 0]])

    def test_same_name_in_different_namespaces_keeps_separate_inputs(self):
        rows = [dict(sample_id='x', class_name='horse', bbox_area_ratio='.05')]
        states = [('shared', 'head_visibility', 'clear'), ('specific', 'head_visibility', 'clear')]
        x = analysis.matrix_for(rows, {'x': {'head_visibility': 'clear'}},
                               {'x': {'head_visibility': 'unknown'}}, states)
        np.testing.assert_array_equal(x, [[True, False]])

    def test_set_order_not_counted_as_disagreement_but_unknown_is_reported(self):
        result = analysis.agreement([['a', 'b'], ['unknown'], ['a']],
                                    [['b', 'a'], ['unknown'], ['unknown']])
        self.assertAlmostEqual(result['exact_agreement_pct'], 200 / 3)
        self.assertEqual(result['both_known_n'], 1)
        self.assertEqual(result['both_known_agreement_pct'], 100)
        self.assertEqual(result['A_known_B_unknown'], 1)
        self.assertEqual(result['both_unknown'], 1)

    def test_tiny_groups_do_not_become_weak_conditions(self):
        x = np.array([[True]] * 19 + [[False]] * 21)
        control = ~x
        correct = np.array([0.] * 19 + [1.] * 21)
        *_, eligible, weak = analysis.reference_stats(x, control, correct)
        self.assertFalse(eligible[0]); self.assertFalse(weak[0])

    def test_set_scalar_and_exclusive_value_mixing_are_rejected(self):
        fields = [dict(feature_id='cover', value_type='set',
                       values=[{'value_id': 'unknown'}, {'value_id': 'red'}],
                       exclusive_values=['unknown'])]
        validate_specific({'cover': ['unknown']}, fields)
        with self.assertRaisesRegex(ValueError, 'JSON array'):
            validate_specific({'cover': 'unknown'}, fields)
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            validate_specific({'cover': ['unknown', 'red']}, fields)


if __name__ == '__main__':
    unittest.main()
