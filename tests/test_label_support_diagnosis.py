"""Hand-constructed counterexamples for support vs weak-gap accounting."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_label_support_diagnosis_20261112 import decompose


class SupportMechanismTests(unittest.TestCase):
    def test_all_four_transitions_and_unchanged_confirmation(self):
        # New eligibility, new despite eligibility, lost eligibility,
        # lost despite eligibility, unchanged confirmed.
        em = np.array([1, 1, 0, 1, 1], bool)
        eb = np.array([0, 1, 1, 1, 1], bool)
        cm = np.array([1, 1, 0, 0, 1], bool)
        cb = np.array([0, 0, 1, 1, 1], bool)
        counts, masks = decompose(em, eb, cm, cb)
        self.assertEqual([counts[k] for k in ['new_from_eligibility', 'new_on_common_eligible',
                                             'lost_from_eligibility', 'lost_on_common_eligible']], [1, 1, 1, 1])
        self.assertEqual(counts['total_difference'], 0)
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks.values()), [1, 1, 1, 1, 0])

    def test_eligibility_gain_can_hide_loss_on_common_states(self):
        counts, _ = decompose([1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 1])
        self.assertEqual(counts['total_difference'], 1)
        self.assertEqual(counts['eligibility_net'], 2)
        self.assertEqual(counts['common_gap_net'], -1)

    def test_zero_eligible_has_no_confirmed_and_invalid_counts_are_rejected(self):
        counts, _ = decompose([0, 0], [0, 0], [0, 0], [0, 0])
        self.assertTrue(all(v == 0 for v in counts.values()))
        with self.assertRaises(AssertionError):
            decompose([0], [1], [1], [0])


if __name__ == '__main__':
    unittest.main()
