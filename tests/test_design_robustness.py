"""Regression checks for label provenance and image-cluster resampling safeguards."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The historical experiment bundle is intentionally local and not part of Git.
HISTORICAL = ROOT / 'output/experiments/论文关键结论验证_20261106/common.py'
if HISTORICAL.exists():
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('design_robustness', ROOT / 'scripts/run_design_robustness_20261110.py')
    dr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dr)
else:
    dr = None


@unittest.skipIf(dr is None, 'Historical frozen experiment bundle is unavailable')
class RobustnessSafeguards(unittest.TestCase):
    def test_cell_comparison_uses_meaning_not_enumeration(self):
        a, b = (1, 10, 0), (2, 20, 0)
        self.assertEqual(dr.cell_change_fraction([a, b], [0, 1], [b, a], [1, 0]), 0)
        self.assertEqual(dr.cell_change_fraction([a], [0], [b], [0]), 1)

    def test_quota_structures_and_pool_proportional_control(self):
        tiers = np.repeat(np.arange(1, 6), 80)
        self.assertEqual(dr.tier_allocation(tiers, 'equal'), [40] * 5)
        self.assertEqual(dr.tier_allocation(tiers, 'pool_proportional'), [40] * 5)
        self.assertEqual(dr.tier_allocation(tiers, 'risk_weighted'), [13, 27, 40, 53, 67])
        uneven = np.repeat(np.arange(1, 6), [40, 60, 80, 100, 120])
        self.assertEqual(dr.tier_allocation(uneven, 'pool_proportional'), [20, 30, 40, 50, 60])

    def test_flip_preserves_unknown_and_geometric_scale_and_set_constraints(self):
        states = [('shared', 'target_scale', 'small'),
                  ('shared', 'enum', 'a'), ('shared', 'enum', 'b'),
                  ('specific', 'set', 'none'), ('specific', 'set', 'x'), ('specific', 'set', 'y')]
        X = np.tile([True, True, False, True, False, False], (100, 1))
        X[0, 1:] = False  # unknown is unencoded and must stay so
        metadata = {('shared', 'enum'): dict(value_type='enum'),
                    ('specific', 'set'): dict(value_type='set', exclusive_values=['none'])}
        Y = dr.corrupt_matrix(X, states, dr.field_groups(states), metadata, np.random.default_rng(4), 1)
        np.testing.assert_array_equal(Y[:, 0], X[:, 0])
        np.testing.assert_array_equal(Y[0], X[0])
        self.assertTrue((Y[1:, 2] & ~Y[1:, 1]).all())
        self.assertFalse((Y[:, 3] & Y[:, 4:].any(1)).any())
        self.assertTrue((Y[1:, 3:].sum(1) >= 1).all())
        self.assertTrue((Y & ~X).any())
        self.assertTrue((X & ~Y).any())

    def test_set_type_is_not_inferred_from_number_of_observed_values(self):
        states = [('specific', 'set', v) for v in ['x', 'y', 'z']]
        X = np.tile([True, False, False], (100, 1))
        Y = dr.corrupt_matrix(X, states, dr.field_groups(states),
                              {('specific', 'set'): dict(value_type='set')}, np.random.default_rng(8), 1)
        self.assertTrue((Y.sum(1) == 2).any(), 'Singleton-valued set fields must allow additions')

    def test_duplicate_image_weights_do_not_reopen_legacy_eligibility(self):
        A = np.r_[np.ones(20), np.zeros(20)][:, None]
        B = 1 - A
        correct = np.r_[np.zeros(20), np.ones(20)]
        W = np.ones((2, 40))
        W[1] = 0
        W[1, [0, 20]] = 20  # 20 target draws per group, but only one distinct image each
        effects, legacy = dr.bootstrap_contrasts(W, A, B, correct)
        np.testing.assert_allclose(effects[:, 0], [-100, -100])
        self.assertTrue(legacy[0, 0])
        self.assertFalse(legacy[1, 0])
        summary = dr.contrast_record(effects[:, 0], legacy[:, 0])
        self.assertEqual(summary['valid_replicates'], 2)
        self.assertEqual(summary['legacy_eligible_replicates'], 1)
        self.assertEqual(summary['boot_leq_minus10'], 1)
        self.assertEqual(summary['boot_leq_minus10_regated'], .5)

    def test_empty_bootstrap_group_is_missing_not_zero(self):
        effects, legacy = dr.bootstrap_contrasts(np.array([[1., 0.]]),
                                                 np.array([[1.], [0.]]),
                                                 np.array([[0.], [1.]]), np.array([0., 1.]))
        self.assertTrue(np.isnan(effects[0, 0]))
        self.assertFalse(legacy[0, 0])

    @unittest.skipUnless((ROOT / 'artifacts/other_classes_specific_experiment_20261018').exists(), 'Local matrices unavailable')
    def test_six_class_clean_identity_and_native_risk_provenance(self):
        contexts = dr.frozen.contexts()
        self.assertEqual(len(dr.clean_rescore_audit(contexts)), 6)
        c = 'horse'
        ct = contexts[c]
        native = dr.native_features()
        features = [native[s] for s in ct['ids']]
        X, risk = dr.perturb_inputs(c, ct, features, np.random.default_rng(5), 1)
        for f, old in zip(risk, features):
            if old['leg_visibility'] not in dr.UNKNOWN:
                self.assertNotEqual(f['leg_visibility'], old['leg_visibility'])
            self.assertIn(f['head_orientation'], set(dr.CONFIG['maps']['head_orientation']) | dr.UNKNOWN)
        # The coverage head-orientation schema includes profile/up_or_down, which are not risk values.
        self.assertIn(('specific', 'head_orientation', 'profile'), ct['states'])
        self.assertFalse(np.array_equal(X, ct['matrix']))


if __name__ == '__main__':
    unittest.main()
