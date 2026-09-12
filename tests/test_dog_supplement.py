import unittest

from scripts.run_dog_targeted_supplement import coverage_band, target_count


class SupplementCoverageRuleTests(unittest.TestCase):
    def test_coverage_bands(self):
        self.assertEqual(coverage_band(0), "missing")
        self.assertEqual(coverage_band(1), "sparse")
        self.assertEqual(coverage_band(9), "sparse")
        self.assertEqual(coverage_band(10), "weak")
        self.assertEqual(coverage_band(29), "weak")
        self.assertEqual(coverage_band(30), "covered")

    def test_targets_do_not_force_rare_states_to_thirty(self):
        self.assertEqual(target_count(0), 10)
        self.assertEqual(target_count(9), 10)
        self.assertEqual(target_count(10), 30)
        self.assertEqual(target_count(29), 30)
        self.assertEqual(target_count(30), 30)


if __name__ == "__main__":
    unittest.main()
