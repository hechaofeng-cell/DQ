"""Regression checks for the Open Images dog supplement experiment."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_openimages_v7_four_model_experiment import check_manifests, slice_recall
from select_openimages_dog_supplements import counts_for, coverage, select_gap


class DogSupplementTests(unittest.TestCase):
    def test_greedy_selection_uses_actual_marginal_coverage(self):
        features = ["coat", "pose"]
        baseline = [{"image_id": "base", "coat": "brown", "pose": "sit"}]
        candidates = [
            {"image_id": "common", "coat": "brown", "pose": "sit"},
            {"image_id": "rare", "coat": "roan", "pose": "jump"},
        ]
        counts = counts_for(baseline, features, {"unknown", "other"})
        selected, audit = select_gap(candidates, counts, features, {"unknown", "other"}, 1, cap=1)
        self.assertEqual(selected[0]["image_id"], "rare")
        self.assertEqual(audit[0]["marginal_state_hits"], 2)
        self.assertEqual(coverage(counts, [("coat", "brown"), ("coat", "roan")], 1), 0.5)

    def test_manifest_preflight_rejects_changed_baseline_and_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            crop = Path(directory) / "crop.jpg"
            crop.write_bytes(b"placeholder")

            def row(image_id, label, class_name):
                return {
                    "sample_id": image_id, "image_id": image_id, "label": str(label),
                    "class_name": class_name, "crop_path": str(crop),
                }

            base = [row("dog", 0, "dog"), row("cat", 1, "cat")]
            random = base + [row("random_dog", 0, "dog")]
            gap = base + [row("gap_dog", 0, "dog")]
            val = [row("val_dog", 0, "dog"), row("val_cat", 1, "cat")]
            test = [row("test_dog", 0, "dog"), row("test_cat", 1, "cat")]
            difficulty = [row("hard_dog", 0, "dog")]
            config = {"classes": {"dog": "id0", "cat": "id1"}, "counts": {
                "train_base_by_class": {"dog": 1, "cat": 1},
                "supplement_budget": 1, "validation_per_class": 1,
                "test_per_class": 1, "difficulty_test_dog": 1,
            }}
            conditions = {"base": base, "random300": random, "gap300": gap}
            check_manifests(conditions, val, test, difficulty, config)
            with self.assertRaisesRegex(ValueError, "unchanged baseline"):
                check_manifests({**conditions, "gap300": [base[0], gap[-1], row("other", 0, "dog")]},
                                val, test, difficulty, config)
            with self.assertRaisesRegex(ValueError, "image overlap"):
                check_manifests(conditions, val, test, [row("random_dog", 0, "dog")], config)

    def test_difficulty_recall_is_not_reported_for_empty_slice(self):
        predictions = [{"class_name": "dog", "correct": 1, "is_occluded": "0"}]
        self.assertIsNone(slice_recall(predictions, "is_occluded"))


if __name__ == "__main__":
    unittest.main()
