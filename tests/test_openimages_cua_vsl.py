"""Regression tests for coverage-utility aligned visual-state learning."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openimages_cua_vsl import (  # noqa: E402
    StateSchema,
    VisualStateMultiTaskModel,
    class_balanced_state_weights,
    visual_state_loss,
)
from select_openimages_cua_supplement import select_two_stage_cua  # noqa: E402


class CuaVslTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = StateSchema.from_path(ROOT / "configs/dog_feature_schema_v3_1.json")

    def feature_row(self):
        row = {}
        for field in self.schema.enum_fields:
            row[field.feature_id] = field.values[0]
        for field in self.schema.set_fields:
            row[field.feature_id] = '["none"]'
        row["coat_primary_color"] = "unknown"
        row["occlusion_parts"] = '["head", "torso"]'
        return row

    def test_schema_has_twenty_one_enum_and_two_set_fields(self):
        self.assertEqual(len(self.schema.fields), 23)
        self.assertEqual(len(self.schema.enum_fields), 21)
        self.assertEqual(len(self.schema.set_fields), 2)

    def test_encoding_masks_unknown_and_supports_multihot_sets(self):
        encoded = self.schema.encode(self.feature_row(), True)
        color_index = next(
            index for index, field in enumerate(self.schema.enum_fields)
            if field.feature_id == "coat_primary_color"
        )
        self.assertEqual(encoded["enum_targets"][color_index].item(), -100)
        field = next(field for field in self.schema.set_fields if field.feature_id == "occlusion_parts")
        start, end = self.schema.set_offsets[field.feature_id]
        target = encoded["set_targets"][start:end]
        self.assertEqual(int(target.sum().item()), 2)
        self.assertEqual(encoded["set_masks"][0].item(), 1.0)

    def test_none_is_exclusive_for_set_fields(self):
        row = self.feature_row()
        row["occlusion_parts"] = '["none", "head"]'
        with self.assertRaisesRegex(ValueError, "must be exclusive"):
            self.schema.encode(row, True)

    def test_non_dog_has_no_valid_state_targets(self):
        encoded = self.schema.encode(None, False)
        self.assertTrue(torch.all(encoded["enum_targets"] == -100))
        self.assertEqual(encoded["set_masks"].sum().item(), 0.0)
        self.assertEqual(encoded["set_targets"].sum().item(), 0.0)

    def test_sampler_preserves_top_level_class_mass(self):
        rows = [
            {"sample_id": "dog_a", "image_id": "a", "label": "0", "class_name": "dog"},
            {"sample_id": "dog_b", "image_id": "b", "label": "0", "class_name": "dog"},
            {"sample_id": "cat_a", "image_id": "c", "label": "1", "class_name": "cat"},
        ]
        feature_a = self.feature_row()
        feature_b = self.feature_row()
        feature_b["coat_pattern"] = "solid"
        priorities = {("coat_pattern", "solid"): 1.0}
        weights = class_balanced_state_weights(
            rows, self.schema, {"dog_a": feature_a, "dog_b": feature_b}, priorities,
            alpha=2.0, max_multiplier=3.0,
        )
        self.assertAlmostEqual(weights[0] + weights[1], weights[2], places=7)

    def test_masked_state_loss_is_zero_for_non_dog_batch(self):
        batch_size = 2
        output = {
            "class_logits": torch.randn(batch_size, 5, requires_grad=True),
            "enum_logits": [
                torch.randn(batch_size, len(field.values), requires_grad=True)
                for field in self.schema.enum_fields
            ],
            "set_logits": [
                torch.randn(batch_size, len(field.values), requires_grad=True)
                for field in self.schema.set_fields
            ],
        }
        batch = {
            "enum_targets": torch.full((batch_size, len(self.schema.enum_fields)), -100),
            "set_targets": torch.zeros(batch_size, self.schema.set_target_size),
            "set_masks": torch.zeros(batch_size, len(self.schema.set_fields)),
            "enum_weights": torch.ones(batch_size, len(self.schema.enum_fields)),
            "set_weights": torch.ones(batch_size, len(self.schema.set_fields)),
        }
        loss, details = visual_state_loss(output, batch, self.schema)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(details["valid_enum_weight"], 0.0)
        self.assertEqual(details["valid_set_weight"], 0.0)

    def test_resnet_multitask_output_shapes(self):
        model = VisualStateMultiTaskModel("resnet18", self.schema, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output["class_logits"].shape), (2, 5))
        self.assertEqual(len(output["enum_logits"]), 21)
        self.assertEqual(len(output["set_logits"]), 2)

    def test_two_stage_cua_respects_stage_budgets(self):
        schema = self.schema
        base = []
        candidates = []
        colors = ["black", "white", "brown", "gray"]
        for index, color in enumerate(colors):
            row = self.feature_row()
            row.update({"image_id": f"candidate_{index}", "coat_primary_color": color})
            candidates.append(row)
        priorities = {(feature_id, value): 0.1 for feature_id, value in schema.coverage_states()}
        priorities[("coat_primary_color", "gray")] = 0.9
        selected, audit = select_two_stage_cua(
            candidates, Counter(), schema, gap_budget=2, utility_budget=1,
            cap=2, state_utility=priorities,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual([row["selection_stage"] for row in audit], ["gap", "gap", "utility_novelty"])


if __name__ == "__main__":
    unittest.main()
