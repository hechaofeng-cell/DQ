"""Check experimental safeguards, not just the exact default implementation."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("five_dimension_pilot", ROOT / "scripts/run_category_five_dimension_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ScoringSafeguards(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(pilot.DEFAULT_CONFIG.read_text())
        self.features = dict(body_visibility="full", head_visibility="clear", leg_visibility="all_expected",
                             occlusion_level="none", truncation_level="none", body_orientation="front",
                             head_orientation="front")

    def test_unknown_keeps_budget_and_uncertainty(self):
        aggregate = pilot.aggregate_components([(1., .5), (None, .5)])
        self.assertEqual(aggregate["lower"], .5)
        self.assertEqual(aggregate["upper"], 1)
        self.assertEqual(aggregate["midpoint"], .75)
        self.assertEqual(aggregate["known_weight"], .5)

    def test_unknown_is_not_observed_absence_or_zero(self):
        missing = dict(self.features, head_visibility="unknown")
        absent = dict(self.features, head_visibility="not_visible")
        a = pilot.score_sample(.2, missing, "dog", "equal_parts", self.config)
        b = pilot.score_sample(.2, absent, "dog", "equal_parts", self.config)
        self.assertFalse(a["fully_known"])
        self.assertTrue(b["fully_known"])
        self.assertLess(a["score_midpoint"], b["score_midpoint"])
        self.assertGreater(a["score_upper"], a["score_lower"])

    def test_best_worst_and_scale_boundaries(self):
        self.assertEqual(pilot.score_sample(.2, self.features, "dog", "equal_parts", self.config)["score_midpoint"], 0)
        worst = dict(self.features, head_visibility="not_visible", leg_visibility="none", occlusion_level="heavy",
                     truncation_level="heavy", body_orientation="overhead_or_unusual", head_orientation="up")
        self.assertAlmostEqual(pilot.score_sample(.001, worst, "dog", "equal_parts", self.config)["score_midpoint"], 100)
        self.assertEqual(pilot.scale_risk(.01, self.config), 2/3)
        self.assertEqual(pilot.scale_risk(.03, self.config), 1/3)
        self.assertEqual(pilot.scale_risk(.10, self.config), 0)
        with self.assertRaises(ValueError):
            pilot.scale_risk(float("nan"), self.config)

    def test_nominal_properties_do_not_create_risk(self):
        a = pilot.score_sample(.02, dict(self.features, coat_primary_color="black"), "dog", "equal_parts", self.config)
        b = pilot.score_sample(.02, dict(self.features, coat_primary_color="white"), "dog", "equal_parts", self.config)
        self.assertEqual(a, b)

    def test_new_unknown_enum_cannot_silently_be_zero(self):
        with self.assertRaises(ValueError):
            pilot.score_sample(.1, dict(self.features, head_visibility="novel_state"), "dog", "equal_parts", self.config)

    def test_car_does_not_apply_leg_or_head_fields(self):
        features = {k: v for k, v in self.features.items() if k not in {"head_visibility", "leg_visibility", "head_orientation"}}
        self.assertEqual(pilot.score_sample(.02, features, "car", "equal_global", self.config),
                         pilot.score_sample(.02, features, "car", "equal_parts", self.config))

    def test_original_score_is_preserved_after_rescaling(self):
        features = dict(self.features, body_visibility="partial", occlusion_level="light",
                        truncation_level="light", body_orientation="back")
        expected_original = 2 + 2 + 1 + .5 + 1
        score = pilot.score_sample(.02, features, "dog", "legacy_global", self.config)
        self.assertAlmostEqual(score["score_midpoint"], expected_original / 13 * 100)

    def test_annotation_cannot_use_another_target(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            pilot.validate_label(dict(image_id="other", target_instance_id="target_1", features={}),
                                 dict(sample_id="target", target_instance_id="target_1"), {})

    def test_native_alias_requires_same_class_views_and_box(self):
        candidate = dict(sample_id="sample", class_name="dog", target_instance_id="official_42",
                         image_path="image.jpg", marked_image_path="marked.jpg", feature_crop_path="crop.jpg",
                         bbox_norm_xyxy="[0.1,0.2,0.8,0.9]")
        native = dict(image_id="sample", target_class="dog", target_instance_id="target_1",
                      image_path="image.jpg", marked_image_path="marked.jpg", crop_image_path="crop.jpg",
                      bbox="[10,20,80,90]")
        bound = pilot.validate_input_binding(candidate, native, (100, 100))
        self.assertEqual(bound["target_instance_id"], "target_1")
        with self.assertRaisesRegex(ValueError, "box differs"):
            pilot.validate_input_binding(candidate, dict(native, bbox="[0,0,100,100]"), (100, 100))
        with self.assertRaisesRegex(ValueError, "class/ID mismatch"):
            pilot.validate_input_binding(candidate, dict(native, target_class="cat"), (100, 100))


if __name__ == "__main__":
    unittest.main()
