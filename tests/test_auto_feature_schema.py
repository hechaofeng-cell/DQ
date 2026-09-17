import unittest

from qe_quality.auto_feature.category import aggregate_category, validate_category_result
from qe_quality.auto_feature.schema_validator import (
    feature_output_schema,
    fingerprint_schema,
    freeze_schema,
    validate_feature_result,
    validate_schema,
)


POLICY = {"min_features": 2, "max_features": 4, "value_types": ["enum", "set", "number", "integer"]}


def schema():
    return {
        "schema_version": "proposal",
        "category": "person",
        "features": [
            {"feature_id": "body_orientation", "group": "orientation", "value_type": "enum", "possible_values": ["front", "unknown"], "allow_unknown": True, "evidence_required": True},
            {"feature_id": "visible_parts", "group": "visibility", "value_type": "set", "possible_values": ["head", "torso", "unknown"], "exclusive_values": ["unknown"], "allow_unknown": True, "evidence_required": True},
        ],
        "rules": [],
    }


class AutoSchemaTests(unittest.TestCase):
    def test_rejects_sensitive_and_missing_unknown(self):
        value = schema()
        value["features"][0]["feature_id"] = "age_group"
        with self.assertRaisesRegex(ValueError, "sensitive"):
            validate_schema(value, POLICY)
        value = schema()
        value["features"][0]["possible_values"] = ["front", "back"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_schema(value, POLICY)

    def test_freeze_has_stable_fingerprint_and_dynamic_output_contract(self):
        frozen = freeze_schema(schema(), POLICY)
        self.assertEqual(frozen["schema_fingerprint"], fingerprint_schema(frozen))
        output = feature_output_schema(frozen)
        self.assertEqual(output["properties"]["features"]["required"], ["body_orientation", "visible_parts"])

    def test_feature_result_rejects_extra_field_and_accepts_set(self):
        frozen = freeze_schema(schema(), POLICY)
        row = {"image_id": "x", "target_instance_id": "target_1"}
        result = {"schema_version": frozen["schema_version"], "image_id": "x", "target_instance_id": "target_1", "features": {"body_orientation": "front", "visible_parts": ["head"]}, "evidence": {"body_orientation": "facing the camera", "visible_parts": "head is visible"}}
        self.assertEqual(validate_feature_result(result, row, frozen), result)
        result["features"]["extra"] = "unknown"
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_feature_result(result, row, frozen)

    def test_category_aggregation_rejects_mixed_batch(self):
        rows = [{"image_id": str(i), "target_instance_id": "target_1"} for i in range(2)]
        values = [validate_category_result({"image_id": str(i), "target_instance_id": "target_1", "predicted_category": category, "fine_name": category, "confidence": .9, "evidence": "visible subject"}, rows[i]) for i, category in enumerate(("person", "animal"))]
        with self.assertRaisesRegex(ValueError, "mixed"):
            aggregate_category(values, min_consensus=.8)

    def test_rules_must_reference_existing_features(self):
        value = schema()
        value["rules"] = [{"if_feature": "missing_feature", "then": "unknown"}]
        with self.assertRaisesRegex(ValueError, "rule_references"):
            validate_schema(value, POLICY)


if __name__ == "__main__":
    unittest.main()
