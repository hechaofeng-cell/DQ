import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from qe_quality.auto_feature.pipeline import run_pipeline
from qe_quality.auto_feature.schema_validator import freeze_schema


VALID_SCHEMA = {
    "schema_version": "proposal",
    "category": "person",
    "features": [
        {"feature_id": f"feature_{index}", "group": "observable", "value_type": "enum", "possible_values": ["visible", "unknown"], "allow_unknown": True, "evidence_required": True}
        for index in range(8)
    ],
    "rules": [],
}


class AutoPipelineTests(unittest.TestCase):
    def test_pipeline_isolated_category_and_retries_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            marked = root / "marked.jpg"
            crop = root / "crop.jpg"
            for path in (image, marked, crop):
                Image.new("RGB", (20, 20), "white").save(path)
            manifest = root / "target_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_id", "target_instance_id", "image_path", "bbox", "marked_image_path", "crop_image_path", "target_class"])
                writer.writeheader()
                writer.writerow({"image_id": "x", "target_instance_id": "target_1", "image_path": str(image), "bbox": "[1,1,19,19]", "marked_image_path": str(marked), "crop_image_path": str(crop), "target_class": "person"})
            policy = root / "policy.json"
            policy.write_text(json.dumps({"min_features": 8, "max_features": 30, "schema_sample_size": 1, "max_schema_attempts": 3, "max_category_rounds": 2, "min_category_consensus": .8, "min_category_confidence": .6}), encoding="utf-8")
            output = root / "out"
            frozen = freeze_schema(VALID_SCHEMA, json.loads(policy.read_text(encoding="utf-8")))
            calls = []
            schema_calls = [
                {"schema_version": "proposal", "category": "person", "features": [{"feature_id": "bad", "group": "x", "value_type": "enum", "possible_values": ["yes"], "allow_unknown": True, "evidence_required": True}], "rules": []},
                VALID_SCHEMA,
            ]

            def fake_call(config, prompt, images, output_path, response_schema=None):
                calls.append((prompt, images, response_schema))
                if "独立的视觉类别判断器" in prompt:
                    return {"image_id": "x", "target_instance_id": "target_1", "predicted_category": "person", "fine_name": "person", "confidence": .95, "evidence": "human body is visible"}
                if "视觉数据协议设计器" in prompt:
                    return schema_calls.pop(0)
                return {"schema_version": frozen["schema_version"], "image_id": "x", "target_instance_id": "target_1", "features": {f"feature_{index}": "visible" for index in range(8)}, "evidence": {f"feature_{index}": "directly visible" for index in range(8)}}

            with patch("qe_quality.auto_feature.pipeline.call_vlm", side_effect=fake_call):
                report = run_pipeline(manifest, output, policy, Path("prompts/category_discovery_v1.txt"), Path("prompts/schema_generator_v1.txt"), Path("prompts/feature_annotation_auto_v1.txt"), {"url": "mock", "model": "mock", "max_retries": 0}, resume=False)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["feature_ok"], 1)
            self.assertEqual(len([call for call in calls if "视觉数据协议设计器" in call[0]]), 2)
            self.assertNotIn("target_class", calls[0][0])
            self.assertEqual(json.loads((output / "schema_frozen.json").read_text())["schema_fingerprint"], frozen["schema_fingerprint"])
            result = (output / "results.csv").read_text(encoding="utf-8")
            self.assertIn("schema_fingerprint", result)


if __name__ == "__main__":
    unittest.main()
