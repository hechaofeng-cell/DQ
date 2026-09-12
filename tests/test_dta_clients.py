import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qe_quality.dta.clients import ollama_annotation


def response(class_id, class_name):
    content = {
        "image_id": "x", "class_id": class_id, "class_name": class_name,
        "class_confidence": .9, "fine_name": "box",
        "subject_role": "foreground_dominant_object", "selection_confidence": .9,
        "subject_selection_evidence": "包装盒位于前景并占据主要视觉区域。",
        "outline_visibility": "clear", "occlusion_level": "none",
        "target_bbox_hint": [10, 10, 900, 900], "evidence": "完整可见的包装盒主体。",
    }
    return json.dumps({"message": {"content": json.dumps(content, ensure_ascii=False)}})


class OllamaRetryTests(unittest.TestCase):
    @patch("qe_quality.dta.clients.time.sleep")
    @patch("qe_quality.dta.clients.post_json")
    def test_retries_schema_mismatch_with_correction_turn(self, post, _sleep):
        post.side_effect = [
            (200, response(8, "outdoor_item"), 1.0, {}),
            (200, response(6, "indoor_item"), 1.1, {}),
        ]
        config = {"url": "http://localhost/test", "model": "model", "num_ctx": 10,
                  "temperature": 0, "num_predict": 10,
                  "timeout_seconds": 5, "max_retries": 2, "retry_wait_seconds": 0}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "x.jpg"
            image.write_bytes(b"image")
            result = ollama_annotation(config, "prompt", {"image_id": "x", "path": str(image)},
                                       Path(directory) / "raw.json")
        self.assertEqual((result["class_id"], result["class_name"], result["attempts"]),
                         (6, "indoor_item", 2))
        retry_body = post.call_args_list[1].args[1]
        self.assertIn("class_id_name_mismatch", retry_body["messages"][-1]["content"])
