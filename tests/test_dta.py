import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from qe_quality.dta.io import unique_json_object
from qe_quality.dta.clients import post_json
from qe_quality.dta.geometry import select_foreground_mask
from qe_quality.dta.reports import deterministic_review_ids, kappa, percentile, report_all
from qe_quality.dta.schema import geometry_from_mask, pad_box, validate_semantic


def semantic(**changes):
    value = {
        "image_id": "sample_1", "class_id": 9, "class_name": "animal",
        "class_confidence": .9, "outline_visibility": "clear",
        "occlusion_level": "none", "target_bbox_hint": [.1, .2, .8, .9],
        "fine_name": "fish", "subject_role": "foreground_displayed_object",
        "selection_confidence": .9,
        "subject_selection_evidence": "鱼位于人物之前并被双手展示。",
        "evidence": "可见一只轮廓完整的动物主体。",
    }
    value.update(changes)
    return value


class StrictJsonTests(unittest.TestCase):
    def test_requires_one_object_and_rejects_duplicates(self):
        self.assertEqual(unique_json_object('{"a":1}'), {"a": 1})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            unique_json_object('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "one_json"):
            unique_json_object('{"a":1} trailing')
        with self.assertRaises(ValueError):
            unique_json_object('[1]')

    def test_local_post_bypasses_environment_proxy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            status, text, _, _ = post_json(
                f"http://127.0.0.1:{server.server_port}/", {}, {}, 2)
            self.assertEqual((status, text), (200, "{}"))
        finally:
            thread.join(2)
            server.server_close()


class SchemaTests(unittest.TestCase):
    def test_validates_and_applies_low_confidence_fallback(self):
        result = validate_semantic(semantic(class_confidence=.5), "sample_1")
        self.assertEqual((result["raw_class_id"], result["class_id"], result["class_name"]),
                         (9, 15, "other"))
        with self.assertRaisesRegex(ValueError, "name"):
            validate_semantic(semantic(class_name="vehicle"), "sample_1")
        with self.assertRaisesRegex(ValueError, "bbox"):
            validate_semantic(semantic(target_bbox_hint=[.8, .2, .1, .9]), "sample_1")
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_semantic({**semantic(), "extra": 1}, "sample_1")
        with self.assertRaisesRegex(ValueError, "subject_role"):
            validate_semantic(semantic(subject_role="largest_person"), "sample_1")
        with self.assertRaisesRegex(ValueError, "selection_confidence"):
            validate_semantic(semantic(selection_confidence=1.1), "sample_1")

    def test_qwen_native_bbox_is_normalized_and_preserved(self):
        result = validate_semantic(semantic(target_bbox_hint=[100, 200, 800, 900]), "sample_1")
        self.assertEqual(result["target_bbox_hint"], [.1, .2, .8, .9])
        self.assertEqual(result["raw_target_bbox_hint"], [100., 200., 800., 900.])
        self.assertEqual(result["target_bbox_scale"], 1000)

    def test_mask_geometry_uses_half_open_bbox_and_dual_axis(self):
        mask = np.zeros((6, 9), dtype=bool)
        mask[0:2, 7:9] = True
        result = geometry_from_mask(mask)
        self.assertEqual(result["bbox"], [7, 0, 9, 2])
        self.assertEqual(result["horizontal_position"], "right")
        self.assertEqual(result["vertical_position"], "top")
        self.assertEqual(result["legacy_position"], "top")
        self.assertAlmostEqual(result["area_ratio"], 4 / 54)
        with self.assertRaisesRegex(ValueError, "empty"):
            geometry_from_mask(np.zeros((2, 2), dtype=bool))

    def test_padding_clips_to_image(self):
        self.assertEqual(pad_box([0, 0, 1, 1], 100, 50), [0., 0., 100., 50.])

    def test_background_like_sam_mask_is_complemented_inside_large_box(self):
        masks = np.zeros((2, 6, 6), dtype=bool)
        masks[0, 0, 1:5] = True
        masks[0, -1, 1:5] = True
        masks[0, 1:-1, 0] = True
        masks[0, 1:-1, -1] = True
        masks[0, 2:4, 2:4] = False
        masks[1, 2:4, 2:4] = True
        best, mask, complemented = select_foreground_mask(masks, np.array([.9, .8]), [0, 0, 6, 6])
        self.assertEqual(best, 0)
        self.assertTrue(complemented)
        self.assertTrue(mask[2:4, 2:4].all())
        self.assertFalse(mask[0, 1])

    def test_small_box_never_uses_background_complement(self):
        masks = np.ones((1, 6, 6), dtype=bool)
        _, mask, complemented = select_foreground_mask(masks, np.array([.9]), [2, 2, 4, 4])
        self.assertFalse(complemented)
        self.assertTrue(mask.all())


class ReportMathTests(unittest.TestCase):
    def test_kappa_and_percentile_edges(self):
        self.assertEqual(kappa([1, 1], [1, 1]), 1)
        self.assertEqual(kappa([1, 2], [1, 2]), 1)
        self.assertIsNone(kappa([], []))
        self.assertEqual(percentile([1, 2, 3], .5), 2)

    def test_review_selection_is_two_per_class(self):
        rows = [{"image_id": f"c{c}_{i}", "source_class_id": str(c)} for c in range(3) for i in range(5)]
        first = deterministic_review_ids(rows, 17)
        self.assertEqual(len(first), 6)
        self.assertEqual(first, deterministic_review_ids(list(reversed(rows)), 17))

    def test_report_writes_empty_safe_disagreement_and_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [{"image_id": "x", "sha256": "abc", "source_class_id": "s", "source_class_name": "source"}]
            semantic_value = validate_semantic(semantic(image_id="x", evidence="依据" * 100), "x")
            semantic_value.update({"elapsed_seconds": 1.0, "backend": "test", "attempts": 1})
            geometry = {"bbox": [0, 0, 1, 1], "area_ratio": 1., "horizontal_position": "center",
                        "vertical_position": "middle", "legacy_position": "center", "sam_score": .9,
                        "background_complement_applied": False, "elapsed_seconds": .2}
            report = report_all(output, rows, {"x": semantic_value}, {"x": semantic_value}, {"x": geometry},
                                [], {"review_seed": 1}, "protocol")
            self.assertEqual(report["decision"], "NEEDS_REVIEW")
            self.assertTrue((output / "features.parquet").is_file())
            self.assertTrue((output / "disagreements.csv").is_file())


if __name__ == "__main__":
    unittest.main()
