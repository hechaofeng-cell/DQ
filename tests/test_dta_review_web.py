import tempfile
import unittest
from pathlib import Path

from qe_quality.dta.io import atomic_csv, atomic_json, read_csv
from qe_quality.dta.reports import REVIEW_FIELDS
from qe_quality.dta.review_web import ReviewStore


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        image = self.output / "source.jpg"
        image.write_bytes(b"image")
        atomic_csv(self.output / "manifest.csv", [{
            "image_id": "x", "path": str(image), "sha256": "abc",
            "source_class_id": "source", "source_class_name": "source-name",
        }], ["image_id", "path", "sha256", "source_class_id", "source_class_name"])
        atomic_csv(self.output / "human_review.csv", [{
            "image_id": "x", "sha256": "abc", "selection_reason": "semantic_disagreement",
            **{field: "" for field in REVIEW_FIELDS[3:]},
        }], REVIEW_FIELDS)
        local = {"class_id": 1, "class_name": "person", "subject_role": "action_agent",
                 "outline_visibility": "clear", "occlusion_level": "none"}
        online = {**local, "class_id": 12, "class_name": "tool", "subject_role": "foreground_operated_object"}
        for backend, result in (("local", local), ("online", online)):
            atomic_json(self.output / "parsed" / backend / "x.json", {"result": result})

    def tearDown(self):
        self.temp.cleanup()

    def test_freezes_only_class_disagreements_and_saves_complete_review(self):
        store = ReviewStore(self.output)
        self.assertEqual(store.queue, ["x"])
        self.assertTrue((self.output / "class_review_queue.json").is_file())
        review = store.save("x", {
            "human_class_id": "12", "human_subject_role": "foreground_operated_object",
            "human_outline_visibility": "clear", "human_occlusion_level": "none",
            "sam_target_match": "yes", "sam_mask_acceptable": "yes",
            "reviewer": "tester", "notes": "online",
        })
        self.assertEqual(review["human_class_id"], "12")
        self.assertTrue(review["reviewed_at"])
        self.assertEqual(read_csv(self.output / "human_review.csv")[0]["reviewer"], "tester")

    def test_rejects_incomplete_review(self):
        store = ReviewStore(self.output)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            store.save("x", {"human_class_id": "12"})

