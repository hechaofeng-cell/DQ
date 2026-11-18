import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

SPEC = importlib.util.spec_from_file_location("fresh_audit", Path(__file__).resolve().parents[1] / "scripts/audit_fresh_confirmation_20261112.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FreshAuditTests(unittest.TestCase):
    def test_ID_crossing_chunk_boundary_is_not_missed(self):
        with tempfile.TemporaryDirectory() as d, patch.object(audit, "ROOT", Path(d)):
            p = Path(d) / "history.json"
            p.write_bytes(b" " * (1024 * 1024 - 5) + b'"000000123456"')
            _, ids, _, _, record = audit.file_evidence(p)
            self.assertIn("000000123456", ids)
            self.assertEqual(record["sha256"], audit.sha(p))

    def test_raw_availability_catalogue_does_not_poison_usage_registry(self):
        with tempfile.TemporaryDirectory() as d, patch.object(audit, "ROOT", Path(d)):
            p = Path(d) / "catalogue.json"
            p.write_text('{"image_id":"000000123456"}')
            with patch.object(audit, "CATALOGUES", {p}):
                _, ids, _, _, record = audit.file_evidence(p)
            self.assertEqual(ids, set())
            self.assertEqual(record["kind"], "raw_source_catalogue_not_usage")

    def test_prediction_cache_IDs_read_without_unpickling_objects(self):
        with tempfile.TemporaryDirectory() as d, patch.object(audit, "ROOT", Path(d)):
            p = Path(d) / "predictions.npz"
            np.savez(p, sample_ids=np.array(["cocoB-train2017-000000123456-dog", "obj365-train-2106735-sheep"]),
                     logits=np.zeros((2, 6)), opaque=np.array([{"not": "trusted"}], dtype=object))
            _, ids, obj, _, record = audit.file_evidence(p)
            self.assertIn("000000123456", ids)
            self.assertIn("2106735", obj)
            self.assertIn("object_array_not_unpickled:opaque.npy", record["issues"])

    def test_source_image_exclusion_removes_every_class_target(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.zip"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("coco/labels/train2017/000000123456.txt", "16 0.5 0.5 0.2 0.3\n18 0.5 0.5 0.2 0.3\n")
                z.writestr("coco/labels/train2017/000000654321.txt", "18 0.5 0.5 0.3 0.4\n18 0.5 0.5 0.1 0.2\n")
            counts, rows = audit.enumerate_coco(p, {"000000123456"})
            self.assertEqual(counts["dog"], 0)
            self.assertEqual(counts["sheep"], 1)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["bbox_area_ratio"], .12)


if __name__ == "__main__":
    unittest.main()
