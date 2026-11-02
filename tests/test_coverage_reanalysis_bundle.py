import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import replay_coverage_diagnostic_bundle as replay


class ReanalysisVerificationTests(unittest.TestCase):
    def test_changed_input_is_rejected_by_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.json"
            data.write_text('{"support":20}')
            record = {"path": data.name,
                      "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
            (root / "manifest.sha256.json").write_text(json.dumps([record]))
            self.assertEqual(replay.verify_manifest(root), 1)
            data.write_text('{"support":21}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                replay.verify_manifest(root)

    def test_csv_roundoff_passes_but_changed_effect_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            expected, actual = (Path(directory) / name for name in ("expected.csv", "actual.csv"))
            expected.write_text("method,effect\ngreedy,0.25\n")
            actual.write_text("method,effect\ngreedy,0.250000000001\n")
            self.assertEqual(replay.compare_csv(actual, expected)["rows"], 1)
            actual.write_text("method,effect\ngreedy,0.26\n")
            with self.assertRaisesRegex(ValueError, "value mismatch"):
                replay.compare_csv(actual, expected)

    def test_csv_reordered_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            expected, actual = (Path(directory) / name for name in ("expected.csv", "actual.csv"))
            expected.write_text("sample_id,value\ntarget_a,1\ntarget_b,1\n")
            actual.write_text("sample_id,value\ntarget_b,1\ntarget_a,1\n")
            with self.assertRaisesRegex(ValueError, "value mismatch"):
                replay.compare_csv(actual, expected)

    def test_changed_selection_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            expected, actual = (Path(directory) / name for name in ("expected.npz", "actual.npz"))
            np.savez_compressed(expected, selected=np.array([1, 3, 5]))
            np.savez_compressed(actual, selected=np.array([1, 3, 5]))
            self.assertEqual(replay.compare_npz(actual, expected)["arrays"], 1)
            np.savez_compressed(actual, selected=np.array([1, 3, 6]))
            with self.assertRaisesRegex(ValueError, "array mismatch"):
                replay.compare_npz(actual, expected)


if __name__ == "__main__":
    unittest.main()
