import csv
import gzip
import io
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

from qe_quality.difficulty.inventory import prepare
from qe_quality.difficulty.io import digest, read_csv, read_json, verify_run, write_csv, write_json
from qe_quality.difficulty.metrics import auc, discrimination, panel_stability, spearman, wilson
from qe_quality.difficulty.pipeline import apply_reviews, apply_splits, evaluate, freeze
from qe_quality.difficulty.scoring import calibration_metrics, calibrate, fit_temperature, probabilities, score_a, score_b


class FormulaTests(unittest.TestCase):
    def test_a_uses_fixed_panel_and_never_shrinks_denominator(self):
        self.assertEqual(score_a({"a": "cat", "b": "dog", "v": "cat"}, "cat", ["a", "b"]), .5)
        self.assertIsNone(score_a({"a": "cat"}, "cat", ["a", "b"]))
        with self.assertRaises(ValueError):
            score_a({"a": "cat"}, "cat", ["a", "a"])

    def test_b_uses_true_class_not_max_and_per_model_softmax(self):
        logits = {"a": [0., 2.], "b": [0., 4.]}
        expected = 1 - (1 / (1 + math.exp(2)) + 1 / (1 + math.exp(4))) / 2
        self.assertAlmostEqual(score_b(logits, 0, ["a", "b"], {"a": 1, "b": 1}), expected)
        self.assertGreater(expected, .9)
        self.assertIsNone(score_b(logits, 0, ["a", "b"], {"a": 1}))
        self.assertEqual(probabilities([10000, 10000]), [.5, .5])
        with self.assertRaises(ValueError):
            probabilities([float("nan"), 0])

    def test_temperature_recovers_known_frequency(self):
        examples = [([6., 0.], 0)] * 80 + [([6., 0.], 1)] * 20
        temp = fit_temperature(examples, [.05, 20])
        self.assertAlmostEqual(probabilities([6, 0], temp)[0], .8, places=5)
        self.assertLess(calibration_metrics(examples, temp)["nll"], calibration_metrics(examples)["nll"])

    def test_ties_and_constant_rank_are_explicit(self):
        self.assertIsNone(spearman([0, 0], [0, 1]))
        self.assertEqual(spearman([0, 1, 1], [0, 1, 1]), 1)
        self.assertEqual(auc([0, 0, 1, 1], [False, True, False, True]), .5)
        self.assertIsNone(auc([0, 1], [False, False]))
        self.assertEqual(discrimination([0, 0, 0, 1])["largest_tie_count"], 3)
        self.assertGreater(wilson(0, 10)[1], .2)


class ReviewAndIsolationTests(unittest.TestCase):
    def rows(self):
        return [{"original_id": str(i), "source_group_id": "shared", "sha256": "abc",
                 "inherited_class_id": "a", "reference_class_id": "a", "review_status": "pending",
                 "split": "unassigned"} for i in range(2)]

    def test_group_leakage_rejected(self):
        assignments = [{"original_id": str(i), "source_group_id": "shared", "split": p}
                       for i, p in enumerate(["calibration_fit", "validation"])]
        with self.assertRaisesRegex(ValueError, "cross use partitions"):
            apply_splits(self.rows(), assignments)

    def test_correction_requires_bound_bytes_and_audit_evidence(self):
        review = {"original_id": "0", "sha256": "abc", "review_status": "corrected",
                  "reference_class_id": "b", "reviewer": "reviewer", "reviewer_kind": "human",
                  "reviewed_at": "2026-09-07", "evidence": "Inspected original; label is b"}
        result = apply_reviews(self.rows(), [review], ["a", "b"])
        self.assertEqual(result[0]["reference_class_id"], "b")
        with self.assertRaisesRegex(ValueError, "bytes"):
            apply_reviews(self.rows(), [dict(review, sha256="other")], ["a", "b"])
        with self.assertRaises(ValueError):
            apply_reviews(self.rows(), [dict(review, evidence="")], ["a", "b"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = self.root / "history"
        self.source = self.history / "gpu"
        self.audit = self.history / "audit"
        self.source.mkdir(parents=True)
        self.audit.mkdir()
        self.archive = self.root / "images.zip"
        manifest = []
        with zipfile.ZipFile(self.archive, "w") as archive:
            for i in range(20):
                cls = "a" if i % 2 == 0 else "b"
                image_id = f"{cls}_{i}"
                buffer = io.BytesIO()
                Image.new("RGB", (8, 8), (i * 10, 10, 20)).save(buffer, format="PNG")
                for member, role in ((f"train/{cls}/{image_id}.png", "original"),
                                     (f"base/{image_id}.png", "original_copy")):
                    archive.writestr(member, buffer.getvalue())
                    manifest.append({"member": member, "image_id": image_id, "group": str(Path(member).parent),
                                     "class_id": cls, "role": role, "source_status": "unknown"})
        write_csv(self.source / "manifest.csv", manifest)
        write_json(self.source / "classes.json", [{"class_id": c, "class_name": c} for c in ["a", "b"]])
        write_json(self.source / "protocol.json", {"class_order": ["a", "b"],
                   "archive": str(self.archive), "archive_bytes": self.archive.stat().st_size})
        for model in ("m1", "m2", "v"):
            write_json(self.source / f"{model}_run.json", {"model": model, "weights": "fixture", "preprocess": "fixture"})
            with gzip.open(self.source / f"{model}.jsonl.gz", "wt") as stream:
                for index, row in enumerate(manifest):
                    i = int(row["image_id"].split("_")[1])
                    predicted = ("b" if row["class_id"] == "a" else "a") if i % 4 == 0 else row["class_id"]
                    logits = [6., 0.] if predicted == "a" else [0., 6.]
                    stream.write(json.dumps({"index": index, "member": row["member"], "prediction": predicted,
                                             "correct": predicted == row["class_id"], "logits_10": logits}) + "\n")
        write_csv(self.audit / "old_pilot_audit.csv", [], ["image_id", "official_training_bbox_id_match"])
        write_csv(self.audit / "validation_candidates.csv", [], ["image_id", "sha256", "true_class_id"])
        self.config = self.root / "config.json"
        write_json(self.config, {"protocol_version": "fixture-v1", "prediction_dir": str(self.source),
            "source_audit_dir": str(self.audit), "archive": str(self.archive), "baseline_group": "base",
            "expected_count": 20, "scoring_models": ["m1", "m2"], "validation_models": ["v"],
            "random_review_per_class": 1, "review_seed": 1, "diagnostic_band_edges": [0, .5, 1],
            "calibration": {"method": "scalar_temperature_nll", "min_per_part": 4, "min_per_class": 1,
                            "min_errors_per_model": 1, "temperature_bounds": [.05, 20]}})

    def prepare(self):
        return prepare(self.config, self.root / "prepared")

    def audit_inputs(self, prepared):
        rows = read_csv(prepared / "samples.csv")
        reviews = []
        assignments = []
        for row in rows:
            i = int(row["original_id"].split("_")[1])
            reviews.append({"original_id": row["original_id"], "sha256": row["sha256"],
                "review_status": "confirmed", "reference_class_id": row["reference_class_id"],
                "reviewer": "synthetic_fixture", "reviewer_kind": "human", "reviewed_at": "2026-09-07",
                "evidence": "Synthetic test only"})
            part = ["calibration_fit", "calibration_check", "development", "validation", "unassigned"][i // 4]
            assignments.append({"original_id": row["original_id"], "source_group_id": row["source_group_id"], "split": part})
        rp, sp = self.root / "reviews.csv", self.root / "splits.csv"
        write_csv(rp, reviews)
        write_csv(sp, assignments)
        return rp, sp

    def test_real_format_pipeline_pending_labels_and_preservation(self):
        before = digest(self.source / "m1.jsonl.gz")
        prepared = self.prepare()
        output = evaluate(prepared, self.root / "evaluated")
        report = read_json(output / "report.json")
        self.assertEqual(report["score_a_count"], 0)
        self.assertEqual(report["score_b_count"], 0)
        self.assertEqual(report["inherited_diagnostic_count"], 20)
        self.assertEqual(digest(self.source / "m1.jsonl.gz"), before)
        self.assertTrue(all(read_json(output / "checks.json").values()))
        verify_run(output)
        with self.assertRaises(FileExistsError):
            evaluate(prepared, output)
        with self.assertRaises(ValueError):
            evaluate(prepared, self.source / "nested_output")
        with self.assertRaises(ValueError):
            freeze(output, self.root / "invalid_freeze")

    def test_calibration_development_freeze_and_validation(self):
        prepared = self.prepare()
        reviews, splits = self.audit_inputs(prepared)
        development = evaluate(prepared, self.root / "dev", reviews, splits)
        report = read_json(development / "report.json")
        self.assertEqual(report["calibration"]["status"], "calibration_checked_exploratory")
        scores = read_csv(development / "scores.csv")
        self.assertTrue(all(not r["score_a"] and not r["score_b"] for r in scores if r["split"] == "validation"))
        self.assertTrue(all(r["score_b"] for r in scores if r["split"] == "development"))
        frozen = freeze(development, self.root / "frozen")
        result = evaluate(prepared, self.root / "validation", frozen=frozen / "frozen.json")
        heldout = read_json(result / "report.json")
        self.assertEqual(set(heldout["analyses"]), {"validation"})
        self.assertEqual(heldout["score_a_count"], 4)
        self.assertEqual(heldout["score_b_count"], 4)
        with self.assertRaises(ValueError):
            evaluate(prepared, self.root / "bad", reviews_path=reviews, frozen=frozen / "frozen.json")

    def test_missing_model_does_not_change_denominator(self):
        (self.source / "m2.jsonl.gz").unlink()
        prepared = self.prepare()
        output = evaluate(prepared, self.root / "run")
        report = read_json(output / "report.json")
        self.assertEqual(report["inherited_diagnostic_count"], 0)
        self.assertEqual(report["score_a_count"], 0)
        self.assertIn("missing_prediction_file", {i["code"] for i in read_csv(prepared / "issues.csv")})

    def test_uncertain_label_has_no_diagnostic_or_score(self):
        prepared = self.prepare()
        reviews, splits = self.audit_inputs(prepared)
        values = read_csv(reviews)
        values[0]["review_status"] = "uncertain"
        different = self.root / "uncertain.csv"
        write_csv(different, values)
        output = evaluate(prepared, self.root / "run", different)
        row = next(r for r in read_csv(output / "scores.csv") if r["original_id"] == values[0]["original_id"])
        self.assertEqual([row[k] for k in ("score_a", "score_b", "diagnostic_a_inherited")], ["", "", ""])

    def test_partial_review_overlay_keeps_uniform_csv_schema(self):
        prepared = self.prepare()
        reviews, _ = self.audit_inputs(prepared)
        values = read_csv(reviews)
        partial = self.root / "partial.csv"
        write_csv(partial, [values[1]])
        output = evaluate(prepared, self.root / "run", partial)
        result = read_json(output / "report.json")
        self.assertEqual(result["score_a_count"], 1)
        self.assertEqual(result["inherited_diagnostic_count"], 19)
        self.assertEqual(len(read_csv(output / "scores.csv")), 20)

    def test_holdout_model_missing_does_not_block_a(self):
        (self.source / "v.jsonl.gz").unlink()
        prepared = self.prepare()
        reviews, _ = self.audit_inputs(prepared)
        output = evaluate(prepared, self.root / "run", reviews)
        report = read_json(output / "report.json")
        self.assertEqual(report["score_a_count"], 20)
        check = report["analyses"]["unassigned"]["score_a"]["independent_models"]["v"]
        self.assertEqual(check["missing_validator_predictions"], 20)
        self.assertEqual(check["n_groups"], 0)

    def test_historical_change_detected(self):
        prepared = self.prepare()
        with (self.source / "protocol.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            evaluate(prepared, self.root / "run")

    def test_roles_cannot_overlap(self):
        config = read_json(self.config)
        config["validation_models"] = ["m1"]
        bad = self.root / "bad_config.json"
        write_json(bad, config)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare(bad, self.root / "bad")

    def rewrite_prediction(self, mutate):
        path = self.source / "m1.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            records = [json.loads(line) for line in stream]
        mutate(records)
        with gzip.open(path, "wt") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def test_invalid_logits_and_argmax_are_listed_and_excluded(self):
        def mutate(records):
            records[1]["logits_10"] = [float("nan"), 1]
            records[3]["prediction"] = "a"
            records[5]["logits_10"] = [1]
        self.rewrite_prediction(mutate)
        prepared = self.prepare()
        issues = read_csv(prepared / "issues.csv")
        self.assertTrue({"invalid_logits", "argmax_mismatch"} <= {i["code"] for i in issues})
        output = evaluate(prepared, self.root / "run")
        self.assertEqual(read_json(output / "report.json")["inherited_diagnostic_count"], 17)

    def test_duplicate_prediction_is_not_silently_overwritten(self):
        self.rewrite_prediction(lambda records: records.append(dict(records[1])))
        prepared = self.prepare()
        issues = read_csv(prepared / "issues.csv")
        self.assertIn("duplicate_prediction", {i["code"] for i in issues})
        output = evaluate(prepared, self.root / "run")
        self.assertEqual(read_json(output / "report.json")["inherited_diagnostic_count"], 19)

    def test_calibration_failure_suppresses_all_b_scores(self):
        prepared = self.prepare()
        reviews, splits = self.audit_inputs(prepared)
        values = read_csv(reviews)
        # Removing a single fit label violates the complete calibration panel guard.
        next(r for r in values if r["original_id"] == "a_0")["review_status"] = "uncertain"
        altered = self.root / "altered.csv"
        write_csv(altered, values)
        output = evaluate(prepared, self.root / "run", altered, splits)
        report = read_json(output / "report.json")
        self.assertEqual(report["calibration"]["status"], "pending_validation")
        self.assertEqual(report["score_b_count"], 0)

    def test_corrected_label_recomputes_correctness_and_a(self):
        prepared = self.prepare()
        reviews, _ = self.audit_inputs(prepared)
        values = read_csv(reviews)
        chosen = next(r for r in values if r["original_id"] == "a_0")
        chosen.update(review_status="corrected", reference_class_id="b")
        altered = self.root / "corrected.csv"
        write_csv(altered, values)
        output = evaluate(prepared, self.root / "run", altered)
        row = next(r for r in read_csv(output / "scores.csv") if r["original_id"] == "a_0")
        self.assertEqual(row["score_a"], "0.0")
        self.assertEqual(row["m1_correct"], "True")

    def test_exact_duplicate_bytes_create_shared_source_group(self):
        # Add an identical image under a second ID without changing legacy file sizes.
        with zipfile.ZipFile(self.archive) as archive:
            content = {name: archive.read(name) for name in archive.namelist()}
        content["train/a/a_2.png"] = content["train/a/a_0.png"]
        content["base/a_2.png"] = content["base/a_0.png"]
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in content.items():
                archive.writestr(name, data)
        protocol = read_json(self.source / "protocol.json")
        protocol["archive_bytes"] = self.archive.stat().st_size
        with (self.source / "protocol.json").open("w") as stream:
            json.dump(protocol, stream)
        prepared = self.prepare()
        rows = {r["original_id"]: r for r in read_csv(prepared / "samples.csv")}
        self.assertEqual(rows["a_0"]["source_group_id"], rows["a_2"]["source_group_id"])
        self.assertIn("same_source_prediction_conflict", {i["code"] for i in read_csv(prepared / "issues.csv")})


if __name__ == "__main__":
    unittest.main()
