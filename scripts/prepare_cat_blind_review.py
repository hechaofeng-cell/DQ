#!/usr/bin/env python3
"""Prepare blind cat_v1.0 annotation sheets for two human annotators."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from openimages_cua_vsl import StateSchema
from run_fixed_train_test_adequacy_experiment import read_csv, sha256
from run_resnet18_dog_coverage_comparison import write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/fixed_train_cat_visual_test_adequacy_20260920"
SCHEMA_PATH = ROOT / "configs/cat_feature_schema_v1_0.json"
OUTPUT = ROOT / "artifacts/cat_v1_0_human_review_20260920"
SEED = 20260920


def stable_key(sample_id: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{sample_id}".encode()).hexdigest()


def blank_row(row: dict, task_index: int, schema: StateSchema) -> dict:
    output = {
        "task_index": task_index,
        "sample_id": row["sample_id"],
        "image_id": row["image_id"],
        "source_group": row["candidate_source"],
        "image_path": row["image_path"],
        "crop_path": row["crop_path"],
        "annotator_id": "",
        "review_status": "",
    }
    for feature in schema.fields:
        output[feature.feature_id] = ""
    output["review_notes"] = ""
    return output


def main() -> None:
    schema = StateSchema.from_path(SCHEMA_PATH)
    candidates = [row for row in read_csv(SOURCE / "candidate_pool.csv") if row["class_name"] == "cat"]
    rank_rows = read_csv(SOURCE / "orders" / "gap_cat_order.csv")
    ranks = {row["sample_id"]: int(row["rank"]) for row in rank_rows}
    if len(candidates) != 200 or len(ranks) != 200:
        raise ValueError("expected 200 cat candidates and coverage ranks")
    for row in candidates:
        row["rank_quartile"] = min(3, (ranks[row["sample_id"]] - 1) // 50)

    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in candidates:
        cells[(row["candidate_source"], row["rank_quartile"])].append(row)
    selected = []
    for cell, rows in sorted(cells.items()):
        if len(rows) < 10:
            raise ValueError(f"double-label stratum {cell} has only {len(rows)} samples")
        rows.sort(key=lambda row: stable_key(row["sample_id"], f"double:{cell}"))
        selected.extend(rows[:10])
    if len(selected) != 80 or len({row["sample_id"] for row in selected}) != 80:
        raise ValueError("double-label subset must contain 80 unique cats")

    annotator_a = sorted(candidates, key=lambda row: stable_key(row["sample_id"], "annotator_a"))
    annotator_b = sorted(selected, key=lambda row: stable_key(row["sample_id"], "annotator_b"))
    write_csv(OUTPUT / "annotator_a_all_200_blind.csv", [
        blank_row(row, index, schema) for index, row in enumerate(annotator_a, 1)
    ])
    write_csv(OUTPUT / "annotator_b_double_80_blind.csv", [
        blank_row(row, index, schema) for index, row in enumerate(annotator_b, 1)
    ])
    write_csv(OUTPUT / "double_subset_audit.csv", [
        {
            "sample_id": row["sample_id"], "candidate_source": row["candidate_source"],
            "coverage_rank": ranks[row["sample_id"]], "rank_quartile": row["rank_quartile"] + 1,
        }
        for row in sorted(selected, key=lambda row: row["sample_id"])
    ])
    protocol = {
        "seed": SEED,
        "schema": "cat_v1.0",
        "annotator_a_count": 200,
        "annotator_b_count": 80,
        "double_subset_rule": "10 samples from each candidate_source x coverage_rank_quartile cell",
        "blinding": "annotation sheets exclude VLM labels, model predictions, correctness, and coverage rank",
        "source_sha256": sha256(SOURCE / "candidate_pool.csv"),
        "schema_sha256": sha256(SCHEMA_PATH),
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "README.md").write_text(
        "# cat_v1.0盲法人工复核任务\n\n"
        "- 标注员A填写`annotator_a_all_200_blind.csv`中的全部200张。\n"
        "- 标注员B独立填写`annotator_b_double_80_blind.csv`中的80张。\n"
        "- 两人不得查看VLM标签、模型预测、错误结果、Coverage排名或对方标注。\n"
        "- `review_status`仅允许`complete`或`invalid_target`。\n"
        "- 集合特征使用JSON数组字符串，例如`[\"head\",\"tail\"]`。\n"
        "- 完成后才能计算一致性、裁决并冻结`cat_v1.0_human_frozen.csv`。\n"
        "- `double_subset_audit.csv`只供实验管理员审计，不提供给标注员。\n",
        encoding="utf-8",
    )
    print({"status": "prepared", "output": str(OUTPUT), "annotator_a": 200, "annotator_b": 80})


if __name__ == "__main__":
    main()
