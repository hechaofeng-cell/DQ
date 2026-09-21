#!/usr/bin/env python3
"""Freeze random, pure-gap, and coverage-utility dog supplements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from openimages_cua_vsl import (
    StateSchema,
    coverage_index,
    feature_rows_by_id,
    read_csv,
    state_counts,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/openimages_v7_cua_vsl_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def greedy_select(
    candidates: list[dict], baseline_counts: Counter, schema: StateSchema,
    budget: int, cap: int, state_utility: dict[tuple[str, str], float] | None,
) -> tuple[list[dict], list[dict]]:
    counts = Counter(baseline_counts)
    pool = list(candidates)
    selected = []
    audit = []
    for rank in range(1, budget + 1):
        scored = []
        for row in pool:
            pairs = schema.state_pairs(row)
            improving = sorted(pair for pair in pairs if counts[pair] < cap)
            hits = len(improving)
            rarity = sum((cap - counts[pair]) / cap for pair in improving)
            weighted_gain = sum(
                (state_utility.get(pair, 1.0) if state_utility is not None else 1.0) / cap
                for pair in improving
            )
            unknown_count = int(float(row.get("feature_unknown_count") or 0))
            primary = weighted_gain if state_utility is not None else hits
            scored.append((primary, hits, rarity, -unknown_count, row["image_id"], row, improving, weighted_gain))
        if not scored:
            raise ValueError(f"candidate pool exhausted at rank {rank}")
        primary, hits, rarity, _, _, chosen, improving, weighted_gain = max(
            scored, key=lambda item: item[:5]
        )
        pool.remove(chosen)
        selected.append(chosen)
        before = {pair: counts[pair] for pair in improving}
        counts.update(schema.state_pairs(chosen))
        audit.append({
            "selection_rank": rank,
            "image_id": chosen["image_id"],
            "primary_score": primary,
            "marginal_state_hits": hits,
            "rarity_tiebreak_score": rarity,
            "utility_weighted_gain": weighted_gain,
            "improving_states": json.dumps(
                [
                    {
                        "feature_id": pair[0], "state": pair[1],
                        "count_before": before[pair], "count_after": counts[pair],
                        "utility": state_utility.get(pair) if state_utility is not None else None,
                    }
                    for pair in improving
                ],
                ensure_ascii=False,
            ),
        })
    return selected, audit


def select_two_stage_cua(
    candidates: list[dict], baseline_counts: Counter, schema: StateSchema,
    gap_budget: int, utility_budget: int, cap: int,
    state_utility: dict[tuple[str, str], float],
) -> tuple[list[dict], list[dict]]:
    if gap_budget + utility_budget <= 0:
        raise ValueError("CUA two-stage budget must be positive")
    gap_selected, gap_audit = greedy_select(
        candidates, baseline_counts, schema, gap_budget, cap, state_utility=None
    )
    counts = Counter(baseline_counts)
    selected_ids = {row["image_id"] for row in gap_selected}
    for row in gap_selected:
        counts.update(schema.state_pairs(row))
    pool = [row for row in candidates if row["image_id"] not in selected_ids]
    selected = list(gap_selected)
    audit = [
        {**row, "selection_stage": "gap", "utility_novelty_score": ""}
        for row in gap_audit
    ]
    for offset in range(1, utility_budget + 1):
        scored = []
        for row in pool:
            pairs = schema.state_pairs(row)
            contributions = {
                pair: state_utility.get(pair, 0.0) / math.sqrt(counts[pair] + 1)
                for pair in pairs
            }
            score = sum(contributions.values()) / len(contributions) if contributions else 0.0
            unknown_count = int(float(row.get("feature_unknown_count") or 0))
            scored.append((score, -unknown_count, row["image_id"], row, contributions))
        if not scored:
            raise ValueError(f"candidate pool exhausted in CUA utility stage at rank {offset}")
        score, _, _, chosen, contributions = max(scored, key=lambda item: item[:3])
        pool.remove(chosen)
        selected.append(chosen)
        counts.update(schema.state_pairs(chosen))
        ranked_contributions = sorted(
            contributions.items(), key=lambda item: (-item[1], item[0])
        )
        audit.append({
            "selection_rank": gap_budget + offset,
            "image_id": chosen["image_id"],
            "selection_stage": "utility_novelty",
            "primary_score": score,
            "marginal_state_hits": sum(counts[pair] <= cap for pair in contributions),
            "rarity_tiebreak_score": "",
            "utility_weighted_gain": "",
            "utility_novelty_score": score,
            "improving_states": json.dumps(
                [
                    {
                        "feature_id": pair[0], "state": pair[1],
                        "smoothed_error": state_utility.get(pair, 0.0),
                        "novelty_contribution": contribution,
                    }
                    for pair, contribution in ranked_contributions
                ],
                ensure_ascii=False,
            ),
        })
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-priority", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = Path(config["data"])
    output = Path(config["output"])
    state_priority_path = args.state_priority or output / "state_priority" / "state_priority.csv"
    if not state_priority_path.is_file():
        raise FileNotFoundError(f"run OOF utility estimation first: {state_priority_path}")
    schema = StateSchema.from_path(Path(config["schema"]))
    base_manifest = read_csv(data / "manifests/train_base.csv")
    candidate_manifest_rows = read_csv(data / "manifests/dog_candidates.csv")
    candidate_manifest = {row["sample_id"]: row for row in candidate_manifest_rows}
    base_features = feature_rows_by_id(Path(config["base_features"]))
    candidate_features = feature_rows_by_id(Path(config["candidate_features"]))
    base_dog_ids = {row["sample_id"] for row in base_manifest if row["class_name"] == "dog"}
    candidate_ids = set(candidate_manifest)
    if set(base_features) != base_dog_ids:
        raise ValueError("base features do not match base dog manifest")
    if set(candidate_features) != candidate_ids:
        raise ValueError("candidate features do not match candidate manifest")
    if base_dog_ids & candidate_ids:
        raise ValueError("base dog and candidate IDs overlap")

    priority_rows = read_csv(state_priority_path)
    state_utility = {
        (row["feature_id"], row["state"]): float(row["mean_smoothed_error"])
        for row in priority_rows
    }
    cap = int(config["coverage_cap"])
    budget = int(config["supplement_budget"])
    base_counts = state_counts(list(base_features.values()), schema)
    candidate_rows = list(candidate_features.values())
    gap_selected, gap_audit = greedy_select(
        candidate_rows, base_counts, schema, budget, cap, state_utility=None
    )
    cua_config = config["cua_selection"]
    gap_stage_budget = int(cua_config["gap_stage_budget"])
    utility_stage_budget = int(cua_config["utility_stage_budget"])
    if gap_stage_budget + utility_stage_budget != budget:
        raise ValueError("CUA stage budgets must sum to the supplement budget")
    cua_selected, cua_audit = select_two_stage_cua(
        candidate_rows, base_counts, schema,
        gap_stage_budget, utility_stage_budget, cap, state_utility,
    )

    selection_dir = output / "selections"
    manifest_dir = output / "manifests"
    fields = list(base_manifest[0])
    write_csv(selection_dir / "gap300.csv", gap_audit)
    write_csv(selection_dir / "cua300.csv", cua_audit)
    selected_sets = {
        "gap300": [row["image_id"] for row in gap_selected],
        "cua300": [row["image_id"] for row in cua_selected],
    }
    random_reports = {}
    for random_index, seed in enumerate(config["random_selection_seeds"]):
        rng = random.Random(int(seed))
        selected = rng.sample(candidate_rows, budget)
        name = f"random300_r{random_index}"
        selected_ids = [row["image_id"] for row in selected]
        selected_sets[name] = selected_ids
        write_csv(
            selection_dir / f"{name}.csv",
            [
                {"selection_rank": rank, "image_id": image_id, "selection_seed": seed}
                for rank, image_id in enumerate(selected_ids, 1)
            ],
        )
        random_counts = state_counts(list(base_features.values()) + selected, schema)
        random_reports[name] = {
            "seed": seed,
            "coverage": coverage_index(random_counts, schema, cap),
        }

    write_csv(manifest_dir / "train_base.csv", base_manifest, fields)
    for name, selected_ids in selected_sets.items():
        if len(selected_ids) != budget or len(set(selected_ids)) != budget:
            raise ValueError(f"{name}: selection is not {budget} unique samples")
        rows = base_manifest + [candidate_manifest[sample_id] for sample_id in selected_ids]
        write_csv(manifest_dir / f"train_{name}.csv", rows, fields)

    gap_counts = state_counts(list(base_features.values()) + gap_selected, schema)
    cua_counts = state_counts(list(base_features.values()) + cua_selected, schema)
    coverages = {
        "base": coverage_index(base_counts, schema, cap),
        "gap300": coverage_index(gap_counts, schema, cap),
        "cua300": coverage_index(cua_counts, schema, cap),
        **{name: value["coverage"] for name, value in random_reports.items()},
    }
    overlap = {}
    names = list(selected_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap[f"{left}__{right}"] = len(set(selected_sets[left]) & set(selected_sets[right]))
    report = {
        "experiment_id": config["experiment_id"],
        "budget": budget,
        "coverage_cap": cap,
        "cua_selection": cua_config,
        "coverages": coverages,
        "random_coverage_mean": sum(value["coverage"] for value in random_reports.values()) / len(random_reports),
        "random_coverage_min": min(value["coverage"] for value in random_reports.values()),
        "random_coverage_max": max(value["coverage"] for value in random_reports.values()),
        "overlap": overlap,
        "input_sha256": {
            "config": sha256(args.config),
            "state_priority": sha256(state_priority_path),
            "train_base": sha256(data / "manifests/train_base.csv"),
            "dog_candidates": sha256(data / "manifests/dog_candidates.csv"),
            "base_features": sha256(Path(config["base_features"])),
            "candidate_features": sha256(Path(config["candidate_features"])),
        },
    }
    write_json(output / "selections" / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
