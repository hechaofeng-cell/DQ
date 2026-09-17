#!/usr/bin/env python3
"""Freeze random-K and dog_v3.1 gap-K supplements from one candidate pool."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/openimages_v7_dog_gap_v1"


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def values(row: dict, key: str) -> list[str]:
    value = row.get(key, "")
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return [value]


def state_pairs(row: dict, feature_ids: list[str], excluded_states: set[str]) -> set[tuple[str, str]]:
    return {
        (feature_id, state)
        for feature_id in feature_ids
        for state in values(row, feature_id)
        if state not in excluded_states
    }


def counts_for(rows: list[dict], feature_ids: list[str], excluded_states: set[str]) -> Counter:
    counts = Counter()
    for row in rows:
        counts.update(state_pairs(row, feature_ids, excluded_states))
    return counts


def coverage(counts: Counter, states: list[tuple[str, str]], cap: int) -> float:
    return sum(min(counts[state] / cap, 1.0) for state in states) / len(states)


def select_gap(
    candidates: list[dict], baseline_counts: Counter, feature_ids: list[str],
    excluded_states: set[str], budget: int, cap: int,
) -> tuple[list[dict], list[dict]]:
    counts = Counter(baseline_counts)
    pool = list(candidates)
    selected = []
    audit = []
    for rank in range(1, budget + 1):
        scored = []
        for row in pool:
            pairs = state_pairs(row, feature_ids, excluded_states)
            improving = sorted(pair for pair in pairs if counts[pair] < cap)
            # Each hit below the cap increases the published coverage index by 1/cap.
            score = len(improving)
            rarity = sum((cap - counts[pair]) / cap for pair in improving)
            unknown_count = int(float(row.get("feature_unknown_count") or 0))
            scored.append((score, rarity, -unknown_count, row["image_id"], row, improving))
        if not scored:
            raise ValueError(f"candidate pool exhausted at rank {rank}")
        score, rarity, _, _, chosen, improving = max(scored, key=lambda item: item[:4])
        pool.remove(chosen)
        selected.append(chosen)
        counts.update(state_pairs(chosen, feature_ids, excluded_states))
        audit.append({
            "selection_rank": rank,
            "image_id": chosen["image_id"],
            "marginal_state_hits": score,
            "rarity_tiebreak_score": rarity,
            "improving_states": json.dumps([f"{key}={state}" for key, state in improving], ensure_ascii=False),
        })
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/openimages_v7_dog_gap_experiment_v1.json")
    parser.add_argument("--schema", type=Path, default=ROOT / "configs/dog_feature_schema_v3_1.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/openimages_v7_dog_supplements_v1")
    parser.add_argument("--exclude-ids", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    definitions = schema["universal_features"] + schema.get("category_specific_features", [])
    feature_ids = [definition["feature_id"] for definition in definitions]
    excluded_states = {"unknown", "other"}
    all_states = [
        (definition["feature_id"], state)
        for definition in definitions
        for state in definition["possible_values"]
        if state not in excluded_states
    ]
    excluded_ids = set()
    if args.exclude_ids:
        excluded_ids = {line.strip().split(",", 1)[0] for line in args.exclude_ids.read_text().splitlines() if line.strip()}

    base_manifest = read_csv(args.data / "manifests/train_base.csv")
    candidate_rows = read_csv(args.data / "manifests/dog_candidates.csv")
    base_ids = {row["sample_id"] for row in base_manifest if row["class_name"] == "dog"}
    candidate_ids = {row["sample_id"] for row in candidate_rows}
    base_results = read_csv(args.base_features)
    candidate_results = read_csv(args.candidate_features)
    if {row["image_id"] for row in base_results} != base_ids:
        raise ValueError("baseline feature results do not match the frozen dog baseline")
    if {row["image_id"] for row in candidate_results} != candidate_ids:
        raise ValueError("candidate feature results do not match the frozen candidate pool")
    if any(row.get("feature_status") != "ok" for row in base_results):
        raise ValueError("baseline has failed feature responses; resolve them before selection")
    if base_ids & excluded_ids:
        raise ValueError("baseline exclusions would change the frozen training set; revise the data version first")
    base_features = base_results
    candidate_features = [
        row for row in candidate_results
        if row.get("feature_status") == "ok" and row["image_id"] not in excluded_ids
    ]
    budget = int(config["counts"]["supplement_budget"])
    if len(candidate_features) < budget:
        raise ValueError(f"need {budget} valid candidates, found {len(candidate_features)}")

    base_counts = counts_for(base_features, feature_ids, excluded_states)
    gap_selected, gap_audit = select_gap(
        candidate_features, base_counts, feature_ids, excluded_states, budget, cap=30
    )
    rng = random.Random(config["selection_seed"] + 1)
    random_selected = rng.sample(candidate_features, budget)

    candidate_manifest = {row["sample_id"]: row for row in candidate_rows}
    gap_ids = [row["image_id"] for row in gap_selected]
    random_ids = [row["image_id"] for row in random_selected]
    missing = (set(gap_ids) | set(random_ids)) - set(candidate_manifest)
    if missing:
        raise ValueError(f"feature IDs missing from candidate manifest: {sorted(missing)[:5]}")

    gap_manifest = base_manifest + [candidate_manifest[image_id] for image_id in gap_ids]
    random_manifest = base_manifest + [candidate_manifest[image_id] for image_id in random_ids]
    manifest_fields = list(base_manifest[0])
    write_csv(args.output / "selections/gap300.csv", gap_audit)
    write_csv(
        args.output / "selections/random300.csv",
        [{"selection_rank": index, "image_id": image_id} for index, image_id in enumerate(random_ids, 1)],
    )
    write_csv(args.output / "manifests/train_base.csv", base_manifest, manifest_fields)
    write_csv(args.output / "manifests/train_random300.csv", random_manifest, manifest_fields)
    write_csv(args.output / "manifests/train_gap300.csv", gap_manifest, manifest_fields)

    random_counts = counts_for(base_features + random_selected, feature_ids, excluded_states)
    gap_counts = counts_for(base_features + gap_selected, feature_ids, excluded_states)
    coverage_values = {
        "base": coverage(base_counts, all_states, 30),
        "random300": coverage(random_counts, all_states, 30),
        "gap300": coverage(gap_counts, all_states, 30),
    }
    report = {
        "experiment_id": config["experiment_id"],
        "feature_protocol": schema["schema_version"],
        "valid_base_dogs": len(base_features),
        "valid_candidates": len(candidate_features),
        "budget": budget,
        "coverage_cap_per_state": 30,
        "coverage": coverage_values,
        "delta_gap_minus_base": coverage_values["gap300"] - coverage_values["base"],
        "delta_gap_minus_random": coverage_values["gap300"] - coverage_values["random300"],
        "random_gap_overlap": len(set(random_ids) & set(gap_ids)),
        "excluded_ids": len(excluded_ids),
        "selection_review_status": "automatic_only_no_human_review",
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
