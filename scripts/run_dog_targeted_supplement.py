#!/usr/bin/env python3
"""Select a targeted dog supplement from a pre-scored candidate pool."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qe_quality.dta.io import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
    digest,
    fingerprint,
    read_csv,
    read_json,
    unique_json_object,
)


def coverage_band(count: int) -> str:
    if count == 0:
        return "missing"
    if count < 10:
        return "sparse"
    if count < 30:
        return "weak"
    return "covered"


def target_count(count: int, sparse_target: int = 10, weak_target: int = 30) -> int:
    if count < sparse_target:
        return sparse_target
    if count < weak_target:
        return weak_target
    return weak_target


def latest_feature_payload(results_dir: Path, image_id: str) -> dict:
    parsed = results_dir / "parsed/features" / f"{image_id}.json"
    if parsed.exists():
        return read_json(parsed)
    attempts = sorted((results_dir / "raw/features").glob(f"{image_id}_attempt*.json"))
    for path in reversed(attempts):
        try:
            outer = read_json(path)
            return unique_json_object(outer["message"]["content"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError(f"no_parseable_feature_payload:{image_id}")


def feature_definitions(schema: dict) -> dict[str, list[str]]:
    items = schema["universal_features"] + schema.get("category_specific_features", [])
    return {item["feature_id"]: item["possible_values"] for item in items}


def state_counts(payloads: list[dict], definitions: dict[str, list[str]]) -> dict:
    result = {feature_id: Counter() for feature_id in definitions}
    for payload in payloads:
        features = payload.get("features", {})
        for feature_id, allowed in definitions.items():
            value = features.get(feature_id)
            if value in allowed:
                result[feature_id][value] += 1
    return result


def build_state_table(
    counts: dict,
    definitions: dict[str, list[str]],
    excluded_values: set[str],
    sparse_target: int,
    weak_target: int,
) -> list[dict]:
    rows = []
    for feature_id, allowed in definitions.items():
        for value in allowed:
            if value in excluded_values:
                continue
            count = counts[feature_id][value]
            band = coverage_band(count)
            target = target_count(count, sparse_target, weak_target)
            rows.append({
                "feature_id": feature_id,
                "state": value,
                "baseline_count": count,
                "baseline_band": band,
                "target_count": target,
                "baseline_deficit": max(0, target - count),
                "is_gap": band != "covered",
            })
    return rows


def solve_selection(candidates: list[dict], gaps: list[dict], metadata: dict, config: dict):
    n_candidates = len(candidates)
    n_gaps = len(gaps)
    variable_count = n_candidates + 2 * n_gaps
    objective = np.zeros(variable_count)
    band_weights = config["band_weights"]

    for index, gap in enumerate(gaps):
        weight = band_weights[gap["baseline_band"]]
        deficit = gap["baseline_deficit"]
        objective[n_candidates + index] = -config["completion_weight"] * weight
        objective[n_candidates + n_gaps + index] = -weight / deficit

    for index, row in enumerate(candidates):
        unknown_count = int(row.get("feature_unknown_count") or 0)
        objective[index] = 1e-5 * unknown_count + 1e-12 * index

    matrices = []
    lower = []
    upper = []

    exact = lil_matrix((1, variable_count))
    exact[0, :n_candidates] = 1
    matrices.append(exact.tocsr())
    lower.append(config["selection_count"])
    upper.append(config["selection_count"])

    for gap_index, gap in enumerate(gaps):
        feature_id, state = gap["feature_id"], gap["state"]
        contribution = [int(row.get(feature_id) == state) for row in candidates]
        constraint = lil_matrix((2, variable_count))
        constraint[0, :n_candidates] = contribution
        constraint[0, n_candidates + gap_index] = -gap["baseline_deficit"]
        constraint[1, :n_candidates] = [-value for value in contribution]
        constraint[1, n_candidates + n_gaps + gap_index] = 1
        matrices.append(constraint.tocsr())
        lower.extend([0, -np.inf])
        upper.extend([np.inf, 0])

    for stratum in config["candidate_strata"]:
        constraint = lil_matrix((1, variable_count))
        constraint[0, :n_candidates] = [
            int(metadata[row["image_id"]]["selection_stratum"] == stratum)
            for row in candidates
        ]
        matrices.append(constraint.tocsr())
        lower.append(config["stratum_minimum"])
        upper.append(config["stratum_maximum"])

    integrality = np.r_[np.ones(n_candidates + n_gaps), np.zeros(n_gaps)]
    variable_upper = np.r_[
        np.ones(n_candidates + n_gaps),
        [gap["baseline_deficit"] for gap in gaps],
    ]
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(vstack(matrices), lower, upper),
        options={"time_limit": 60},
    )
    if not result.success:
        raise RuntimeError(f"selection_optimizer_failed:{result.message}")
    selected = [row for index, row in enumerate(candidates) if result.x[index] > 0.5]
    if len(selected) != config["selection_count"]:
        raise RuntimeError(f"selection_count_mismatch:{len(selected)}")
    return selected, result


def contribution_details(row: dict, gaps: list[dict], weights: dict) -> tuple[list[str], list[str], float]:
    all_gaps, critical = [], []
    score = 0.0
    for gap in gaps:
        if row.get(gap["feature_id"]) != gap["state"]:
            continue
        key = f"{gap['feature_id']}={gap['state']}"
        all_gaps.append(key)
        if gap["baseline_band"] in {"missing", "sparse"}:
            critical.append(key)
        score += weights[gap["baseline_band"]]
    return all_gaps, critical, score


def scenario_counts(rows: list[dict]) -> dict[str, int]:
    return {
        "tiny_target_area_lt_0_01": sum(float(row["target_area_ratio"]) < 0.01 for row in rows),
        "small_target_area_lt_0_03": sum(float(row["target_area_ratio"]) < 0.03 for row in rows),
        "multiple_dogs": sum(int(row["visible_dog_count_in_scene"]) > 1 for row in rows),
        "occlusion_partial_or_heavy": sum(row["occlusion_level"] in {"partial", "heavy"} for row in rows),
        "truncation_partial_or_heavy": sum(row["truncation_level"] in {"partial", "heavy"} for row in rows),
        "dynamic_pose_walking_running_jumping": sum(row["pose"] in {"walking", "running", "jumping"} for row in rows),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Dog targeted supplement experiment v1",
        "",
        f"- Status: **{report['status']}**",
        f"- Frozen baseline: {report['baseline']['images']} images",
        f"- Strict candidate results: {report['candidate_pool']['strict_valid']}/400",
        f"- Confirmed non-living candidates excluded: {report['candidate_pool']['excluded_nonliving']}",
        f"- Selected supplement: {report['selection']['count']} images",
        f"- Low-frequency states resolved: {report['coverage']['resolved_gaps']}/{report['coverage']['baseline_gaps']}",
        f"- State coverage index: {report['coverage']['baseline_index']:.1%} -> {report['coverage']['post_index']:.1%}",
        f"- Readiness score: {report['scores']['baseline']:.1f} -> {report['scores']['post_supplement']:.1f}",
        "- Candidate classification: not run; this experiment evaluates feature coverage only",
        "",
        "## Selected strata",
        "",
        "| Stratum | Images |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in report["selection"]["strata"].items())
    lines += [
        "",
        "## Decision",
        "",
        "The existing candidate pool resolves every baseline gap that is feasible under the frozen targets. "
        "The remaining states require a new targeted search rather than selecting more low-value images "
        "from the same pool.",
        "",
        "The selected supplement remains `model_selected_unreviewed` until the generated review queue is checked.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    paths = {key: (ROOT / value).resolve() for key, value in config["paths"].items()}
    output = paths["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=True)

    schema = read_json(paths["schema"])
    definitions = feature_definitions(schema)
    baseline_manifest = read_csv(paths["baseline_manifest"])
    baseline_result_rows = read_csv(paths["baseline_results"] / "results.csv")
    baseline_payloads = [
        latest_feature_payload(paths["baseline_results"], row["image_id"])
        for row in baseline_manifest
    ]
    baseline_counts = state_counts(baseline_payloads, definitions)
    sparse_target = config["state_targets"]["missing_or_sparse"]
    weak_target = config["state_targets"]["weak"]
    state_table = build_state_table(
        baseline_counts,
        definitions,
        set(config["excluded_state_values"]),
        sparse_target,
        weak_target,
    )
    gaps = [row for row in state_table if row["is_gap"]]

    candidate_rows_all = read_csv(paths["candidate_results"] / "results.csv")
    candidate_metadata_rows = read_csv(paths["candidate_metadata_manifest"])
    candidate_metadata = {row["image_id"]: row for row in candidate_metadata_rows}
    candidate_target_rows = read_csv(paths["candidate_target_manifest"])
    candidate_targets = {row["image_id"]: row for row in candidate_target_rows}
    excluded_candidates = set(config["confirmed_nonliving_candidate_ids"])
    strict_candidates = [row for row in candidate_rows_all if row["feature_status"] == "ok"]
    eligible_candidates = [row for row in strict_candidates if row["image_id"] not in excluded_candidates]
    eligible_candidates.sort(key=lambda row: row["image_id"])

    selected, solver = solve_selection(eligible_candidates, gaps, candidate_metadata, config)
    selected.sort(key=lambda row: (
        -contribution_details(row, gaps, config["band_weights"])[2],
        int(row.get("feature_unknown_count") or 0),
        row["image_id"],
    ))

    selected_rows = []
    for rank, row in enumerate(selected, 1):
        metadata = candidate_metadata[row["image_id"]]
        target = candidate_targets[row["image_id"]]
        gap_states, critical_states, score = contribution_details(row, gaps, config["band_weights"])
        selected_rows.append({
            "selection_rank": rank,
            "image_id": row["image_id"],
            "target_instance_id": row["target_instance_id"],
            "selection_stratum": metadata["selection_stratum"],
            "target_area_ratio": row["target_area_ratio"],
            "visible_dog_count_in_scene": row["visible_dog_count_in_scene"],
            "feature_unknown_count": row["feature_unknown_count"],
            "gap_contribution_score": score,
            "gap_states": json.dumps(gap_states, ensure_ascii=False),
            "critical_gap_states": json.dumps(critical_states, ensure_ascii=False),
            "image_path": target["image_path"],
            "marked_image_path": target["marked_image_path"],
            "crop_image_path": target["crop_image_path"],
            "image_sha256": target["image_sha256"],
            "label_sha256": target["label_sha256"],
            "review_status": "unreviewed",
            "selection_status": "model_selected_unreviewed",
        })

    selected_fields = list(selected_rows[0])
    atomic_csv(output / "selected_candidates.csv", selected_rows, selected_fields)

    full_candidate_counts = {
        (row["feature_id"], row["state"]): sum(
            candidate.get(row["feature_id"]) == row["state"] for candidate in eligible_candidates
        )
        for row in gaps
    }
    post_state_rows = []
    for row in state_table:
        selected_count = sum(candidate.get(row["feature_id"]) == row["state"] for candidate in selected)
        post_count = row["baseline_count"] + selected_count
        key = (row["feature_id"], row["state"])
        post_state_rows.append({
            **row,
            "selected_count": selected_count,
            "post_count": post_count,
            "post_band": coverage_band(post_count),
            "target_met": post_count >= row["target_count"],
            "remaining_needed": max(0, row["target_count"] - post_count),
            "eligible_pool_count": full_candidate_counts.get(key, 0),
            "feasible_from_existing_pool": (
                row["baseline_count"] + full_candidate_counts.get(key, 0) >= row["target_count"]
                if row["is_gap"] else True
            ),
        })
    state_fields = list(post_state_rows[0])
    atomic_csv(output / "pre_post_state_coverage.csv", post_state_rows, state_fields)
    atomic_csv(output / "baseline_gap_manifest.csv", gaps, list(gaps[0]))

    search_rows = [
        {
            "feature_id": row["feature_id"],
            "state": row["state"],
            "baseline_count": row["baseline_count"],
            "selected_count": row["selected_count"],
            "post_count": row["post_count"],
            "target_count": row["target_count"],
            "new_images_needed": row["remaining_needed"],
            "baseline_band": row["baseline_band"],
            "existing_pool_exhausted": not row["feasible_from_existing_pool"],
            "search_status": "needs_new_source",
        }
        for row in post_state_rows
        if row["is_gap"] and not row["target_met"]
    ]
    atomic_csv(output / "targeted_search_requirements.csv", search_rows, list(search_rows[0]))

    combined_rows = []
    baseline_flags = config["baseline_audit_flags"]
    for row in baseline_manifest:
        combined_rows.append({
            **row,
            "dataset_source": "baseline500",
            "selection_status": "frozen_baseline",
            "selection_stratum": "",
            "quality_flag": baseline_flags.get(row["image_id"], ""),
        })
    for row in selected:
        target = candidate_targets[row["image_id"]]
        combined_rows.append({
            **target,
            "split": "supplement_unreviewed",
            "dataset_source": "candidate400",
            "selection_status": "model_selected_unreviewed",
            "selection_stratum": candidate_metadata[row["image_id"]]["selection_stratum"],
            "quality_flag": "",
        })
    combined_fields = list(combined_rows[0])
    atomic_csv(output / "combined_manifest_600.csv", combined_rows, combined_fields)
    if len({row["image_id"] for row in combined_rows}) != len(combined_rows):
        raise RuntimeError("combined_manifest_duplicate_image_id")
    if len({row["image_sha256"] for row in combined_rows}) != len(combined_rows):
        raise RuntimeError("combined_manifest_duplicate_image_hash")
    if not all(
        Path(row[path_field]).is_file()
        for row in combined_rows
        for path_field in ("image_path", "marked_image_path", "crop_image_path")
    ):
        raise RuntimeError("combined_manifest_missing_view")

    flagged_rows = [
        {
            "image_id": image_id,
            "quality_flag": reason,
            "marked_image_path": next(
                row["marked_image_path"] for row in baseline_manifest if row["image_id"] == image_id
            ),
            "recommended_action": "retain_in_frozen_baseline_but_exclude_or_replace_in_clean_release",
        }
        for image_id, reason in baseline_flags.items()
    ]
    atomic_csv(output / "baseline_audit_flags.csv", flagged_rows, list(flagged_rows[0]))

    critical_review = []
    remaining_review = []
    for row in selected_rows:
        target_list = critical_review if json.loads(row["critical_gap_states"]) else remaining_review
        target_list.append(row)
    random.Random(config["seed"]).shuffle(remaining_review)
    review_rows = []
    for row in critical_review:
        review_rows.append({
            "image_id": row["image_id"],
            "review_reason": "contributes_missing_or_sparse_state",
            "gap_states": row["gap_states"],
            "marked_image_path": row["marked_image_path"],
            "crop_image_path": row["crop_image_path"],
            "target_match": "",
            "feature_state_confirmed": "",
            "reviewer": "",
            "notes": "",
        })
    for row in remaining_review[:config["review_random_count"]]:
        review_rows.append({
            "image_id": row["image_id"],
            "review_reason": "deterministic_random_quality_audit",
            "gap_states": row["gap_states"],
            "marked_image_path": row["marked_image_path"],
            "crop_image_path": row["crop_image_path"],
            "target_match": "",
            "feature_state_confirmed": "",
            "reviewer": "",
            "notes": "",
        })
    atomic_csv(output / "review_queue.csv", review_rows, list(review_rows[0]))

    all_states = [(row["feature_id"], row["state"]) for row in state_table]
    baseline_index = sum(min(baseline_counts[f][state] / 30, 1) for f, state in all_states) / len(all_states)
    post_index = sum(
        min((baseline_counts[f][state] + sum(row.get(f) == state for row in selected)) / 30, 1)
        for f, state in all_states
    ) / len(all_states)

    baseline_coverage_report = read_json(paths["baseline_results"] / "automatic_coverage_report.json")
    combined_denominator = len(baseline_manifest) + len(selected)
    combined_strict_rate = (
        baseline_coverage_report["accepted_responses"] + len(selected)
    ) / combined_denominator
    combined_mean_validity = sum(
        (metric["valid"] + len(selected)) / combined_denominator
        for metric in baseline_coverage_report["features"]
    ) / len(definitions)
    combined_mean_evidence = sum(
        (metric["evidence_valid"] + len(selected)) / combined_denominator
        for metric in baseline_coverage_report["features"]
    ) / len(definitions)
    combined_annotation_score = 20 * (
        1 + combined_strict_rate + combined_mean_validity + combined_mean_evidence
    ) / 4
    combined_scenarios = scenario_counts(baseline_result_rows + selected)
    combined_scenario_index = sum(
        min(count / 30, 1) for count in combined_scenarios.values()
    ) / len(combined_scenarios)
    combined_scenario_score = 15 * combined_scenario_index
    post_score = 20 + combined_annotation_score + 35 * post_index + combined_scenario_score + 10

    selected_strata = Counter(candidate_metadata[row["image_id"]]["selection_stratum"] for row in selected)
    resolved_gaps = sum(row["is_gap"] and row["target_met"] for row in post_state_rows)
    pool_feasible_gaps = sum(
        row["is_gap"] and row["feasible_from_existing_pool"] for row in post_state_rows
    )
    resolved_feasible_gaps = sum(
        row["is_gap"] and row["feasible_from_existing_pool"] and row["target_met"]
        for row in post_state_rows
    )
    if resolved_feasible_gaps != pool_feasible_gaps:
        raise RuntimeError(
            f"optimizer_left_feasible_gaps:{resolved_feasible_gaps}/{pool_feasible_gaps}"
        )
    source_digests = {
        "config": digest(config_path),
        "schema": digest(paths["schema"]),
        "baseline_manifest": digest(paths["baseline_manifest"]),
        "baseline_results": digest(paths["baseline_results"] / "results.csv"),
        "candidate_target_manifest": digest(paths["candidate_target_manifest"]),
        "candidate_metadata_manifest": digest(paths["candidate_metadata_manifest"]),
        "candidate_results": digest(paths["candidate_results"] / "results.csv"),
    }
    protocol_hash = fingerprint({"config": config, "source_digests": source_digests})
    report = {
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "protocol_hash": protocol_hash,
        "status": "needs_review",
        "baseline": {
            "images": len(baseline_manifest),
            "candidate_pool_used_to_define_gaps": False,
            "audit_flags": len(baseline_flags),
        },
        "candidate_pool": {
            "planned": len(candidate_rows_all),
            "strict_valid": len(strict_candidates),
            "excluded_nonliving": len(excluded_candidates),
            "eligible": len(eligible_candidates),
        },
        "selection": {
            "count": len(selected),
            "strata": {key: selected_strata[key] for key in config["candidate_strata"]},
            "unknown_values_total": sum(int(row.get("feature_unknown_count") or 0) for row in selected),
            "solver": "scipy.optimize.milp/HiGHS",
            "solver_success": bool(solver.success),
            "solver_message": solver.message,
            "review_queue": len(review_rows),
        },
        "coverage": {
            "evaluable_states": len(state_table),
            "baseline_gaps": len(gaps),
            "resolved_gaps": resolved_gaps,
            "remaining_gaps": len(gaps) - resolved_gaps,
            "pool_feasible_gaps": pool_feasible_gaps,
            "resolved_feasible_gaps": resolved_feasible_gaps,
            "baseline_index": baseline_index,
            "post_index": post_index,
        },
        "combined_challenge_scenarios": {
            "counts": combined_scenarios,
            "coverage_index": combined_scenario_index,
            "score": combined_scenario_score,
        },
        "scores": {
            "baseline": read_json(paths["baseline_score"])["score"]["total"],
            "post_supplement": round(post_score, 1),
            "score_is_not_accuracy": True,
        },
        "classification": {
            "baseline_dog_positive_recall": 0.936,
            "selected_supplement_classification_run": False,
        },
        "source_digests": source_digests,
        "outputs": {
            "selected_candidates": "selected_candidates.csv",
            "combined_manifest": "combined_manifest_600.csv",
            "pre_post_state_coverage": "pre_post_state_coverage.csv",
            "targeted_search_requirements": "targeted_search_requirements.csv",
            "review_queue": "review_queue.csv",
        },
    }
    atomic_json(output / "report.json", report)
    atomic_text(output / "report.md", markdown_report(report))

    command = f".venv/bin/python scripts/run_dog_targeted_supplement.py --config {config_path.relative_to(ROOT)}"
    log = f"""# Research log: targeted dog supplementation v1

## Decision

Test whether 100 strictly valid images from the existing 400-image candidate pool can
close the feature-state gaps frozen from the original 500-image baseline.

## Hypothesis and stop criteria

- Hypothesis: constrained selection improves the state coverage index and resolves all
  gaps that are feasible from the existing candidate pool.
- Primary metric: feature-state coverage index, higher is better.
- Success: every pool-feasible frozen gap meets its target.
- Stop: do not select additional low-value candidates after all feasible gaps close;
  unresolved gaps move to a new targeted search.

## Fixed factors

- Missing/sparse target: {sparse_target}
- Weak target: {weak_target}
- Selection count: {config['selection_count']}
- Per-stratum bounds: {config['stratum_minimum']} to {config['stratum_maximum']}
- Candidate feature protocol: dog_v2 / dog-v2.1-candidate400-upscaled
- Seed: {config['seed']}

## Execution

```bash
{command}
```

- Protocol hash: `{protocol_hash}`
- Solver: scipy.optimize.milp/HiGHS
- Raw inputs and SHA-256 hashes: `report.json`

## Result

- Selected: {len(selected)}
- Frozen gaps: {len(gaps)}
- Resolved: {resolved_gaps}
- Remaining: {len(gaps) - resolved_gaps}
- Coverage index: {baseline_index:.6f} -> {post_index:.6f}
- Readiness score: {report['scores']['baseline']:.1f} -> {report['scores']['post_supplement']:.1f}

## Boundary

The result supports feature-coverage improvement only. The selected 100 images have
not yet completed the generated review queue or an independent classification run.
"""
    atomic_text(output / "research_log.md", log)
    claims = f"""# Claim-evidence ledger

| Claim | Evidence | Status | Boundary |
|---|---|---|---|
| The candidate selection improves state coverage. | `pre_post_state_coverage.csv`; index {baseline_index:.6f} -> {post_index:.6f} | supported | Frozen dog_v2 features only |
| The selection resolves all gaps feasible from the existing pool. | `pre_post_state_coverage.csv`; {resolved_gaps} resolved | supported | Uses strict candidate results after four confirmed non-living targets are excluded |
| The combined 600-image manifest is ready for final release. | `review_queue.csv` is incomplete | unsupported | Status remains needs_review |
| The supplement improves dog classification. | No candidate classification run | unsupported | No recognition claim is made |
"""
    atomic_text(output / "claim_evidence.md", claims)
    readme = """# Targeted dog supplement v1

This directory preserves a model-selected, unreviewed 100-image supplement derived
from the frozen 500-image baseline and the pre-scored 400-image candidate pool.

Start with `report.md`, then inspect `targeted_search_requirements.csv` and complete
`review_queue.csv`. Do not treat `combined_manifest_600.csv` as a final release until
the review queue is complete.
"""
    atomic_text(output / "README.md", readme)
    atomic_json(output / "pipeline_status.json", {
        "status": "needs_review",
        "stage": "targeted_selection_complete",
        "selected": len(selected),
        "review_remaining": len(review_rows),
        "protocol_hash": protocol_hash,
    })
    print(json.dumps({
        "status": report["status"],
        "selected": len(selected),
        "resolved_gaps": resolved_gaps,
        "remaining_gaps": len(gaps) - resolved_gaps,
        "coverage_before": baseline_index,
        "coverage_after": post_index,
        "score_before": report["scores"]["baseline"],
        "score_after": report["scores"]["post_supplement"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
