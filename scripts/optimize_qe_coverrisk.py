#!/usr/bin/env python3
"""Solve finite-pool cover/risk selection, retaining every solver certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


CLASSES = ("dog", "cat", "horse", "sheep", "person", "car")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_class(path):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    matrix = data["matrix"]
    n = len(data["sample_ids"])
    assert matrix.shape == (n, 71), (path, matrix.shape)
    assert n and np.all(np.isin(matrix, [0, 1]))
    assert len(set(data["sample_ids"].tolist())) == n
    for field in ("risk_lower", "risk_midpoint", "risk_upper", "high_risk"):
        assert data[field].shape == (n,), (path, field)
        assert np.isfinite(data[field]).all()
    assert np.all(data["risk_lower"] <= data["risk_midpoint"] + 1e-8)
    assert np.all(data["risk_midpoint"] <= data["risk_upper"] + 1e-8)
    assert np.isin(data["high_risk"], [0, 1]).all()
    data["reachable"] = np.flatnonzero(matrix.sum(axis=0) > 0)
    return data


def feasible(indices, data, k=None):
    ids = np.asarray(indices, dtype=int)
    if len(set(ids.tolist())) != len(ids) or (k is not None and len(ids) != k):
        return False
    if np.any(ids < 0) or np.any(ids >= len(data["sample_ids"])):
        return False
    columns = data["reachable"]
    return not len(columns) or bool(np.all(data["matrix"][ids][:, columns].sum(axis=0) >= 1))


def greedy_cover(data):
    matrix = data["matrix"][:, data["reachable"]]
    missing = np.ones(matrix.shape[1], dtype=bool)
    selected = []
    while missing.any():
        gain = matrix[:, missing].sum(axis=1).astype(int)
        gain[selected] = -1
        idx = int(np.argmax(gain))
        if gain[idx] <= 0:
            raise RuntimeError("Unreachable column erroneously included in cover constraints")
        selected.append(idx)
        missing &= ~matrix[idx].astype(bool)
    return selected


def summary(indices, data):
    ids = np.asarray(indices, dtype=int)
    covered = np.flatnonzero(data["matrix"][ids].sum(axis=0) > 0)
    return {
        "selected_indices": ids.tolist(),
        "sample_ids": data["sample_ids"][ids].tolist(),
        "count": len(ids),
        "covered_state_indices": covered.tolist(),
        "covered_states": len(covered),
        "reachable_states": len(data["reachable"]),
        "reachable_coverage": float(len(covered) / len(data["reachable"])) if len(data["reachable"]) else None,
        "fixed_dictionary_coverage": float(len(covered) / 71),
        "risk_lower_sum": float(data["risk_lower"][ids].sum()),
        "risk_lower_mean": float(data["risk_lower"][ids].mean()) if len(ids) else None,
        "risk_midpoint_mean": float(data["risk_midpoint"][ids].mean()) if len(ids) else None,
        "risk_upper_mean": float(data["risk_upper"][ids].mean()) if len(ids) else None,
        "high_risk_count": int(data["high_risk"][ids].sum()),
        "high_risk_rate": float(data["high_risk"][ids].mean()) if len(ids) else None,
    }


def solve(data, objective, time_limit, k=None, fallback=None, objective_scale=1.0, maximize=False):
    n = len(data["sample_ids"])
    a = csc_matrix(data["matrix"][:, data["reachable"]].T, dtype=float)
    lb = np.ones(a.shape[0])
    ub = np.full(a.shape[0], np.inf)
    if k is not None:
        a = vstack([a, csc_matrix(np.ones((1, n)))], format="csc")
        lb = np.append(lb, k)
        ub = np.append(ub, k)
    started = time.monotonic()
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(a, lb, ub),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    elapsed = time.monotonic() - started
    ids = None
    incumbent_valid = False
    if result.x is not None:
        rounded = np.rint(result.x)
        ids_try = np.flatnonzero(rounded > 0.5).tolist()
        incumbent_valid = bool(np.max(np.abs(result.x - rounded)) < 1e-5 and feasible(ids_try, data, k))
        if incumbent_valid:
            ids = ids_try
    used_fallback = ids is None
    if ids is None:
        if fallback is None or not feasible(fallback, data, k):
            raise RuntimeError(f"Solver supplied no verified feasible incumbent: {result.message}")
        ids = fallback
    certificate = {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_proved_optimal": bool(result.status == 0 and incumbent_valid),
        "solver_incumbent_verified": incumbent_valid,
        "used_fallback": used_fallback,
        "time_limit_seconds": time_limit,
        "elapsed_seconds": elapsed,
        "objective_scale": objective_scale,
        "objective_direction": "maximize" if maximize else "minimize",
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count),
        "solver_minimization_fun": None if result.fun is None else float(result.fun),
        "solver_minimization_dual_bound": None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound),
    }
    bound = certificate["solver_minimization_dual_bound"]
    certificate["objective_bound"] = None if bound is None else float((-bound if maximize else bound) / objective_scale)
    certificate.update(summary(ids, data))
    return certificate


def augmented_cover(base_ids, data, k):
    selected = list(base_ids)
    order = np.argsort(-data["risk_lower"], kind="stable")
    existing = set(selected)
    for i in order:
        if len(selected) == k:
            break
        if int(i) not in existing:
            selected.append(int(i))
            existing.add(int(i))
    assert feasible(selected, data, k)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/QE_VisualState_CoverRisk_v1_20261013"))
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    args = parser.parse_args()
    output = args.root / "optimization"
    output.mkdir(parents=True, exist_ok=True)
    data = {cls: load_class(args.root / "prepared" / f"{cls}.npz") for cls in args.classes}
    minimum = {}
    for cls in args.classes:
        d = data[cls]
        result = solve(d, np.ones(len(d["sample_ids"])), args.time_limit, fallback=greedy_cover(d))
        minimum[cls] = result
        save_json(output / f"minimum_cover_{cls}.json", result)
        print(json.dumps({"phase": "minimum_cover", "class": cls, "count": result["count"], "optimal": result["solver_proved_optimal"], "gap": result["mip_gap"]}), flush=True)
    k = max(r["count"] for r in minimum.values())
    max_feasible = min(len(d["sample_ids"]) for d in data.values())
    budgets = sorted({b for b in [k, max(k, 20), max(k, 40), max(k, 100)] if b <= max_feasible})
    protocol = {
        "classes": args.classes,
        "fixed_dictionary_states_per_class": 71,
        "common_compact_budget_per_class": k,
        "common_budget_proved_minimal": all(r["solver_proved_optimal"] for r in minimum.values()),
        "budgets_per_class": budgets,
        "coverage_requirement": "Every state observed in the prepared candidate pool has at least one selected target, separately for every class.",
        "primary_objective": "Maximize total risk lower bound with exact per-class count and full reachable-state coverage.",
        "risk_rate": "Fraction of selected targets assigned pooled per-class midpoint Q5; it is distinct from mean risk score.",
        "not_a_claim": "Full reachable-state coverage is not full predefined-dictionary coverage, combinatorial coverage, or deployment sufficiency.",
    }
    save_json(output / "protocol.json", protocol)
    results = {}
    for budget in budgets:
        budget_results = {}
        for cls in args.classes:
            d = data[cls]
            raw_scaled = d["risk_lower"] * 12
            round_error = float(np.max(np.abs(raw_scaled - np.rint(raw_scaled))))
            integer_scaled = round_error < 1e-5
            cost = np.rint(raw_scaled) if integer_scaled else raw_scaled
            fallback = augmented_cover(minimum[cls]["selected_indices"], d, budget)
            constrained = solve(d, -cost, args.time_limit, budget, fallback, 12, True)
            constrained["integer_scaled_risk_objective"] = integer_scaled
            constrained["risk_scale_rounding_error_max"] = round_error
            top_ids = np.argsort(-d["risk_lower"], kind="stable")[:budget].tolist()
            top_risk = summary(top_ids, d)
            top_risk["analytical_optimum"] = True
            top_risk["objective"] = "risk_lower_sum_without_coverage_constraints"
            top_risk["analytical_high_risk_count_upper_bound"] = min(budget, int(d["high_risk"].sum()))
            high_ids = np.lexsort((-d["risk_lower"], -d["high_risk"].astype(float)))[:budget].tolist()
            high_summary = summary(high_ids, d)
            high_summary["analytical_optimum"] = True
            high_summary["objective"] = "high_risk_count_without_coverage_constraints"
            budget_results[cls] = dict(constrained)
            budget_results[cls].update({"coverage_constrained_max_risk": constrained, "unconstrained_top_risk": top_risk, "unconstrained_top_high_risk_count": high_summary})
            save_json(output / f"budget_{budget}_{cls}.json", budget_results[cls])
            print(json.dumps({"phase": "maximize_risk", "budget": budget, "class": cls, "optimal": constrained["solver_proved_optimal"], "gap": constrained["mip_gap"], "risk_lower_mean": constrained["risk_lower_mean"], "high_risk_rate": constrained["high_risk_rate"]}), flush=True)
        results[str(budget)] = budget_results
    highest_q5 = {}
    for cls in args.classes:
        d = data[cls]
        fallback = results[str(k)][cls]["selected_indices"]
        constrained = solve(d, -d["high_risk"].astype(float), args.time_limit, k, fallback, 1, True)
        constrained["objective"] = "high_risk_count_under_full_reachable_coverage"
        constrained["risk_objective_not_optimized"] = True
        highest_q5[cls] = constrained
        save_json(output / f"highest_q5_budget_{k}_{cls}.json", constrained)
        print(json.dumps({"phase": "maximize_q5_count", "budget": k, "class": cls, "optimal": constrained["solver_proved_optimal"], "gap": constrained["mip_gap"], "high_risk_count": constrained["high_risk_count"]}), flush=True)
    save_json(output / "solutions.json", {"compact_budget_per_class": k, "protocol": protocol, "minimum_cover": minimum, "budgets": results, "highest_q5_under_full_coverage": highest_q5})


if __name__ == "__main__":
    main()
