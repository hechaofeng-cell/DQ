#!/usr/bin/env python3
"""Joint equal-class-budget fractional MILP with auditable Dinkelbach certificates."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csc_matrix, hstack, vstack

from optimize_qe_coverrisk import CLASSES, feasible, load_class, save_json, summary


def certificate(result, elapsed, limit):
    return {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_proved_optimal": bool(result.status == 0),
        "elapsed_seconds": elapsed,
        "time_limit_seconds": limit,
        "solver_minimization_fun": None if result.fun is None else float(result.fun),
        "solver_minimization_dual_bound": None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count),
    }


def make_constraints(data, min_budget, max_budget):
    counts = [len(data[cls]["sample_ids"]) for cls in CLASSES]
    offsets = np.cumsum([0] + counts)
    n = sum(counts)
    state_a = block_diag([csc_matrix(data[cls]["matrix"][:, data[cls]["reachable"]].T, dtype=float) for cls in CLASSES], format="csc")
    state_a = hstack([state_a, csc_matrix((state_a.shape[0], 1))], format="csc")
    class_a = np.zeros((len(CLASSES), n + 1))
    for i in range(len(CLASSES)):
        class_a[i, offsets[i]:offsets[i + 1]] = 1
        class_a[i, -1] = -1
    a = vstack([state_a, csc_matrix(class_a)], format="csc")
    lb = np.concatenate([np.ones(state_a.shape[0]), np.zeros(len(CLASSES))])
    ub = np.concatenate([np.full(state_a.shape[0], np.inf), np.zeros(len(CLASSES))])
    bounds = Bounds(np.concatenate([np.zeros(n), [min_budget]]), np.concatenate([np.ones(n), [max_budget]]))
    return counts, offsets, a, LinearConstraint(a, lb, ub), bounds


def run_objective(name, weights, risk_scale, data, offsets, constraint, bounds, args, output):
    ratio = Fraction(0, 1)
    iterations = []
    final = None
    all_exact = True
    for iteration in range(args.max_iterations):
        c = np.concatenate([-weights.astype(float), [float(ratio) * len(CLASSES)]])
        started = time.monotonic()
        result = milp(c=c, integrality=np.ones(len(c)), bounds=bounds, constraints=constraint,
                      options={"time_limit": args.time_limit, "mip_rel_gap": 0.0})
        rec = certificate(result, time.monotonic() - started, args.time_limit)
        rec.update(iteration=iteration, input_ratio_numerator=ratio.numerator, input_ratio_denominator=ratio.denominator)
        all_exact &= result.status == 0
        if result.x is None:
            rec["verified_feasible_incumbent"] = False
            iterations.append(rec)
            save_json(output / f"{name}_iteration_{iteration}.json", rec)
            break
        rounded = np.rint(result.x).astype(int)
        assert np.max(np.abs(result.x - rounded)) < 1e-5
        k = int(rounded[-1])
        assert bounds.lb[-1] <= k <= bounds.ub[-1]
        selections = {}
        for i, cls in enumerate(CLASSES):
            ids = np.flatnonzero(rounded[offsets[i]:offsets[i + 1]]).tolist()
            assert feasible(ids, data[cls], k)
            selections[cls] = summary(ids, data[cls])
        numerator = int(np.dot(weights, rounded[:-1]))
        denominator = len(CLASSES) * k
        improvement_integer_numerator = numerator * ratio.denominator - ratio.numerator * denominator
        improvement = float(Fraction(improvement_integer_numerator, ratio.denominator))
        updated_ratio = Fraction(numerator, denominator)
        rec.update(verified_feasible_incumbent=True, budget_per_class=k, total_targets=denominator,
                   selected_objective_numerator=numerator, selected_objective_denominator=denominator,
                   selected_ratio=float(updated_ratio),
                   exact_improvement_numerator=improvement_integer_numerator,
                   exact_improvement_denominator=ratio.denominator, improvement=improvement,
                   objective_scaling=risk_scale)
        iterations.append(rec)
        save_json(output / f"{name}_iteration_{iteration}.json", rec)
        final = {
            "budget_per_class": k,
            "total_targets": denominator,
            "selected_indices": {cls: selections[cls]["selected_indices"] for cls in CLASSES},
            "classes": selections,
            "risk_lower_mean": sum(v["risk_lower_sum"] for v in selections.values()) / denominator,
            "high_risk_count": sum(v["high_risk_count"] for v in selections.values()),
            "high_risk_fraction": sum(v["high_risk_count"] for v in selections.values()) / denominator,
            "objective_value": float(updated_ratio) / risk_scale,
            "global_optimum_proved": False,
            "proof_type": "Exact integer-fraction Dinkelbach improvement with floating MILP solver certificates",
        }
        print(json.dumps({"objective": name, "iteration": iteration, "budget_per_class": k,
                          "objective_value": final["objective_value"], "optimal_linear_solve": result.status == 0,
                          "improvement": improvement}), flush=True)
        if result.status != 0:
            final["termination"] = "linear_milp_not_proved_optimal"
            break
        if improvement_integer_numerator <= 0:
            final["global_optimum_proved"] = bool(all_exact)
            final["termination"] = "nonpositive_exact_fraction_improvement"
            break
        ratio = updated_ratio
    else:
        if final:
            final["termination"] = "iteration_limit"
    if final is None:
        final = {"global_optimum_proved": False, "termination": "no_verified_feasible_incumbent"}
    final["iterations"] = iterations
    save_json(output / f"{name}.json", final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/QE_VisualState_CoverRisk_v1_20261013"))
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-budget", type=int)
    parser.add_argument("--max-budget", type=int)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--max-iterations", type=int, default=12)
    args = parser.parse_args()
    prepared = args.prepared or args.root / "prepared"
    output = args.output or args.root / "global_ratio"
    data = {cls: load_class(prepared / f"{cls}.npz") for cls in CLASSES}
    if args.min_budget is None:
        prior = json.loads((args.root / "optimization" / "solutions.json").read_text())
        assert prior["protocol"]["common_budget_proved_minimal"], "Minimum feasible common budget must be proved first"
        args.min_budget = prior["compact_budget_per_class"]
    max_budget = min(len(d["sample_ids"]) for d in data.values())
    if args.max_budget is not None:
        max_budget = min(max_budget, args.max_budget)
    assert 1 <= args.min_budget <= max_budget
    counts, offsets, a, constraint, bounds = make_constraints(data, args.min_budget, max_budget)
    raw_risk = np.concatenate([data[cls]["risk_lower"] * 12 for cls in CLASSES])
    rounding_error = float(np.max(np.abs(raw_risk - np.rint(raw_risk))))
    assert rounding_error < 1e-5, f"Risk scores cannot be safely integerized by 12; max error {rounding_error}"
    risk = np.rint(raw_risk).astype(np.int64)
    high = np.concatenate([data[cls]["high_risk"].astype(np.int64) for cls in CLASSES])
    protocol = {
        "prepared": str(prepared), "classes": list(CLASSES), "class_candidate_counts": dict(zip(CLASSES, counts)),
        "minimum_budget_per_class": args.min_budget, "maximum_budget_per_class": max_budget,
        "binary_target_variables": sum(counts), "integer_shared_budget_variables": 1,
        "reachable_state_constraints": sum(len(d["reachable"]) for d in data.values()),
        "equal_class_count_constraints": len(CLASSES), "constraint_matrix_shape": list(a.shape),
        "risk_integer_scaling": 12, "risk_rounding_error_max": rounding_error,
        "solver": "scipy.optimize.milp (HiGHS)", "method": "Dinkelbach integer fractional optimization",
        "time_limit_per_linear_milp": args.time_limit, "maximum_iterations_per_objective": args.max_iterations,
        "certificate_rule": "Every linear MILP is proved optimal and final rational improvement is nonpositive",
        "coverage_scope": "Every state observable in the prepared pool; unobserved dictionary states remain missing",
    }
    save_json(output / "protocol.json", protocol)
    risk_solution = run_objective("risk_mean", risk, 12, data, offsets, constraint, bounds, args, output)
    high_solution = run_objective("high_risk_fraction", high, 1, data, offsets, constraint, bounds, args, output)
    save_json(output / "solutions.json", {"protocol": protocol, "risk_mean": risk_solution, "high_risk_fraction": high_solution})


if __name__ == "__main__":
    main()
