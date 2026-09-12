"""Candidate formulas and temperature scaling on isolated calibration data."""

import math
from collections import Counter


ACCEPTED = {"confirmed", "corrected"}


def probabilities(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not logits or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                             and math.isfinite(x) for x in logits):
        raise ValueError("Invalid logits")
    maximum = max(logits)
    values = [math.exp((x - maximum) / temperature) for x in logits]
    total = sum(values)
    return [x / total for x in values]


def score_a(predictions, truth, models):
    if not models or len(set(models)) != len(models):
        raise ValueError("A fixed nonempty unique model panel is required")
    if any(not predictions.get(m) for m in models):
        return None
    return sum(predictions[m] != truth for m in models) / len(models)


def score_b(logits, truth_index, models, temperatures):
    if not models or len(set(models)) != len(models):
        raise ValueError("A fixed nonempty unique model panel is required")
    if any(m not in logits or m not in temperatures for m in models):
        return None
    return 1 - sum(probabilities(logits[m], temperatures[m])[truth_index]
                   for m in models) / len(models)


def calibration_metrics(examples, temperature=1.0, bins=10):
    if not examples:
        return {"n": 0, "nll": None, "brier": None, "ece": None}
    nll = brier = 0.0
    buckets = [[] for _ in range(bins)]
    for logits, truth in examples:
        p = probabilities(logits, temperature)
        # Log-sum-exp keeps NLL valid even when the true probability underflows.
        shifted = [(v - max(logits)) / temperature for v in logits]
        nll += math.log(sum(math.exp(v) for v in shifted)) - shifted[truth]
        brier += sum((v - int(j == truth)) ** 2 for j, v in enumerate(p))
        predicted = max(range(len(p)), key=p.__getitem__)
        confidence = p[predicted]
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, predicted == truth))
    ece = sum(abs(sum(c for c, _ in b) - sum(ok for _, ok in b)) for b in buckets)
    return {"n": len(examples), "nll": nll / len(examples),
            "brier": brier / len(examples), "ece": ece / len(examples)}


def fit_temperature(examples, bounds):
    low, high = map(math.log, bounds)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = high - ratio * (high - low), low + ratio * (high - low)
    objective = lambda log_t: calibration_metrics(examples, math.exp(log_t))["nll"]
    fc, fd = objective(c), objective(d)
    for _ in range(70):
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - ratio * (high - low)
            fc = objective(c)
        else:
            low, c, fc = c, d, fd
            d = low + ratio * (high - low)
            fd = objective(d)
    return math.exp((low + high) / 2)


def calibrate(rows, predictions, config):
    policy = config["calibration"]
    classes = config["class_order"]
    models = config["scoring_models"]
    result = {"status": "pending_validation", "models": {}, "reasons": []}
    for model in models:
        parts = {}
        for part in ("calibration_fit", "calibration_check"):
            # Entire same-source groups are assigned together; count once per group.
            selected = {r["source_group_id"]: r for r in rows if r["split"] == part
                        and r["review_status"] in ACCEPTED and not r["exclusion_reasons"]}
            selected = list(selected.values())
            coverage = Counter(r["reference_class_id"] for r in selected)
            examples = [(predictions[r["original_id"]][model]["logits"],
                         classes.index(r["reference_class_id"])) for r in selected
                        if predictions[r["original_id"]].get(model, {}).get("valid")]
            errors = sum(max(range(len(z)), key=z.__getitem__) != y for z, y in examples)
            parts[part] = {"n": len(examples), "errors": errors, "coverage": dict(coverage)}
            if (len(examples) < policy["min_per_part"] or errors < policy["min_errors_per_model"]
                    or any(coverage[c] < policy["min_per_class"] for c in classes)):
                result["reasons"].append(f"{model}:{part}:insufficient_reviewed_samples_or_errors")
            parts[part]["examples"] = examples
        details = {p: {k: v for k, v in info.items() if k != "examples"}
                   for p, info in parts.items()}
        if not any(reason.startswith(model + ":") for reason in result["reasons"]):
            temp = fit_temperature(parts["calibration_fit"]["examples"], policy["temperature_bounds"])
            details["temperature"] = temp
            for part, info in parts.items():
                details[part]["before"] = calibration_metrics(info["examples"])
                details[part]["after"] = calibration_metrics(info["examples"], temp)
            lo, hi = policy["temperature_bounds"]
            check = details["calibration_check"]
            if temp <= lo * 1.001 or temp >= hi / 1.001:
                result["reasons"].append(f"{model}:temperature_at_search_boundary")
            if any(check["after"][metric] > check["before"][metric] + 1e-10 for metric in ("nll", "brier")):
                result["reasons"].append(f"{model}:held_out_calibration_did_not_improve")
        result["models"][model] = details
    if not result["reasons"]:
        result["status"] = "calibration_checked_exploratory"
    return result
