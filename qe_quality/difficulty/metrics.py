"""Tie-aware rankings and descriptive held-out model checks."""

import math
from collections import Counter


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2 + 1
        start = end
    return result


def spearman(left, right):
    if len(left) != len(right):
        raise ValueError("Paired ranks require equal lengths")
    if len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    aa, bb = sum((x - ma) ** 2 for x in a), sum((x - mb) ** 2 for x in b)
    if not aa or not bb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(aa * bb)


def wilson(errors, n):
    if not n:
        return None
    z = 1.959963984540054
    p = errors / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - half), min(1.0, center + half)]


def auc(scores, errors):
    positives = sum(errors)
    negatives = len(errors) - positives
    if not positives or not negatives:
        return None
    return (sum(r for r, error in zip(ranks(scores), errors) if error)
            - positives * (positives + 1) / 2) / (positives * negatives)


def discrimination(values):
    counts = Counter(values)
    n = len(values)
    return {"n": n, "unique_scores": len(counts),
            "largest_tie_count": max(counts.values(), default=0),
            "largest_tie_fraction": max(counts.values(), default=0) / n if n else None,
            "tied_pair_fraction": sum(k * (k - 1) for k in counts.values()) / (n * (n - 1)) if n > 1 else None,
            "distribution": {str(s): k for s, k in sorted(counts.items())} if len(counts) <= 20 else None}


def independent_check(rows, field, model, edges):
    eligible = [r for r in rows if r[field] is not None and r.get(model + "_correct") is not None]
    # Byte/identity duplicates are not independent Bernoulli observations.
    groups = {}
    for row in sorted(eligible, key=lambda r: r["original_id"]):
        groups.setdefault(row["source_group_id"], row)
    unique = list(groups.values())
    bands = []
    for i in range(len(edges) - 1):
        low, high = edges[i:i + 2]
        selected = [r for r in unique if low <= r[field] and
                    (r[field] < high or i == len(edges) - 2 and r[field] <= high)]
        errors = sum(not r[model + "_correct"] for r in selected)
        bands.append({"lower": low, "upper": high, "upper_inclusive": i == len(edges) - 2,
                      "n_groups": len(selected), "errors": errors,
                      "error_rate": errors / len(selected) if selected else None,
                      "wilson_95": wilson(errors, len(selected))})
    return {"n_images": len(eligible), "n_groups": len(unique),
            "missing_validator_predictions": sum(r[field] is not None and r.get(model + "_correct") is None for r in rows),
            "error_auc": auc([r[field] for r in unique], [not r[model + "_correct"] for r in unique]),
            "bands": bands,
            "uncertainty": "Wilson 95% on unique source groups; near duplicates and training overlap remain uncontrolled"}


def panel_stability(rows, models, field="score_a"):
    results = {}
    complete = [r for r in rows if r[field] is not None and all(r.get(m + "_correct") is not None for m in models)]
    for removed in models:
        remaining = [m for m in models if m != removed]
        if not remaining:
            continue
        base = [r[field] for r in complete]
        alternative = [sum(not r[m + "_correct"] for m in remaining) / len(remaining) for r in complete]
        left, right = ranks(base), ranks(alternative)
        n = len(base)
        results["remove_" + removed] = {"n": n, "spearman": spearman(base, alternative),
            "mean_absolute_rank_shift": sum(abs(a - b) for a, b in zip(left, right)) / n if n else None,
            "max_absolute_rank_shift": max((abs(a - b) for a, b in zip(left, right)), default=None),
            "changed_score_count": sum(a != b for a, b in zip(base, alternative))}
    return results
