"""Prompt construction and deterministic sampling for schema generation."""

import json


def representative_rows(rows, limit=12):
    """Select stable rows while retaining different target sizes when available."""
    ordered = sorted(rows, key=lambda row: row["image_id"])
    if len(ordered) <= limit:
        return ordered
    # Deterministic evenly spaced selection avoids depending on filesystem order.
    indices = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    return [ordered[index] for index in indices]


def build_schema_prompt(template, category_summary, policy):
    return template + "\n类别聚合结果：\n" + json.dumps(category_summary, ensure_ascii=False, sort_keys=True) + \
        "\n固定元规则：\n" + json.dumps(policy, ensure_ascii=False, sort_keys=True)
