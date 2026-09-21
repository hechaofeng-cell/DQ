#!/usr/bin/env python3
"""Shared components for coverage-utility aligned visual-state learning."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import models


EXCLUDED_COVERAGE_STATES = {"unknown", "other"}


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError("fields are required when writing an empty CSV")
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_values(value: str | list | None) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON list, got {type(parsed).__name__}")
        return [str(item) for item in parsed]
    return [value]


@dataclass(frozen=True)
class StateField:
    feature_id: str
    value_type: str
    values: tuple[str, ...]


class StateSchema:
    def __init__(self, schema: dict):
        definitions = schema["universal_features"] + schema.get("category_specific_features", [])
        self.version = schema["schema_version"]
        self.fields = tuple(
            StateField(
                definition["feature_id"],
                definition["value_type"],
                tuple(value for value in definition["possible_values"] if value != "unknown"),
            )
            for definition in definitions
        )
        self.enum_fields = tuple(field for field in self.fields if field.value_type == "enum")
        self.set_fields = tuple(field for field in self.fields if field.value_type == "set")
        if len(self.fields) != 23 or len(self.enum_fields) != 21 or len(self.set_fields) != 2:
            raise ValueError(
                f"dog_v3.1 requires 23 fields (21 enum, 2 set), got "
                f"{len(self.fields)} ({len(self.enum_fields)} enum, {len(self.set_fields)} set)"
            )
        self.enum_value_to_index = {
            field.feature_id: {value: index for index, value in enumerate(field.values)}
            for field in self.enum_fields
        }
        self.set_offsets = {}
        offset = 0
        for field in self.set_fields:
            self.set_offsets[field.feature_id] = (offset, offset + len(field.values))
            offset += len(field.values)
        self.set_target_size = offset

    @classmethod
    def from_path(cls, path: Path) -> "StateSchema":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def coverage_states(self) -> list[tuple[str, str]]:
        return [
            (field.feature_id, value)
            for field in self.fields
            for value in field.values
            if value not in EXCLUDED_COVERAGE_STATES
        ]

    def state_pairs(self, row: dict) -> set[tuple[str, str]]:
        pairs = set()
        for field in self.fields:
            for value in parse_values(row.get(field.feature_id)):
                if value not in EXCLUDED_COVERAGE_STATES:
                    pairs.add((field.feature_id, value))
        return pairs

    def encode(self, row: dict | None, is_dog: bool) -> dict[str, torch.Tensor | set]:
        enum_targets = torch.full((len(self.enum_fields),), -100, dtype=torch.long)
        set_targets = torch.zeros(self.set_target_size, dtype=torch.float32)
        set_masks = torch.zeros(len(self.set_fields), dtype=torch.float32)
        active_pairs: set[tuple[str, str]] = set()
        if not is_dog or row is None:
            return {
                "enum_targets": enum_targets,
                "set_targets": set_targets,
                "set_masks": set_masks,
                "active_pairs": active_pairs,
            }

        for index, field in enumerate(self.enum_fields):
            values = parse_values(row.get(field.feature_id))
            if len(values) != 1 or values[0] == "unknown":
                continue
            value = values[0]
            if value not in self.enum_value_to_index[field.feature_id]:
                raise ValueError(f"invalid {field.feature_id} value: {value}")
            enum_targets[index] = self.enum_value_to_index[field.feature_id][value]
            if value not in EXCLUDED_COVERAGE_STATES:
                active_pairs.add((field.feature_id, value))

        for index, field in enumerate(self.set_fields):
            values = parse_values(row.get(field.feature_id))
            if not values or "unknown" in values:
                continue
            if "none" in values and len(values) != 1:
                raise ValueError(f"{field.feature_id}=none must be exclusive")
            value_to_index = {value: idx for idx, value in enumerate(field.values)}
            start, _ = self.set_offsets[field.feature_id]
            for value in values:
                if value not in value_to_index:
                    raise ValueError(f"invalid {field.feature_id} value: {value}")
                set_targets[start + value_to_index[value]] = 1.0
                if value not in EXCLUDED_COVERAGE_STATES:
                    active_pairs.add((field.feature_id, value))
            set_masks[index] = 1.0

        return {
            "enum_targets": enum_targets,
            "set_targets": set_targets,
            "set_masks": set_masks,
            "active_pairs": active_pairs,
        }

    def mapping(self) -> dict:
        return {
            "schema_version": self.version,
            "enum_fields": [
                {"feature_id": field.feature_id, "values": list(field.values)}
                for field in self.enum_fields
            ],
            "set_fields": [
                {"feature_id": field.feature_id, "values": list(field.values)}
                for field in self.set_fields
            ],
        }


def feature_rows_by_id(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    if any(row.get("feature_status") != "ok" for row in rows):
        raise ValueError(f"feature file contains non-ok rows: {path}")
    mapping = {row["image_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate feature image_id in {path}")
    return mapping


def normalize_priority_scores(scores: list[float], percentile: float = 95.0) -> list[float]:
    positive = [score for score in scores if score > 0]
    if not positive:
        return [0.0] * len(scores)
    scale = float(np.percentile(positive, percentile))
    if scale <= 0:
        return [0.0] * len(scores)
    return [min(max(score / scale, 0.0), 1.0) for score in scores]


def row_priority(
    schema: StateSchema, feature_row: dict | None,
    state_priority: dict[tuple[str, str], float],
) -> float:
    if feature_row is None:
        return 0.0
    values = [state_priority.get(pair, 0.0) for pair in schema.state_pairs(feature_row)]
    return float(np.mean(values)) if values else 0.0


def class_balanced_state_weights(
    rows: list[dict], schema: StateSchema, features: dict[str, dict],
    state_priority: dict[tuple[str, str], float] | None,
    alpha: float, max_multiplier: float,
) -> list[float]:
    class_counts = Counter(int(row["label"]) for row in rows)
    dog_indices = [index for index, row in enumerate(rows) if row["class_name"] == "dog"]
    raw_scores = [
        row_priority(schema, features.get(rows[index]["sample_id"]), state_priority or {})
        for index in dog_indices
    ]
    normalized = normalize_priority_scores(raw_scores)
    multipliers = [min(1.0 + alpha * score, max_multiplier) for score in normalized]
    mean_multiplier = float(np.mean(multipliers)) if multipliers else 1.0
    dog_multiplier = {
        index: multiplier / mean_multiplier
        for index, multiplier in zip(dog_indices, multipliers)
    }
    return [
        dog_multiplier.get(index, 1.0) / class_counts[int(row["label"])]
        for index, row in enumerate(rows)
    ]


class StateAwareCropDataset(Dataset):
    def __init__(
        self, rows: list[dict], transform, schema: StateSchema,
        features: dict[str, dict], state_priority: dict[tuple[str, str], float] | None = None,
        state_weight_alpha: float = 0.0, state_weight_max: float = 3.0,
    ):
        self.rows = rows
        self.transform = transform
        self.schema = schema
        self.features = features
        self.state_priority = state_priority or {}
        self.state_weight_alpha = state_weight_alpha
        self.state_weight_max = state_weight_max

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        is_dog = row["class_name"] == "dog"
        feature_row = self.features.get(row["sample_id"]) if is_dog else None
        encoded = self.schema.encode(feature_row, is_dog)
        enum_weights = torch.ones(len(self.schema.enum_fields), dtype=torch.float32)
        set_weights = torch.ones(len(self.schema.set_fields), dtype=torch.float32)
        if is_dog and feature_row is not None and self.state_weight_alpha > 0:
            for field_index, field in enumerate(self.schema.enum_fields):
                values = parse_values(feature_row.get(field.feature_id))
                score = self.state_priority.get((field.feature_id, values[0]), 0.0) if len(values) == 1 else 0.0
                enum_weights[field_index] = min(1.0 + self.state_weight_alpha * score, self.state_weight_max)
            for field_index, field in enumerate(self.schema.set_fields):
                scores = [
                    self.state_priority.get((field.feature_id, value), 0.0)
                    for value in parse_values(feature_row.get(field.feature_id))
                    if value not in EXCLUDED_COVERAGE_STATES
                ]
                score = float(np.mean(scores)) if scores else 0.0
                set_weights[field_index] = min(1.0 + self.state_weight_alpha * score, self.state_weight_max)
        return {
            "image": tensor,
            "label": int(row["label"]),
            "index": index,
            "enum_targets": encoded["enum_targets"],
            "set_targets": encoded["set_targets"],
            "set_masks": encoded["set_masks"],
            "enum_weights": enum_weights,
            "set_weights": set_weights,
        }


def build_feature_backbone(model_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        dimension = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        dimension = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif model_name == "maxvit_t":
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT if pretrained else None)
        dimension = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        dimension = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"unsupported CUA-VSL model: {model_name}")
    return model, dimension


class VisualStateMultiTaskModel(nn.Module):
    def __init__(self, model_name: str, schema: StateSchema, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.schema = schema
        self.backbone, dimension = build_feature_backbone(model_name, pretrained)
        self.class_head = nn.Linear(dimension, num_classes)
        self.enum_heads = nn.ModuleList(nn.Linear(dimension, len(field.values)) for field in schema.enum_fields)
        self.set_heads = nn.ModuleList(nn.Linear(dimension, len(field.values)) for field in schema.set_fields)

    def forward(self, images: torch.Tensor, include_states: bool = True) -> dict:
        features = self.backbone(images)
        output = {"class_logits": self.class_head(features)}
        if include_states:
            output["enum_logits"] = [head(features) for head in self.enum_heads]
            output["set_logits"] = [head(features) for head in self.set_heads]
        return output


def visual_state_loss(output: dict, batch: dict, schema: StateSchema) -> tuple[torch.Tensor, dict[str, float]]:
    device = output["class_logits"].device
    zero = output["class_logits"].sum() * 0.0
    enum_numerator = zero
    enum_denominator = torch.zeros((), device=device)
    for index, logits in enumerate(output.get("enum_logits", [])):
        targets = batch["enum_targets"][:, index].to(device)
        valid = targets != -100
        if valid.any():
            losses = F.cross_entropy(logits[valid], targets[valid], reduction="none")
            weights = batch["enum_weights"][:, index].to(device)[valid]
            enum_numerator = enum_numerator + (losses * weights).sum()
            enum_denominator = enum_denominator + weights.sum()
    enum_loss = enum_numerator / enum_denominator.clamp_min(1.0)

    set_numerator = zero
    set_denominator = torch.zeros((), device=device)
    for index, (field, logits) in enumerate(zip(schema.set_fields, output.get("set_logits", []))):
        start, end = schema.set_offsets[field.feature_id]
        targets = batch["set_targets"][:, start:end].to(device)
        valid = batch["set_masks"][:, index].to(device) > 0
        if valid.any():
            losses = F.binary_cross_entropy_with_logits(logits[valid], targets[valid], reduction="none").mean(1)
            weights = batch["set_weights"][:, index].to(device)[valid]
            set_numerator = set_numerator + (losses * weights).sum()
            set_denominator = set_denominator + weights.sum()
    set_loss = set_numerator / set_denominator.clamp_min(1.0)
    total = enum_loss + set_loss
    return total, {
        "enum_loss": float(enum_loss.detach().cpu()),
        "set_loss": float(set_loss.detach().cpu()),
        "valid_enum_weight": float(enum_denominator.detach().cpu()),
        "valid_set_weight": float(set_denominator.detach().cpu()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_state_priority(path: Path, column: str) -> dict[tuple[str, str], float]:
    rows = read_csv(path)
    return {(row["feature_id"], row["state"]): float(row[column]) for row in rows}


def state_counts(rows: list[dict], schema: StateSchema) -> Counter:
    counts = Counter()
    for row in rows:
        counts.update(schema.state_pairs(row))
    return counts


def coverage_index(counts: Counter, schema: StateSchema, cap: int) -> float:
    states = schema.coverage_states()
    return sum(min(counts[state] / cap, 1.0) for state in states) / len(states)


def validate_no_split_overlap(groups: dict[str, list[dict]]) -> None:
    names = list(groups)
    ids = {name: {row["image_id"] for row in rows} for name, rows in groups.items()}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise ValueError(f"image overlap between {left} and {right}: {sorted(overlap)[:3]}")


def finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite: {value}")
    return value
