#!/usr/bin/env python3
"""Audit exact and near duplicates in the multiclass candidate pool."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/multiclass_object_pool_v1_20260928/manifests/candidate_pool.csv"
TRAIN = Path("/home/hcf/project/QE/artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv")
FEATURE_ROOT = ROOT / "artifacts/multiclass_object_selection_benchmark_20260928/runs/convnext_tiny/20260916"
DEFAULT_OUTPUT = ROOT / "artifacts/near_duplicate_audit_20260929"
PHASH_STRICT = 4
PHASH_RELAXED = 8
COSINE_STRICT = 0.980
COSINE_RELAXED = 0.995


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> np.uint64:
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.float32)
    low = dctn(values, type=2, norm="ortho")[:8, :8]
    threshold = float(np.median(low.reshape(-1)[1:]))
    bits = (low >= threshold).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def hamming_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lookup = np.asarray([int(index).bit_count() for index in range(256)], dtype=np.uint8)
    xor = np.bitwise_xor(left[:, None], right[None, :])
    octets = xor.view(np.uint8).reshape(xor.shape + (8,))
    return lookup[octets].sum(axis=-1, dtype=np.uint8)


def align_features(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    stored = np.load(path, allow_pickle=True)
    mapping = {str(sample_id): index for index, sample_id in enumerate(stored["sample_ids"])}
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in mapping]
    if missing:
        raise ValueError(f"feature cache misses {len(missing)} sample IDs")
    values = stored["features"][[mapping[row["sample_id"]] for row in rows]].astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def status(distance: int, cosine: float, exact: bool, same_image: bool = False) -> str:
    if exact:
        return "exact_crop"
    if same_image:
        return "same_source_image"
    if distance <= PHASH_STRICT and cosine >= COSINE_STRICT:
        return "near_duplicate_strict"
    if distance <= PHASH_RELAXED and cosine >= COSINE_RELAXED:
        return "near_duplicate_corroborated"
    return "not_flagged"


def pair_row(
    left: dict[str, str], right: dict[str, str], distance: int, cosine: float,
    exact: bool, scope: str, same_image: bool = False,
) -> dict[str, object]:
    return {
        "scope": scope,
        "left_sample_id": left["sample_id"],
        "left_class": left["class_name"],
        "left_path": left["crop_path"],
        "right_sample_id": right["sample_id"],
        "right_class": right["class_name"],
        "right_path": right["crop_path"],
        "phash_distance": int(distance),
        "embedding_cosine": float(cosine),
        "exact_crop_sha256": int(exact),
        "same_source_image": int(same_image),
        "decision": status(distance, cosine, exact, same_image),
    }


def make_contact_sheet(rows: list[dict[str, object]], output: Path, title: str, limit: int = 12) -> None:
    selected = rows[:limit]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(8.5, 2.8 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected):
        for column, side in enumerate(("left", "right")):
            axis = axes[row_index, column]
            path = Path(str(row[f"{side}_path"]))
            with Image.open(path) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            axis.axis("off")
            axis.set_title(
                f"{row[f'{side}_class']} | {row[f'{side}_sample_id']}",
                fontsize=8,
            )
        axes[row_index, 0].set_ylabel(
            f"{row['decision']}\ndH={row['phash_distance']}, cos={row['embedding_cosine']:.3f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(POOL)
    train = read_csv(TRAIN)
    if len(candidates) != 2000 or len(train) != 1500:
        raise ValueError(f"unexpected manifest sizes: {len(candidates)} candidates, {len(train)} train")

    for row in candidates + train:
        path = Path(row["crop_path"])
        if not path.is_file():
            raise FileNotFoundError(path)

    candidate_sha = np.asarray([row.get("crop_sha256") or sha256(Path(row["crop_path"])) for row in candidates])
    train_sha = np.asarray([sha256(Path(row["crop_path"])) for row in train])
    candidate_hash = np.asarray([phash(Path(row["crop_path"])) for row in candidates], dtype=np.uint64)
    train_hash = np.asarray([phash(Path(row["crop_path"])) for row in train], dtype=np.uint64)
    candidate_features = align_features(FEATURE_ROOT / "candidate_outputs.npz", candidates)
    train_features = align_features(FEATURE_ROOT / "train_features.npz", train)

    cc_hamming = hamming_matrix(candidate_hash, candidate_hash)
    ct_hamming = hamming_matrix(candidate_hash, train_hash)
    cc_cosine = candidate_features @ candidate_features.T
    ct_cosine = candidate_features @ train_features.T

    internal: list[dict[str, object]] = []
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left, right = candidates[left_index], candidates[right_index]
            exact = candidate_sha[left_index] == candidate_sha[right_index]
            same_image = bool(left.get("image_sha256") and left.get("image_sha256") == right.get("image_sha256"))
            decision = status(
                int(cc_hamming[left_index, right_index]),
                float(cc_cosine[left_index, right_index]), exact, same_image,
            )
            if decision != "not_flagged":
                internal.append(pair_row(
                    left, right, int(cc_hamming[left_index, right_index]),
                    float(cc_cosine[left_index, right_index]), exact, "candidate_internal", same_image,
                ))

    cross: list[dict[str, object]] = []
    for left_index, left in enumerate(candidates):
        likely = np.where(
            (candidate_sha[left_index] == train_sha)
            | ((ct_hamming[left_index] <= PHASH_STRICT) & (ct_cosine[left_index] >= COSINE_STRICT))
            | ((ct_hamming[left_index] <= PHASH_RELAXED) & (ct_cosine[left_index] >= COSINE_RELAXED))
        )[0]
        for right_index in likely:
            right = train[int(right_index)]
            cross.append(pair_row(
                left, right, int(ct_hamming[left_index, right_index]),
                float(ct_cosine[left_index, right_index]),
                bool(candidate_sha[left_index] == train_sha[right_index]), "candidate_vs_train",
            ))

    ordering = {
        "exact_crop": 0, "same_source_image": 1,
        "near_duplicate_strict": 2, "near_duplicate_corroborated": 3,
    }
    sort_key = lambda row: (
        ordering[str(row["decision"])], int(row["phash_distance"]), -float(row["embedding_cosine"])
    )
    internal.sort(key=sort_key)
    cross.sort(key=sort_key)
    fields = list(pair_row(candidates[0], candidates[1], 0, 1.0, False, "x"))
    write_csv(output / "tables/candidate_internal_flagged_pairs.csv", internal, fields)
    write_csv(output / "tables/candidate_train_flagged_pairs.csv", cross, fields)

    internal_samples = {str(row["left_sample_id"]) for row in internal} | {str(row["right_sample_id"]) for row in internal}
    cross_samples = {str(row["left_sample_id"]) for row in cross}
    summary = {
        "audit": "multiclass-near-duplicate-v1",
        "created_at": now(),
        "candidate_count": len(candidates),
        "train_count": len(train),
        "thresholds_fixed_before_inspection": {
            "strict": f"pHash <= {PHASH_STRICT} and cosine >= {COSINE_STRICT}",
            "corroborated": f"pHash <= {PHASH_RELAXED} and cosine >= {COSINE_RELAXED}",
        },
        "candidate_internal": {
            "flagged_pairs": len(internal),
            "affected_candidates": len(internal_samples),
            "decision_counts": dict(Counter(str(row["decision"]) for row in internal)),
        },
        "candidate_vs_train": {
            "flagged_pairs": len(cross),
            "affected_candidates": len(cross_samples),
            "decision_counts": dict(Counter(str(row["decision"]) for row in cross)),
        },
    }
    summary["candidate_train_leakage_detected"] = bool(cross)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_contact_sheet(internal, output / "figures/candidate_internal_pairs.png", "候选池内部疑似重复")
    make_contact_sheet(cross, output / "figures/candidate_train_pairs.png", "候选池与训练集疑似重复")

    report = f"""# 多类别候选池重复与近重复审计

审计对象为固定候选池 {len(candidates)} 张与固定训练集 {len(train)} 张目标裁剪图。判定同时使用裁剪图 SHA-256、64-bit pHash 和固定 ConvNeXt-Tiny 表征余弦相似度；阈值在查看配对结果前固定。

## 阈值

- 精确重复：裁剪图 SHA-256 相同；
- 严格近重复：pHash 距离不超过 {PHASH_STRICT} 且表征余弦相似度不低于 {COSINE_STRICT:.3f}；
- 交叉佐证近重复：pHash 距离不超过 {PHASH_RELAXED} 且表征余弦相似度不低于 {COSINE_RELAXED:.3f}；
- 候选池内部另报告同一原始图像产生的多个目标裁剪。

## 结果

- 候选池内部：{len(internal)} 对被标记，涉及 {len(internal_samples)}/{len(candidates)} 个候选样本；
- 候选池与训练集之间：{len(cross)} 对被标记，涉及 {len(cross_samples)}/{len(candidates)} 个候选样本；
- 训练泄漏判定：`{bool(cross)}`。

被标记配对与可视化保存在 `tables/` 和 `figures/`。pHash 与单模型表征是保守筛查工具，不等价于人工身份判定；若存在交叉集标记样本，主分析应剔除后复算或进行敏感性分析。
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
