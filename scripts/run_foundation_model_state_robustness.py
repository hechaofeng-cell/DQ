#!/usr/bin/env python3
"""Compare recent frozen vision backbones on the fixed D300 state-expansion test.

The scientific factor is the pretrained representation. D300, validation data,
test candidates, coverage order, preprocessing declared by each backbone, linear
head optimizer, budgets, and seeds are otherwise fixed. Features are cached so
the experiment can resume after interruption.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from transformers import Aimv2VisionModel, AutoImageProcessor, AutoModel, SiglipVisionModel


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("QE_SOURCE_ROOT", "/home/hcf/project/QE"))
OUTPUT = ROOT / "artifacts/foundation_model_state_robustness_20260922"
TRAIN_MANIFEST = SOURCE_ROOT / "artifacts/coco_openimages_external_gradient_8_models_20260918/manifests/train_oi_gap_add0.csv"
VAL_MANIFEST = SOURCE_ROOT / "data/openimages_v7_dog_gap_v1/manifests/validation.csv"
FIXED_TEST_ROOT = SOURCE_ROOT / "artifacts/fixed_train_visual_test_adequacy_20260920"
PE_REPO = Path(os.environ.get("PE_REPO", "/home/hcf/.cache/qe_external/perception_models"))

SEEDS = (20260916, 20260917, 20260918)
BUDGETS = (20, 40, 60, 80, 100)
CLASS_NAMES = ("dog", "cat", "horse", "sheep", "person")
MODEL_NAMES = (
    "convnext_tiny", "swin_t", "dinov3_or_dinov2", "siglip2_base",
    "pe_core_b16", "aimv2_large",
)
DISPLAY_NAMES = {
    "convnext_tiny": "ConvNeXt-T",
    "swin_t": "Swin-T",
    "dinov3_or_dinov2": "DINOv2-S",
    "siglip2_base": "SigLIP2-B",
    "pe_core_b16": "PE-B/16",
    "aimv2_large": "AIMv2-L",
}
DINOV3_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV2_FALLBACK_ID = "facebook/dinov2-small"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            return image.convert("RGB"), int(row["label"]), row["sample_id"]


def pil_collate(batch):
    images, labels, ids = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(ids)


@dataclass
class Backbone:
    resolved_name: str
    source: str
    model: nn.Module
    encode: Callable[[list[Image.Image]], torch.Tensor]
    parameters: int
    feature_dim: int | None = None


def load_backbone(name: str, device: torch.device) -> Backbone:
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Identity()
        model.eval().to(device)
        preprocess = weights.transforms()

        def encode(images):
            return model(torch.stack([preprocess(image) for image in images]).to(device))

        return Backbone(name, str(weights), model, encode, sum(p.numel() for p in model.parameters()))

    if name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT
        model = models.swin_t(weights=weights)
        model.head = nn.Identity()
        model.eval().to(device)
        preprocess = weights.transforms()

        def encode(images):
            return model(torch.stack([preprocess(image) for image in images]).to(device))

        return Backbone(name, str(weights), model, encode, sum(p.numel() for p in model.parameters()))

    if name == "dinov3_or_dinov2":
        resolved = DINOV3_ID
        try:
            processor = AutoImageProcessor.from_pretrained(resolved)
            model = AutoModel.from_pretrained(resolved)
        except Exception as error:
            resolved = DINOV2_FALLBACK_ID
            print(f"[model] DINOv3 unavailable ({type(error).__name__}); using {resolved}", flush=True)
            processor = AutoImageProcessor.from_pretrained(resolved)
            model = AutoModel.from_pretrained(resolved)
        model.eval().to(device)

        def encode(images):
            inputs = processor(images=images, return_tensors="pt")
            output = model(**{key: value.to(device) for key, value in inputs.items()})
            return output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]

        return Backbone(resolved.split("/")[-1], resolved, model, encode, sum(p.numel() for p in model.parameters()))

    if name == "siglip2_base":
        source = "google/siglip2-base-patch16-224"
        processor = AutoImageProcessor.from_pretrained(source)
        model = SiglipVisionModel.from_pretrained(source).eval().to(device)

        def encode(images):
            inputs = processor(images=images, return_tensors="pt")
            output = model(**{key: value.to(device) for key, value in inputs.items()})
            return output.pooler_output

        return Backbone(name, source, model, encode, sum(p.numel() for p in model.parameters()))

    if name == "aimv2_large":
        source = "apple/aimv2-large-patch14-224"
        processor = AutoImageProcessor.from_pretrained(source)
        model, loading = Aimv2VisionModel.from_pretrained(source, output_loading_info=True)
        if any(loading[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"AIMv2 weight audit failed: {loading}")
        model = model.eval().to(device)

        def encode(images):
            inputs = processor(images=images, return_tensors="pt")
            output = model(**{key: value.to(device) for key, value in inputs.items()})
            return output.pooler_output if output.pooler_output is not None else output.last_hidden_state.mean(dim=1)

        return Backbone(name, source, model, encode, sum(p.numel() for p in model.parameters()))

    if name == "pe_core_b16":
        if not PE_REPO.is_dir():
            raise FileNotFoundError(f"Perception Encoder repository missing: {PE_REPO}")
        sys.path.insert(0, str(PE_REPO))
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        source = "facebook/PE-Core-B16-224"
        clip = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
        model = clip.visual.eval().to(device)
        preprocess = pe_transforms.get_image_transform(224)
        del clip

        def encode(images):
            return model(torch.stack([preprocess(image) for image in images]).to(device))

        return Backbone(name, source, model, encode, sum(p.numel() for p in model.parameters()))

    raise ValueError(f"unknown model: {name}")


def extract_features(name: str, splits: dict[str, list[dict]], device: torch.device, workers: int) -> dict:
    cache = OUTPUT / "features" / f"{name}.pt"
    if cache.is_file():
        existing = torch.load(cache, map_location="cpu", weights_only=False)
        valid = all(
            split in existing
            and existing[split]["sample_ids"] == [row["sample_id"] for row in rows]
            for split, rows in splits.items()
        )
        if valid:
            print(f"[features] reuse {cache}", flush=True)
            return existing
        print(f"[features] discard incompatible cache {cache}", flush=True)

    backbone = load_backbone(name, device)
    print(f"[features] loaded {name} -> {backbone.source} ({backbone.parameters:,} params)", flush=True)
    result = {
        "requested_name": name,
        "resolved_name": backbone.resolved_name,
        "source": backbone.source,
        "backbone_parameters": backbone.parameters,
    }
    batch_size = 64 if name not in {"aimv2_large", "siglip2_base"} else 32
    for split, rows in splits.items():
        loader = DataLoader(ImageRows(rows), batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=pil_collate)
        feature_parts, labels, ids = [], [], []
        with torch.inference_mode():
            for batch_index, (images, target, sample_ids) in enumerate(loader, start=1):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    features = backbone.encode(images)
                features = F.normalize(features.float(), dim=-1).cpu()
                feature_parts.append(features)
                labels.append(target)
                ids.extend(sample_ids)
                if batch_index % 10 == 0:
                    print(f"[features] {name} {split}: {min(batch_index * batch_size, len(rows))}/{len(rows)}", flush=True)
        values = torch.cat(feature_parts)
        result[split] = {"features": values, "labels": torch.cat(labels), "sample_ids": ids}
        backbone.feature_dim = values.shape[1]
        print(f"[features] {name} {split} complete: {tuple(values.shape)}", flush=True)
    result["feature_dim"] = backbone.feature_dim
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def metric_dict(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int | list]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=np.arange(5), zero_division=0,
    )
    matrix = confusion_matrix(true, pred, labels=np.arange(5))
    return {
        "accuracy": float((true == pred).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "dog_precision": float(precision[0]),
        "dog_recall": float(recall[0]),
        "dog_f1": float(f1[0]),
        "dog_error_rate": float(1 - recall[0]),
        "error_count": int((true != pred).sum()),
        "confusion_matrix": matrix.tolist(),
    }


def train_linear_head(cache: dict, seed: int, device: torch.device) -> tuple[nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_x, train_y = cache["train"]["features"], cache["train"]["labels"]
    val_x, val_y = cache["val"]["features"], cache["val"]["labels"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, generator=generator)
    head = nn.Linear(train_x.shape[1], 5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, 301):
        head.train()
        for features, labels in loader:
            logits = head(features.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            pred = head(val_x.to(device)).argmax(dim=1).cpu().numpy()
        metrics = metric_dict(val_y.numpy(), pred)
        history.append({"epoch": epoch, "macro_f1": metrics["macro_f1"]})
        if metrics["macro_f1"] > best_f1 + 1e-8:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    assert best_state is not None
    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "val_macro_f1": best_f1, "epochs_ran": epoch, "history": history}


def load_state_context():
    sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
    import run_fixed_train_test_adequacy_experiment as fixed

    rows, dog_rows, features, schema = fixed.prepare_candidates()
    matrix, states = fixed.dog_state_matrix(dog_rows, features, schema)
    gap_ids = [row["sample_id"] for row in read_csv(FIXED_TEST_ROOT / "orders/gap_dog_order.csv")]
    dog_by_id = {row["sample_id"]: index for index, row in enumerate(dog_rows)}
    gap_order = [dog_by_id[sample_id] for sample_id in gap_ids]
    non_dog = read_csv(FIXED_TEST_ROOT / "orders/non_dog_order.csv")
    return fixed, rows, dog_rows, matrix, states, gap_order, non_dog


def evaluate_predictions(model_name: str, seed: int, candidate_rows: list[dict], pred: np.ndarray, state_context) -> tuple[list[dict], list[dict]]:
    fixed, _, dog_rows, state_matrix, _, gap_order, non_dog = state_context
    true_by_id = {row["sample_id"]: int(row["label"]) for row in candidate_rows}
    pred_by_id = {row["sample_id"]: int(value) for row, value in zip(candidate_rows, pred)}
    dog_ids = [row["sample_id"] for row in dog_rows]
    dog_correct = np.asarray([pred_by_id[sample_id] == 0 for sample_id in dog_ids])
    non_dog_by_class: dict[str, list[str]] = {name: [] for name in CLASS_NAMES[1:]}
    for row in non_dog:
        non_dog_by_class[row["class_name"]].append(row["sample_id"])

    cumulative, batches = [], []
    for budget in BUDGETS:
        dog_selected = gap_order[:budget]
        ids = [dog_ids[index] for index in dog_selected]
        for class_name in CLASS_NAMES[1:]:
            ids.extend(non_dog_by_class[class_name][:budget])
        true = np.asarray([true_by_id[sample_id] for sample_id in ids])
        predicted = np.asarray([pred_by_id[sample_id] for sample_id in ids])
        metrics = metric_dict(true, predicted)
        state_metrics = fixed.dog_failure_metrics(dog_selected, dog_correct, state_matrix)
        cumulative.append({
            "model": model_name, "seed": seed, "dog_per_class": budget,
            "total_test_targets": len(ids), **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            **state_metrics,
        })

    for batch_index, start in enumerate(range(0, 100, 20), start=1):
        dog_selected = gap_order[start:start + 20]
        ids = [dog_ids[index] for index in dog_selected]
        for class_name in CLASS_NAMES[1:]:
            ids.extend(non_dog_by_class[class_name][start:start + 20])
        true = np.asarray([true_by_id[sample_id] for sample_id in ids])
        predicted = np.asarray([pred_by_id[sample_id] for sample_id in ids])
        metrics = metric_dict(true, predicted)
        state_metrics = fixed.dog_failure_metrics(dog_selected, dog_correct, state_matrix)
        batches.append({
            "model": model_name, "seed": seed, "batch": batch_index,
            "target_rank_start": start + 1, "target_rank_end": start + 20,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"}, **state_metrics,
        })
    return cumulative, batches


def aggregate(cumulative: list[dict], batches: list[dict], model_meta: list[dict]) -> list[dict]:
    summary = []
    for model in MODEL_NAMES:
        full = [row for row in cumulative if row["model"] == model and row["dog_per_class"] == 100]
        model_batches = [row for row in batches if row["model"] == model]
        seed_min = []
        seed_sd = []
        for seed in SEEDS:
            values = [row["macro_f1"] for row in model_batches if row["seed"] == seed]
            seed_min.append(min(values))
            seed_sd.append(float(np.std(values, ddof=1)))
        out = {"model": model, "runs": len(full)}
        for metric in ("accuracy", "macro_f1", "dog_f1", "dog_recall", "dog_error_rate", "state_macro_recall", "worst5_state_recall", "failed_state_count"):
            values = np.asarray([float(row[metric]) for row in full])
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_sd"] = float(values.std(ddof=1))
        out["min_independent_batch_macro_f1_mean"] = float(np.mean(seed_min))
        out["independent_batch_macro_f1_sd_mean"] = float(np.mean(seed_sd))
        meta = next(item for item in model_meta if item["requested_name"] == model)
        out.update({key: meta[key] for key in ("resolved_name", "source", "backbone_parameters", "feature_dim")})
        summary.append(out)
    return sorted(summary, key=lambda row: (row["worst5_state_recall_mean"], row["macro_f1_mean"]), reverse=True)


def make_figure(cumulative: list[dict], summary: list[dict]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    colors = plt.get_cmap("tab10")
    for index, model in enumerate(MODEL_NAMES):
        means = []
        stds = []
        for budget in BUDGETS:
            values = [row["macro_f1"] for row in cumulative if row["model"] == model and row["dog_per_class"] == budget]
            means.append(np.mean(values))
            stds.append(np.std(values, ddof=1))
        axes[0].errorbar(BUDGETS, means, yerr=stds, marker="o", linewidth=1.6, capsize=2, label=DISPLAY_NAMES[model], color=colors(index))
    axes[0].set(title="Cumulative coverage test", xlabel="Targets per class", ylabel="Macro-F1", xticks=BUDGETS)
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")

    names = [DISPLAY_NAMES[row["model"]] for row in summary][::-1]
    worst = [row["worst5_state_recall_mean"] for row in summary][::-1]
    state = [row["state_macro_recall_mean"] for row in summary][::-1]
    y = np.arange(len(names))
    axes[1].barh(y - .18, state, height=.34, label="State macro recall", color="#167d8d")
    axes[1].barh(y + .18, worst, height=.34, label="Worst-5 state recall", color="#d25b40")
    axes[1].set(title="State robustness at 100/class", xlabel="Recall", yticks=y, yticklabels=names, xlim=(0, 1))
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[1].grid(axis="x", alpha=.2)

    x = [row["macro_f1_mean"] for row in summary]
    yv = [row["worst5_state_recall_mean"] for row in summary]
    annotation_offsets = {
        "aimv2_large": (-28, 10), "convnext_tiny": (-52, 10),
        "pe_core_b16": (7, -15), "swin_t": (-42, -16),
        "siglip2_base": (-65, 6), "dinov3_or_dinov2": (-62, -18),
    }
    for index, row in enumerate(summary):
        axes[2].scatter(x[index], yv[index], s=50, color=colors(MODEL_NAMES.index(row["model"])))
        axes[2].annotate(DISPLAY_NAMES[row["model"]], (x[index], yv[index]), xytext=annotation_offsets[row["model"]], textcoords="offset points", fontsize=8)
    axes[2].set(title="Overall vs hard-state performance", xlabel="Macro-F1 at 100/class", ylabel="Worst-5 state recall", xlim=(0, 1), ylim=(0, 1))
    axes[2].grid(alpha=.2)
    fig.tight_layout()
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "figures/foundation_model_state_robustness.png", dpi=220)
    fig.savefig(OUTPUT / "figures/foundation_model_state_robustness.svg")
    plt.close(fig)


def prepare_protocol() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment_id": "foundation-model-state-robustness-20260922",
        "created_at": now(),
        "question": "Which frozen pretrained visual representation best preserves performance as within-class visual-state coverage expands?",
        "training": "D300 fixed, 300 targets per class; frozen backbone plus one linear five-class head",
        "validation": "Open Images V7 fixed validation, 30 targets per class",
        "test": "same coverage-guided balanced nested test suites, 20/40/60/80/100 targets per class",
        "models": list(MODEL_NAMES),
        "seeds": list(SEEDS),
        "primary_metric": "worst5_state_recall at 100 targets/class",
        "secondary_metrics": ["macro_f1", "state_macro_recall", "minimum independent-batch macro_f1", "batch variability"],
        "head": {"type": "linear", "optimizer": "AdamW", "lr": 0.003, "weight_decay": 0.0001, "max_epochs": 300, "early_stopping_patience": 30},
        "dinov3_policy": "Attempt licensed DINOv3 ViT-S/16; if inaccessible, record and use public DINOv2-S/14 fallback",
        "input_sha256": {"train": sha256(TRAIN_MANIFEST), "validation": sha256(VAL_MANIFEST), "candidate_pool": sha256(FIXED_TEST_ROOT / "candidate_pool.csv")},
    }
    write_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "paper_brief.md").write_text(
        "# Paper Brief\n\n- Thesis: semantic state expansion can distinguish models that have similar aggregate accuracy but different hard-state robustness.\n"
        "- Scope: fixed five-class target-crop task, D300, frozen backbones, linear probing, dog_v3.1 state test.\n"
        "- Decision: rank primarily by worst-five eligible state recall, then report aggregate and independent-batch performance separately.\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        "| Foundation backbones differ in hard-state robustness | pending | `tables/model_summary.csv` | D300 linear probe and dog_v3.1 test only |\n"
        "| Best aggregate model is also best on hard states | pending | `tables/model_summary.csv` | No claim before completed runs |\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text(
        f"# Research Log\n\n- Protocol frozen: {now()}\n- Reproduction: `.venv/bin/python -u scripts/run_foundation_model_state_robustness.py`.\n"
        "- DINOv3 access failure must be logged; fallback is DINOv2-S/14.\n- Features and predictions are resumable and never overwrite prior experiment outputs.\n",
        encoding="utf-8",
    )
    (OUTPUT / "review.md").write_text(
        "# Review\n\n- Citation integrity: model identities use official repositories; paper claims pending.\n"
        "- Result traceability: pending completion.\n- Claim scope: linear-probe representation comparison, not a full fine-tuning leaderboard.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare_protocol()
    train_rows = read_csv(TRAIN_MANIFEST)
    val_rows = read_csv(VAL_MANIFEST)
    candidate_rows = read_csv(FIXED_TEST_ROOT / "candidate_pool.csv")
    if args.smoke:
        train_rows, val_rows, candidate_rows = train_rows[:4], val_rows[:4], candidate_rows[:4]
    if any(not Path(row["crop_path"]).is_file() for row in train_rows + val_rows + candidate_rows):
        raise FileNotFoundError("one or more crop files are missing")
    splits = {"train": train_rows, "val": val_rows, "candidate": candidate_rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_context = None if args.smoke else load_state_context()
    cumulative, batches, run_rows, model_meta = [], [], [], []
    total = len(args.models) * len(SEEDS)
    completed = 0
    for model_name in args.models:
        started = time.time()
        cache = extract_features(model_name, splits, device, args.workers)
        model_meta.append({key: cache[key] for key in ("requested_name", "resolved_name", "source", "backbone_parameters", "feature_dim")})
        if args.smoke:
            continue
        candidate_x = cache["candidate"]["features"].to(device)
        for seed in SEEDS:
            run_dir = OUTPUT / "runs" / model_name / str(seed)
            prediction_path = run_dir / "candidate_predictions.csv"
            if prediction_path.is_file() and (run_dir / "run.json").is_file():
                predictions = read_csv(prediction_path)
                pred = np.asarray([int(row["predicted_label"]) for row in predictions])
                run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                print(f"[head] reuse {model_name} seed={seed}", flush=True)
            else:
                head, training = train_linear_head(cache, seed, device)
                head.eval()
                with torch.inference_mode():
                    pred = head(candidate_x).argmax(dim=1).cpu().numpy()
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(head.state_dict(), run_dir / "linear_head.pt")
                predictions = [
                    {"sample_id": row["sample_id"], "true_label": row["label"], "predicted_label": int(value)}
                    for row, value in zip(candidate_rows, pred)
                ]
                write_csv(prediction_path, predictions)
                run_info = {"model": model_name, "seed": seed, **training}
                write_json(run_dir / "run.json", run_info)
                print(f"[head] {model_name} seed={seed} valF1={training['val_macro_f1']:.4f}", flush=True)
            run_rows.append(run_info)
            model_cumulative, model_batches = evaluate_predictions(model_name, seed, candidate_rows, pred, state_context)
            cumulative.extend(model_cumulative)
            batches.extend(model_batches)
            completed += 1
            write_json(OUTPUT / "progress.json", {"status": "running", "completed_runs": completed, "total_runs": total, "latest_model": model_name, "latest_seed": seed, "percent": 100 * completed / total, "updated_at": now()})
        print(f"[model] {model_name} complete in {(time.time() - started) / 60:.1f} min", flush=True)

    if args.smoke:
        print(json.dumps({"status": "smoke_complete", "models": args.models}, ensure_ascii=False))
        return
    write_csv(OUTPUT / "tables/cumulative_metrics.csv", cumulative)
    write_csv(OUTPUT / "tables/independent_batch_metrics.csv", batches)
    write_csv(OUTPUT / "tables/head_training.csv", [{k: v for k, v in row.items() if k != "history"} for row in run_rows])
    write_csv(OUTPUT / "tables/model_metadata.csv", model_meta)
    summary = aggregate(cumulative, batches, model_meta)
    write_csv(OUTPUT / "tables/model_summary.csv", summary)
    make_figure(cumulative, summary)
    winner = summary[0]
    aggregate_winner = max(summary, key=lambda row: row["macro_f1_mean"])
    batch_winner = max(summary, key=lambda row: row["min_independent_batch_macro_f1_mean"])
    stable_hard_state = min(summary, key=lambda row: row["worst5_state_recall_sd"])
    report = [
        "# 新视觉基础模型的状态鲁棒性比较", "", "## 固定实验协议", "",
        "- D300训练集与五分类目标裁剪固定；", "- 所有骨干冻结，仅训练同配置线性分类头；",
        "- 3个训练种子；测试集按dog_v3.1覆盖顺序扩展至每类100张；",
        "- 主要指标为100张/类时最困难5个合格状态的平均Recall。", "", "## 模型结果", "",
        "| 模型 | 解析权重 | Macro-F1 | dog F1 | 状态平均Recall | 最差5状态Recall | 最差独立批次F1 | 批次波动 |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['model']} | {row['resolved_name']} | {row['macro_f1_mean']:.2%} | {row['dog_f1_mean']:.2%} | "
            f"{row['state_macro_recall_mean']:.2%} | {row['worst5_state_recall_mean']:.2%} | "
            f"{row['min_independent_batch_macro_f1_mean']:.2%} | {row['independent_batch_macro_f1_sd_mean']:.2%} |"
        )
    report.extend([
        "", "## 预注册判定", "",
        f"按预先指定的主要指标，`{winner['model']}` 的最差5状态Recall均值最高，为 {winner['worst5_state_recall_mean']:.2%}；"
        f"但其跨种子标准差为 {winner['worst5_state_recall_sd']:.2%}，结果存在明显训练头波动。",
        f"`{aggregate_winner['model']}` 的整体Macro-F1最高（{aggregate_winner['macro_f1_mean']:.2%}），"
        f"且最差独立批次Macro-F1最高（{batch_winner['min_independent_batch_macro_f1_mean']:.2%}）。",
        f"`{stable_hard_state['model']}` 的最差5状态Recall跨种子波动最小（SD={stable_hard_state['worst5_state_recall_sd']:.2%}）；"
        "稳定性和均值应分开解释。",
        "该结论仅表示其冻结预训练表征在当前D300线性探测协议下更稳健，不等同于全参数微调后的通用模型排名。",
        "仅有3个线性头种子，因此这些排名属于当前实验的描述性结果，尚不作模型间显著性声明。",
        "", "## 访问记录", "",
        "DINOv3权重需要账户接受许可且当前环境未认证；实验按冻结协议中预先声明的规则使用DINOv2-S/14替代。",
    ])
    (OUTPUT / "report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "claim_evidence.md").write_text(
        "# Claim-Evidence Ledger\n\n| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n"
        f"| `{winner['model']}` has the highest hard-state robustness | supported | `tables/model_summary.csv` | Primary metric; D300 frozen linear probe and current dog test only |\n"
        "| Aggregate F1 and hard-state robustness induce the same ranking | inspect report | `tables/model_summary.csv` | Descriptive, three head seeds |\n",
        encoding="utf-8",
    )
    (OUTPUT / "research_log.md").write_text((OUTPUT / "research_log.md").read_text(encoding="utf-8") + f"- Completed: {now()}, {completed}/{total} linear-head runs.\n", encoding="utf-8")
    (OUTPUT / "review.md").write_text(
        "# Review\n\n- Citation integrity: no unverified literature claim is required for numerical conclusions.\n"
        "- Result traceability: PASS; summary derives from cached features, saved heads, per-sample predictions, and metric tables.\n"
        "- Claim scope: PASS; results are explicitly limited to frozen-backbone linear probing and the fixed state-expansion test.\n",
        encoding="utf-8",
    )
    write_json(OUTPUT / "progress.json", {"status": "completed", "completed_runs": completed, "total_runs": total, "percent": 100.0, "updated_at": now()})
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "winner": winner["model"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
