#!/usr/bin/env python3
"""Small offline check of the merged workspace; preserve frozen source files."""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
POOLS = {
    'B': 'sixclass_independent_confirmation_20261004',
    'C': 'sixclass_voc_external_confirmation_20261005',
    'D': 'objects365_independent_confirmation_20261005',
}
MODELS = ('resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large',
          'densenet121', 'efficientnet_b0', 'convnext_tiny', 'swin_t')
PATH_FIELDS = ('image_path', 'crop_path', 'marked_image_path', 'feature_crop_path')


def relocated(value):
    match = re.match(r'^/home/hcf/\.codex/worktrees/[^/]+/QE/(.+)$', value)
    return ROOT / match[1] if match else Path(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/integration_smoke_20261009/image_inference')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import numpy as np
    import torch
    from torchvision import models
    import run_four_model_gap_vs_random as training
    from run_semantic_test_selection_baselines import extract_classifier_inputs

    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is unavailable; use --device cpu for an execution-only probe.'
        # Historical inference used CUDA batches of 96. Its precision flags
        # were not recorded explicitly; this TF32 configuration was checked
        # against the sampled caches. Reduced batch shapes can change borderline
        # predictions, so pad this
        # small probe to the historical batch size without adding unique targets.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    training.CLASS_SPECS = tuple(zip(CLASSES, (16, 15, 17, 18, 0, 2)))
    all_rows = []
    checks = []
    references = {}
    source_hashes = {}
    for pool, directory in POOLS.items():
        base = ROOT / 'artifacts' / directory
        manifest = base / 'manifests/candidate_pool.csv'
        source_hashes[manifest] = digest(manifest)
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        stale = 0
        missing = []
        remapped = []
        for row in rows:
            new = dict(row)
            for field in PATH_FIELDS:
                path = relocated(row[field])
                stale += str(path) != row[field]
                if not path.is_file():
                    missing.append({'sample_id': row['sample_id'], 'field': field, 'path': str(path)})
                new[field] = str(path)
            remapped.append(new)
        check = dict(pool=pool, targets=len(rows), checked_paths=len(rows)*len(PATH_FIELDS),
                     legacy_worktree_paths=stale, missing_paths=missing)
        checks.append(check)
        assert not missing, check
        # Save separate relocation copies; never rewrite the frozen originals.
        with (args.output / f'{pool}_relocated_candidate_pool.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(remapped)
        chosen = [next(row for row in remapped if row['class_name'] == c) for c in CLASSES]
        for row in chosen:
            for field, hash_field in [('image_path', 'image_sha256'), ('crop_path', 'crop_sha256')]:
                if row.get(hash_field):
                    assert digest(Path(row[field])) == row[hash_field], (pool, row['sample_id'], field)
        ids = [row['sample_id'] for row in rows]
        positions = np.array([ids.index(row['sample_id']) for row in chosen])
        all_rows.extend(chosen)
        for name in MODELS:
            cache = base / 'benchmark/runs' / name / '20260916/candidate_outputs.npz'
            source_hashes[cache] = digest(cache)
            with np.load(cache) as z:
                assert z['sample_ids'].tolist() == ids
                references[pool, name] = z['logits'][positions].copy()
        print(f'{pool}: checked {check["checked_paths"]} relocated paths, selected 6 real images', flush=True)

    records = []
    for name in MODELS:
        # Exercise the existing model builder, then load the full trained weights;
        # skip redundant ImageNet initialization and all download requests.
        factory = getattr(models, name)
        def offline_factory(*a, **kw):
            kw['weights'] = None
            return factory(*a, **kw)
        with patch.object(models, name, offline_factory):
            model = training.build_model(name)
        checkpoint = ROOT / 'artifacts/sixclass_car_extension_20261004/training/runs' / name / 'oi_gap_add0/20260916/best.pt'
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(loaded['model'], strict=True)
        model = model.to(device)
        inference_rows = all_rows + [all_rows[-1]] * (96 - len(all_rows))
        with torch.inference_mode():
            _, padded_logits = extract_classifier_inputs(model, inference_rows, device, 0)
        logits = padded_logits[:len(all_rows)]
        assert logits.shape == (18, 6) and np.isfinite(logits).all()
        np.save(args.output / f'{name}_logits.npy', logits)
        for i, pool in enumerate(POOLS):
            current = logits[i*6:(i+1)*6]
            historical = references[pool, name]
            matches = int((current.argmax(-1) == historical.argmax(-1)).sum())
            record = dict(pool=pool, model=name, seed=20260916, targets=6,
                          prediction_matches=matches,
                          max_absolute_logit_difference=float(np.abs(current-historical).max()),
                          mismatches=[dict(sample_id=all_rows[i*6+j]['sample_id'],
                                           historical_prediction=int(historical[j].argmax()),
                                           current_prediction=int(current[j].argmax()))
                                      for j in np.flatnonzero(current.argmax(-1) != historical.argmax(-1))])
            records.append(record)
        matched = sum(r['prediction_matches'] for r in records if r['model'] == name)
        print(f'{name}: {matched}/18 predictions match the historical cache', flush=True)
        del loaded, model
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    assert all(digest(p) == h for p, h in source_hashes.items())
    all_match = all(r['prediction_matches'] == r['targets'] for r in records)
    report = dict(status='pass' if all_match else 'prediction_mismatch', root=str(ROOT), elapsed_seconds=round(time.monotonic()-started, 2),
                  path_checks=checks, real_images=18, architectures=8, seeds_per_model=1,
                  predictions_checked=len(all_rows)*len(MODELS), inference_device=args.device,
                  batch_size=96, unique_targets=18, padding_rows=78,
                  allow_tf32=args.device == 'cuda',
                  source_manifests_and_caches_unchanged=True, new_training=False,
                  vlm_calls=0, legacy_worktree_paths=sum(c['legacy_worktree_paths'] for c in checks),
                  historical_comparisons=records,
                  boundary='Original manifests retain legacy absolute paths; relocation copies and this check use current-root paths. No full rerun or VLM service validation.')
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'{report["status"]}: {report["predictions_checked"]} real-image predictions checked; original inputs unchanged.', flush=True)
    if not all_match:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
