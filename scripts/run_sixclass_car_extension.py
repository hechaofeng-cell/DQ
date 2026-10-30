#!/usr/bin/env python3
"""Launch and resume all stages of the preregistered six-class car extension."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/sixclass_car_extension_20261004'
PYTHON = '/home/hcf/project/QE/.venv/bin/python'


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def status(stage: str, **extra) -> None:
    write_json(STUDY / 'progress.json', {'status': 'running', 'stage': stage,
               'pid': os.getpid(), 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra})
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] stage={stage} {extra}", flush=True)


def preregister() -> None:
    path = STUDY / 'preregistration.json'
    if path.exists():
        return
    write_json(path, {
        'registered_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'classes': ['dog', 'cat', 'horse', 'sheep', 'person', 'car'],
        'models': ['resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large',
                   'densenet121', 'efficientnet_b0', 'convnext_tiny', 'swin_t'],
        'seeds': [20260916, 20260917, 20260918],
        'epochs': 20, 'samples_per_epoch': 3000, 'batch_size': 64,
        'optimizer': 'AdamW(lr=1e-4,weight_decay=1e-4)', 'scheduler': 'cosine',
        'checkpoint_selection': 'validation Macro-F1 only',
        'train_per_class': 300, 'candidate_per_class': 400, 'validation_per_class': 15,
        'base_test_per_class': 50, 'budgets_per_class': [20, 40, 60, 80, 100],
        'random_replicates': 1000, 'shared_fields': 14, 'frequency_cap': 20, 'minimum_state_support': 3,
        'primary_family_I1': ['sixclass_equal_error_confusion', 'sixclass_equal_error_clusters'],
        'secondary_family_I2': ['car_equal_error_confusion', 'car_equal_error_clusters'],
        'test': 'architecture-level sign flip; 5000 hierarchy bootstrap; Holm within each predefined family plus global sensitivity',
        'stopping': 'one complete frozen run; no model/point/class selection by outcomes',
        'car_annotator': 'sensenova-6.8-flash-lite',
        'old_classes_annotator': 'local qwen3-vl:30b-a3b-instruct',
        'near_duplicate_rule': 'pHash<=4 and ViT cosine>=.980 OR pHash<=8 and cosine>=.995; exact hashes always excluded',
        'limitations': ['category and annotator are not fully crossed',
                        'comparison to five-class study changes task and feature projection'],
        'protocol_file': 'docs/sixclass_car_extension_protocol_20261004.md',
    })


def spawn_stage(name: str, arguments: list[str]) -> tuple[subprocess.Popen, object]:
    logs = STUDY / 'logs'
    logs.mkdir(exist_ok=True)
    stream = (logs / f'{name}.log').open('a', encoding='utf-8')
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
    child = subprocess.Popen([PYTHON, '-u', *arguments], cwd=ROOT, stdout=stream,
                             stderr=subprocess.STDOUT, env=environment)
    write_json(STUDY / f'{name}_process.json', {'pid': child.pid, 'arguments': arguments,
               'log': str((logs / f'{name}.log').resolve()), 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    return child, stream


def wait_stage(child: subprocess.Popen, stage: str, **extra) -> None:
    while child.poll() is None:
        status(stage, child_pid=child.pid, **extra)
        time.sleep(20)
    if child.returncode:
        raise RuntimeError(f'{stage} failed: exit={child.returncode}; see logs/{stage}.log')


def train_models(limit: int = 0) -> None:
    import torch
    from sixclass_car_support import configure, benchmark, training
    configure()
    torch.set_num_threads(8)
    train = benchmark.read_csv(STUDY / 'manifests/train.csv')
    evaluation = benchmark.read_csv(STUDY / 'manifests/eval.csv')
    val = [row for row in evaluation if row['split'] == 'val']
    test = [row for row in evaluation if row['split'] == 'test']
    configurations = list(itertools.product(benchmark.MODELS, benchmark.SEEDS))
    if limit:
        configurations = configurations[:limit]
    for index, (model, seed) in enumerate(configurations, 1):
        write_json(STUDY / 'training/progress.json', {'status': 'running', 'completed_runs': index - 1,
                   'total_runs': len(configurations), 'current_model': model, 'current_seed': seed,
                   'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        print(f'[sixclass training] {index}/24 {model}/{seed}', flush=True)
        training.train_one(model, 'oi_gap_add0', seed, train, val, test, epochs=20, workers=8)
        torch.cuda.empty_cache()
    write_json(STUDY / 'training/progress.json', {'status': 'complete' if not limit else 'initial_run_complete',
               'completed_runs': len(configurations), 'total_runs': 24})


def preliminary_audit() -> None:
    from sixclass_car_support import benchmark
    import numpy as np
    from audit_multiclass_near_duplicates import phash, hamming_matrix, status as duplicate_status
    partitions = {
        'candidate': benchmark.read_csv(STUDY / 'manifests/candidate_pool.csv'),
        'train': benchmark.read_csv(STUDY / 'manifests/train.csv'),
        'validation_and_base_test': benchmark.read_csv(STUDY / 'manifests/eval.csv'),
    }
    hashes = {name: [benchmark.sha256(Path(row['crop_path'])) for row in rows] for name, rows in partitions.items()}
    duplicates = []
    for left, values in hashes.items():
        for right, others in hashes.items():
            if left >= right:
                continue
            common = set(values) & set(others)
            if common:
                duplicates.append({'left': left, 'right': right, 'common_crop_hashes': sorted(common)})
    if duplicates:
        write_json(STUDY / 'pretraining_audit.json', {'status': 'failed', 'duplicates': duplicates})
        raise ValueError('exact crop leakage detected before training')
    embeddings = {}
    for name, rows in partitions.items():
        print(f'[leakage audit] independent ViT embeddings {name}/{len(rows)}', flush=True)
        embeddings[name] = benchmark.candidate_embeddings({'rows': rows}, STUDY / f'audit/{name}_vit_embeddings.npz', 8)
        embeddings[name] /= np.maximum(np.linalg.norm(embeddings[name], axis=1, keepdims=True), 1e-12)
    phashes = {name: np.asarray([phash(Path(row['crop_path'])) for row in rows], dtype=np.uint64)
               for name, rows in partitions.items()}
    pairs = []
    for left_name, right_name in (('candidate', 'train'), ('candidate', 'validation_and_base_test'), ('train', 'validation_and_base_test'), ('candidate', 'candidate')):
        distances = hamming_matrix(phashes[left_name], phashes[right_name])
        cosine = embeddings[left_name] @ embeddings[right_name].T
        mask = ((distances <= 4) & (cosine >= .980)) | ((distances <= 8) & (cosine >= .995))
        if left_name == right_name:
            mask = np.triu(mask, k=1)
        for left, right in zip(*np.where(mask)):
            lrow, rrow = partitions[left_name][int(left)], partitions[right_name][int(right)]
            exact = hashes[left_name][int(left)] == hashes[right_name][int(right)]
            pairs.append({'scope': f'{left_name}_vs_{right_name}', 'left_id': lrow['sample_id'], 'right_id': rrow['sample_id'],
                          'phash_distance': int(distances[left, right]), 'vit_cosine': float(cosine[left, right]),
                          'decision': duplicate_status(int(distances[left, right]), float(cosine[left, right]), exact)})
    write_json(STUDY / 'pretraining_audit.json', {'status': 'pass' if not pairs else 'needs_resolution',
               'counts': {name: len(rows) for name, rows in partitions.items()}, 'flagged_pairs': pairs,
               'encoder': 'ImageNet-pretrained ViT-B/16 independent of six-class training'})
    if pairs:
        raise ValueError(f'{len(pairs)} near-duplicate pairs need deterministic data-only resolution before training')


def smoke_analysis() -> None:
    import torch
    from sixclass_car_support import configure, benchmark, decoupled, stratified, incremental
    configure()
    torch.set_num_threads(8)
    context = benchmark.build_context()
    model, seed = benchmark.MODELS[0], benchmark.SEEDS[0]
    smoke = STUDY / 'smoke'
    benchmark.run_benchmark(smoke / 'benchmark', context, 8, True)
    for module in (stratified, incremental):
        module.MODELS = (model,)
        module.SEEDS = (seed,)
    decoupled.BASE = smoke / 'benchmark'
    incremental.BASE = smoke / 'benchmark'
    stratified.run(smoke / 'strict', 20)
    outputs = [smoke / 'benchmark/tables/fixed_method_metrics.csv', smoke / 'strict/tables/class_equal_error_metrics.csv']
    for path, expected in zip(outputs, (20, 20)):
        rows = benchmark.read_csv(path)
        if len(rows) != expected:
            raise ValueError(f'smoke row count: {path} {len(rows)} != {expected}')
    write_json(smoke / 'audit.json', {'status': 'pass', 'models': 1, 'seeds': 1, 'logit_classes': 6,
               'test_budgets': [20, 40, 60, 80, 100], 'not_confirmatory': True})


def technical_check() -> None:
    """Exercise six-class inference and all selection scores on real annotated targets."""
    import numpy as np
    import torch
    from sixclass_car_support import configure, benchmark, stratified, error_structures, CLASSES, LABELS
    configure()
    torch.set_num_threads(8)
    full_rows = benchmark.read_csv(STUDY / 'manifests/candidate_pool.csv')
    old_features = {row['image_id']: row for row in benchmark.read_csv(
        ROOT / 'artifacts/multiclass_object_pool_v1_20260928/annotations/results.csv')}
    car_features = {}
    for path in sorted((STUDY / 'online_car_annotations/parsed').glob('*.json'))[:3]:
        parsed = json.loads(path.read_text())
        car_features[parsed['image_id']] = {
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in parsed['features'].items()
        }
    if len(car_features) != 3:
        raise ValueError('technical check needs three actual online car responses')
    rows, features, by_class = [], {**old_features, **car_features}, {}
    schema = json.loads((STUDY / 'configs/shared_object_schema.json').read_text())
    for name in CLASSES:
        subset = [row for row in full_rows if row['class_name'] == name and row['sample_id'] in features][:3]
        if len(subset) != 3:
            raise ValueError(f'technical check has insufficient {name} targets')
        start = len(rows)
        rows.extend(subset)
        matrix, states = benchmark.state_matrix(subset, features, schema)
        by_class[name] = {'rows': subset, 'matrix': matrix, 'states': states,
                          'global_indices': np.arange(start, start + 3)}
    context = {'rows': rows, 'features': features, 'by_class': by_class,
               'train_rows': benchmark.read_csv(STUDY / 'manifests/train.csv')}
    output = STUDY / 'technical_smoke'
    arrays = benchmark.infer_and_extract(output, context, benchmark.MODELS[0], benchmark.SEEDS[0], 8)
    assert arrays['candidate_logits'].shape == (18, 6)
    assert set(arrays['train_labels'].tolist()) == set(range(6))
    probabilities = benchmark.softmax(arrays['candidate_logits'])
    predicted = probabilities.argmax(1)
    independent = benchmark.make_model_independent_orders(context, output, 8)
    dsa = benchmark.dsa_scores(arrays['train_features'], arrays['train_labels'], arrays['candidate_features'], predicted)
    gini = 1 - np.square(probabilities).sum(1)
    orders = {**independent, 'deepgini': {}, 'dsa': {}}
    for name in CLASSES:
        section = by_class[name]
        indices = section['global_indices']
        orders['deepgini'][name] = benchmark.ranked(gini[indices], section['rows'])
        orders['dsa'][name] = benchmark.ranked(dsa[indices], section['rows'])
    orders['random'] = benchmark.random_orders(by_class, 1)[0]
    structures = error_structures(context, arrays['candidate_features'], predicted, 'resnet18', benchmark.SEEDS[0])
    records = []
    for method, order in orders.items():
        for budget in (1, 2, 3):
            metrics = benchmark.evaluate_suite(context, order, budget, predicted)
            assert all(np.isfinite(value) for value in metrics.values())
            pair, per_class = stratified.class_stratified_pair(context, orders['coverage'], order, budget, predicted, structures)
            assert len(per_class) == 6 and all(np.isfinite(value) for value in pair.values())
            records.append({'method': method, 'budget': budget, **metrics, **pair})
    benchmark.write_csv(output / 'selection_metrics.csv', records)
    write_json(output / 'audit.json', {'status': 'pass', 'real_targets': 18, 'classes': 6,
               'sixclass_logit_shape': [18, 6], 'scored_methods': list(orders), 'records': len(records),
               'not_confirmatory': True, 'note': 'technical subset excluded from scientific comparisons'})


def full_analysis() -> None:
    import torch
    from sixclass_car_support import configure, benchmark, decoupled, stratified, category_specific_diagnostics, plot_and_report
    configure()
    torch.set_num_threads(8)
    context = benchmark.build_context()
    benchmark.run_benchmark(STUDY / 'benchmark', context, 8, False)
    stratified.run(STUDY / 'strict', 1000)
    decoupled.run(STUDY / 'holdout', False)
    category_specific_diagnostics()
    plot_and_report()
    expected = {
        'benchmark/tables/fixed_method_metrics.csv': 480,
        'benchmark/tables/random_summary.csv': 120,
        'benchmark/tables/auc_by_run.csv': 120,
        'strict/tables/class_equal_error_metrics.csv': 480,
        'strict/tables/class_equal_error_by_class.csv': 2880,
        'strict/tables/class_equal_error_comparisons.csv': 8,
        'strict/tables/hybrid_class_equal_error_metrics.csv': 480,
        'strict/tables/hybrid_class_equal_error_by_class.csv': 2880,
        'holdout/tables/fixed_metrics.csv': 2400,
        'holdout/tables/random_summary.csv': 600,
        'diagnostics/native_state_budget_metrics.csv': 3600,
        'diagnostics/car_equal_error_comparisons.csv': 8,
    }
    observed = {}
    for name, count in expected.items():
        rows = benchmark.read_csv(STUDY / name)
        if len(rows) != count:
            raise ValueError(f'audit mismatch {name}: {len(rows)} != {count}')
        observed[name] = len(rows)
    import numpy as np
    numeric_metrics = {
        'benchmark/tables/fixed_method_metrics.csv': ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_frequency_coverage'),
        'strict/tables/class_equal_error_metrics.csv': ('left_confusion', 'right_confusion', 'left_clusters', 'right_clusters'),
        'strict/tables/class_equal_error_comparisons.csv': ('mean_delta_auc', 'ci95_low', 'ci95_high', 'permutation_p'),
        'diagnostics/native_state_budget_metrics.csv': ('frequency_coverage', 'presence_coverage', 'error_rate'),
        'diagnostics/car_equal_error_comparisons.csv': ('mean_delta_auc', 'ci95_low', 'ci95_high', 'permutation_p'),
    }
    for name, fields in numeric_metrics.items():
        for row in benchmark.read_csv(STUDY / name):
            if any(not np.isfinite(float(row[field])) for field in fields):
                raise ValueError(f'nonfinite or missing audited metric in {name}')
    for model, seed in itertools.product(benchmark.MODELS, benchmark.SEEDS):
        path = STUDY / 'benchmark/runs' / model / str(seed) / 'candidate_outputs.npz'
        with np.load(path) as arrays:
            if arrays['logits'].shape != (2400, 6) or not np.isfinite(arrays['logits']).all():
                raise ValueError(f'invalid six-class logits: {path}')
    manifest = {str(path.relative_to(STUDY)): benchmark.sha256(path)
                for path in STUDY.rglob('*') if path.is_file() and path.suffix in {'.csv', '.npz', '.pt'} }
    write_json(STUDY / 'artifact_manifest.sha256.json', manifest)
    write_json(STUDY / 'final_audit.json', {'status': 'pass', 'observed_rows': observed,
               'total_audited_rows': sum(observed.values()), 'models': 8, 'seeds': 3, 'classes': 6})
    with (STUDY / 'research_log.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## 全量完成\n\n'
                     f'- {time.strftime("%Y-%m-%dT%H:%M:%S%z")}：24个六分类运行、五方法、逐类别控制、状态留出和原生状态诊断完成。\n'
                     f'- 最终核对{sum(observed.values())}行预定结果与24个2400×6有限logits数组，审计通过。\n'
                     '- 判定与负结果全部依据冻结主指标，详见report_zh.md，不改旧五类冻结论文结论。\n')
    (STUDY / 'review.md').write_text('# Final Review\n\n'
        '- Result traceability: PASS; logits, orders, result tables and SHA-256 recorded.\n'
        '- Claim scope: six-class constructed configuration; car-only effects reported separately.\n'
        '- Annotation limitation: five old classes local, car online; category and annotator confounded.\n'
        '- No direct five-class vs six-class F1 difference claim.\n', encoding='utf-8')


def pipeline() -> None:
    preregister()
    if not (STUDY / 'data_audit.json').is_file():
        status('prepare')
        child, stream = spawn_stage('prepare', ['scripts/prepare_sixclass_car_extension.py'])
        wait_stage(child, 'prepare')
        stream.close()
    status('pretraining_audit')
    preliminary_audit()
    annotation_args = ['scripts/annotate_car_online_resumable.py', '--input',
                       str(STUDY / 'inputs/car_feature_manifest.csv'), '--output',
                       str(STUDY / 'online_car_annotations'), '--schema',
                       str(STUDY / 'configs/car_annotation_schema.json'), '--prompt',
                       str(ROOT / 'prompts/feature_annotation_car_shared_v1_0.txt'),
                       '--workers', '4', '--start-interval', '3', '--resume']
    annotation_child, annotation_stream = spawn_stage('online_car', annotation_args)
    training_child, training_stream = spawn_stage('training', ['scripts/run_sixclass_car_extension.py', '--stage', 'train', '--train-limit', '1'])
    status('training_and_online_annotations', training_pid=training_child.pid, annotation_pid=annotation_child.pid)
    while training_child.poll() is None or annotation_child.poll() is None:
        for name, child in (('training', training_child), ('online_car', annotation_child)):
            if child.poll() not in (None, 0):
                raise RuntimeError(f'{name} failed, exit={child.returncode}; see logs/{name}.log')
        status('training_and_online_annotations', training_pid=training_child.pid,
               annotation_pid=annotation_child.pid, training_exit=training_child.poll(), annotation_exit=annotation_child.poll())
        time.sleep(20)
    training_stream.close()
    annotation_stream.close()
    for recovery_round in range(1, 3):
        report = json.loads((STUDY / 'online_car_annotations/report.json').read_text())
        if report['successful'] == 400 and report['failed'] == 0:
            break
        status('retry_incomplete_annotations', recovery_round=recovery_round,
               remaining=report['failed'])
        child, stream = spawn_stage('online_car', annotation_args)
        wait_stage(child, 'online_car', recovery_round=recovery_round)
        stream.close()
    from prepare_sixclass_car_extension import merge_annotations
    merge_annotations(STUDY)
    status('smoke_analysis')
    child, stream = spawn_stage('smoke_analysis', ['scripts/run_sixclass_car_extension.py', '--stage', 'smoke'])
    wait_stage(child, 'smoke_analysis')
    stream.close()
    status('full_training')
    child, stream = spawn_stage('training', ['scripts/run_sixclass_car_extension.py', '--stage', 'train'])
    wait_stage(child, 'training')
    stream.close()
    status('full_analysis')
    child, stream = spawn_stage('full_analysis', ['scripts/run_sixclass_car_extension.py', '--stage', 'analysis'])
    wait_stage(child, 'full_analysis')
    stream.close()
    write_json(STUDY / 'progress.json', {'status': 'complete', 'stage': 'complete', 'pid': os.getpid(),
               'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'report': str(STUDY / 'report_zh.md')})
    print('[sixclass pipeline] all stages complete', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pipeline', 'train', 'smoke', 'analysis', 'check'), default='pipeline')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--train-limit', type=int, default=0)
    args = parser.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.background:
        pid_path = STUDY / 'background_process.json'
        if pid_path.is_file():
            previous = json.loads(pid_path.read_text())
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"existing pipeline still active pid={previous['pid']}")
        preregister()
        stream = (STUDY / 'pipeline.log').open('a', encoding='utf-8')
        child = subprocess.Popen([PYTHON, '-u', str(Path(__file__).resolve())], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
        stream.close()
        write_json(pid_path, {'pid': child.pid, 'log': str(STUDY / 'pipeline.log')})
        print(json.dumps({'status': 'started', 'pid': child.pid, 'log': str(STUDY / 'pipeline.log')}, ensure_ascii=False))
        return
    try:
        if args.stage == 'pipeline':
            pipeline()
        elif args.stage == 'train':
            train_models(args.train_limit)
        elif args.stage == 'smoke':
            smoke_analysis()
        elif args.stage == 'check':
            technical_check()
        else:
            full_analysis()
    except Exception as error:
        write_json(STUDY / f'{args.stage}_failure.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        if args.stage == 'pipeline':
            write_json(STUDY / 'progress.json', {'status': 'failed', 'error': str(error), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
