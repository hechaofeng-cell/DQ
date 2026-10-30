#!/usr/bin/env python3
"""Adopt audited same-class replacements, infer only new targets, then diagnose.

Original studies, weights, and refused samples remain untouched. This adapter
reuses the frozen analysis and records the explicit pre-analysis pool amendment.
"""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

import dual_vlm_common as common
import analyze_dual_vlm_study as analysis

ROOT, ORIGINAL, SOURCE = common.ROOT, common.STUDY, common.SOURCE
REPLACEMENTS = ROOT / 'artifacts/dual_vlm_replacement_pair_20261004'
STUDY = ROOT / 'artifacts/sixclass_dual_vlm_replaced_20261004'
PROTOCOL = ROOT / 'docs/sixclass_dual_vlm_replacement_protocol_20261004.md'
read_csv, write_csv, write_json, sha256, now = common.read_csv, common.write_csv, common.write_json, common.sha256, common.now


def note(stage, **extra):
    value = dict(status='running', stage=stage, pid=os.getpid(), heartbeat=now(), **extra)
    write_json(STUDY / 'progress.json', value)
    print(f'[{now()}] {stage} {json.dumps(extra, ensure_ascii=False)}', flush=True)


def prepare():
    note('validate_replacement_pool')
    analysis.verify_code_freeze()
    source_check = common.verify_freeze()
    mapping = read_csv(REPLACEMENTS / 'replacement_mapping.csv')
    old_rows = read_csv(ORIGINAL / 'manifests/candidate_pool.csv')
    rows = read_csv(REPLACEMENTS / 'candidate_pool_with_replacements.csv')
    changed = [(old, new) for old, new in zip(old_rows, rows) if old != new]
    if len(rows) != 2400 or len(changed) != 2 or len({r['sample_id'] for r in rows}) != 2400:
        raise ValueError('replacement identity/count audit failed')
    if {(o['sample_id'], n['sample_id']) for o, n in changed} != {(r['old_sample_id'], r['new_sample_id']) for r in mapping}:
        raise ValueError('replacement mapping does not match full pool')
    if Counter(r['class_name'] for r in rows) != Counter({c: 400 for c in common.CLASSES}):
        raise ValueError('class budgets changed')
    for directory in ('configs', 'prompts', 'manifests'):
        destination = STUDY / directory
        destination.mkdir(parents=True, exist_ok=True)
        for path in (ORIGINAL / directory).glob('*'):
            if path.is_file() and not (destination / path.name).exists():
                shutil.copyfile(path, destination / path.name)
    write_csv(STUDY / 'manifests/candidate_pool.csv', rows)
    shutil.copyfile(REPLACEMENTS / 'replacement_mapping.csv', STUDY / 'manifests/replacement_mapping.csv')
    replacement_inputs = {r['image_id']: r for r in read_csv(REPLACEMENTS / 'replacement_annotation_manifest.csv')}
    replacements = {r['old_sample_id']: r['new_sample_id'] for r in mapping}
    native_counts = {}
    for domain in ('old', 'car'):
        inputs = [replacement_inputs[replacements[r['image_id']]] if r['image_id'] in replacements else r
                  for r in read_csv(ORIGINAL / 'inputs' / f'{domain}.csv')]
        write_csv(STUDY / 'inputs' / f'{domain}.csv', inputs)
        schema = json.loads(common.schema_path(domain).read_text())
        for backend in analysis.BACKENDS:
            destination = STUDY / 'annotations' / backend / domain / 'parsed'
            destination.mkdir(parents=True, exist_ok=True)
            for row in inputs:
                key = row['image_id']
                source = (REPLACEMENTS / 'annotations' / backend / 'parsed' / f'{key}.json'
                          if key in replacements.values() else common.job_path(backend, domain) / 'parsed' / f'{key}.json')
                analysis.validate_feature(json.loads(source.read_text()), row, schema)
                shutil.copyfile(source, destination / source.name)
            native_counts[backend, domain] = len(inputs)
    paths = [PROTOCOL, Path(__file__), *[Path(p) for p in json.loads((ORIGINAL / 'analysis_code_freeze.sha256.json').read_text())]]
    paths += list((STUDY / 'manifests').glob('*.csv')) + list((STUDY / 'inputs').glob('*.csv'))
    paths += list((STUDY / 'configs').glob('*')) + list((STUDY / 'prompts').glob('*'))
    paths += list((STUDY / 'annotations').glob('*/*/parsed/*.json'))
    paths += [REPLACEMENTS / 'audit.json', REPLACEMENTS / 'selection_protocol.json']
    for row in (n for _o, n in changed):
        for field in ('image_path', 'crop_path', 'marked_image_path', 'feature_crop_path'):
            paths.append(Path(row[field]))
        if sha256(row['image_path']) != row['image_sha256'] or sha256(row['crop_path']) != row['crop_sha256']:
            raise ValueError('new target image hash mismatch')
    freeze_path = STUDY / 'amendment_input_freeze.sha256.json'
    hashes = {str(p): sha256(p) for p in paths}
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != hashes:
        raise ValueError('amended inputs changed since prior execution')
    write_json(freeze_path, hashes)
    write_json(STUDY / 'preparation_audit.json', dict(status='pass', source_freeze=source_check,
               targets=2400, changed=2, classes={c:400 for c in common.CLASSES},
               local_labels=sum(n for (b, _), n in native_counts.items() if b == 'local'),
               online_labels=sum(n for (b, _), n in native_counts.items() if b == 'online')))
    if not (STUDY / 'research_log.md').exists():
        (STUDY / 'research_log.md').write_text(
            '# 修订执行日志\n\n' + f'{now()}：用户授权分析替换结果并执行后续。\n\n'
            '原池2个目标因线上服务拒绝作同类替换；未使用分类结果选图。'
            '训练1800、24份权重、预算20/40/60/80/100、检验族、1000随机重复固定；'
            '新两张只补推理，所有测试次序重算。原失败包及替换包均保留。\n\n'
            '命令：`/home/hcf/project/QE/.venv/bin/python -u scripts/run_dual_vlm_replaced_study.py --background`\n'
            '方案：docs/sixclass_dual_vlm_replacement_protocol_20261004.md。'
            '运行前输入及代码哈希：amendment_input_freeze.sha256.json。\n', encoding='utf-8')
        (STUDY / 'paper_brief.md').write_text(
            '# 研究范围\n\n同类替换2个线上被拒目标后的平衡六类2400池；'
            '检验状态覆盖测试诊断相对随机的收益是否在两种自动标注器下复现。'
            '复用固定模型，不主张训练改善或标签人工准确。全部强基线负结果保留。\n', encoding='utf-8')
    return rows, [n for _o, n in changed]


def infer_replacements(rows, new_rows):
    import torch
    from torch import nn
    from torchvision import models
    from torch.utils.data import DataLoader
    from run_semantic_test_selection_baselines import extract_classifier_inputs
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = [next(i for i,r in enumerate(rows) if r['sample_id'] == n['sample_id']) for n in new_rows]
    # cuDNN TF32 convolution can change numerical results when batch size
    # changes. Recreate each original 96-target batch containing a replacement,
    # audit all unchanged probes, and adopt only the two new target outputs.
    probe_indices = sorted({i for index in indices for i in range((index//96)*96,min((index//96+1)*96,2400))})
    probes = [rows[i] for i in probe_indices]
    changed_probe = [probe_indices.index(index) for index in indices]
    anchor_probe = [i for i,index in enumerate(probe_indices) if index not in indices]
    anchor_indices = [probe_indices[i] for i in anchor_probe]
    ids = np.asarray([r['sample_id'] for r in rows])
    old_ids = [r['sample_id'] for r in read_csv(ORIGINAL / 'manifests/candidate_pool.csv')]
    unchanged = np.ones(2400, dtype=bool)
    unchanged[indices] = False
    checks = []
    for run_i, (model_name, seed) in enumerate(itertools.product(common.MODELS, common.SEEDS), 1):
        note('replacement_classifier_inference', completed_runs=run_i-1, total_runs=24, model=model_name, seed=seed, device=str(device))
        output = STUDY / 'benchmark/runs' / model_name / str(seed)
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'candidate_outputs.npz'
        checkpoint = SOURCE / 'training/runs' / model_name / 'oi_gap_add0' / str(seed) / 'best.pt'
        original = SOURCE / 'benchmark/runs' / model_name / str(seed) / 'candidate_outputs.npz'
        with np.load(original) as saved:
            if saved['sample_ids'].tolist() != old_ids:
                raise ValueError('original classifier cache identities differ')
            features, logits = saved['features'].copy(), saved['logits'].copy()
        if not path.exists():
            model = analysis.b.build_model(model_name)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            model.to(device).eval()
            new_features, new_logits = extract_classifier_inputs(model, probes, device, 0)
            if not np.allclose(new_logits[anchor_probe], logits[anchor_indices], atol=1e-4, rtol=1e-4):
                raise ValueError(f'unchanged anchor inference mismatch: {model_name}/{seed}')
            if not np.allclose(new_features[anchor_probe],features[anchor_indices],atol=1e-4,rtol=1e-4):
                raise ValueError('unchanged anchor feature mismatch')
            if not np.array_equal(new_logits[anchor_probe].argmax(1), logits[anchor_indices].argmax(1)):
                raise ValueError('unchanged anchor argmax mismatch')
            features[indices], logits[indices] = new_features[changed_probe], new_logits[changed_probe]
            np.savez_compressed(path, sample_ids=ids, features=features, logits=logits)
            np.savez_compressed(output / 'replacement_outputs.npz', sample_ids=np.asarray([r['sample_id'] for r in new_rows]),
                                features=new_features[changed_probe], logits=new_logits[changed_probe], anchor_logits=new_logits[anchor_probe])
            del model
            torch.cuda.empty_cache()
        with np.load(path) as revised, np.load(original) as old:
            if revised['sample_ids'].tolist() != ids.tolist() or revised['logits'].shape != (2400,6):
                raise ValueError('revised classifier cache mismatch')
            if not all(np.isfinite(revised[k]).all() for k in ('features','logits')):
                raise ValueError('nonfinite classifier outputs')
            if not all(np.array_equal(revised[k][unchanged], old[k][unchanged]) for k in ('features','logits')):
                raise ValueError('unchanged 2398 target predictions were modified')
            full_features, full_logits = revised['features'], revised['logits']
        training_path = SOURCE / 'benchmark/runs' / model_name / str(seed) / 'train_features.npz'
        with np.load(training_path) as train:
            dsa = analysis.b.dsa_scores(train['features'], train['labels'], full_features, full_logits.argmax(1))
        gini = 1 - np.square(analysis.b.softmax(full_logits)).sum(1)
        for method, scores in (('deepgini',gini),('dsa',dsa)):
            order_rows = []
            for name in common.CLASSES:
                positions = [i for i,r in enumerate(rows) if r['class_name'] == name]
                subset = [rows[i] for i in positions]
                order = analysis.b.ranked(scores[positions], subset)
                order_rows.extend(dict(method=method,class_name=name,rank=rank+1,sample_id=subset[i]['sample_id'],score=float(scores[positions[i]]))
                                  for rank,i in enumerate(order))
            write_csv(STUDY / 'benchmark/orders' / model_name / str(seed) / f'{method}.csv', order_rows)
        checks.append(dict(model=model_name,seed=seed,unchanged_predictions=2398,new_predictions=2,
                           audit_probe_targets=len(anchor_probe),inference_batch_size=96,
                           checkpoint_sha256=sha256(checkpoint),source_cache_sha256=sha256(original),
                           training_features_sha256=sha256(training_path),revised_cache_sha256=sha256(path)))
        write_json(STUDY / 'inference_audit.json',dict(status='running' if run_i<24 else 'pass',runs=checks))
    note('replacement_vit_embeddings')
    embedding_path = STUDY / 'benchmark/embeddings/vit_b16_candidates.npz'
    source_embedding = SOURCE / 'benchmark/embeddings/vit_b16_candidates.npz'
    if not embedding_path.exists():
        with np.load(source_embedding) as saved:
            if saved['sample_ids'].tolist() != old_ids:
                raise ValueError('original embedding identity mismatch')
            embeddings = saved['embeddings'].copy()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        encoder = models.vit_b_16(weights=weights)
        encoder.heads.head = nn.Identity()
        encoder.to(device).eval()
        loader = DataLoader(analysis.b.CropDataset(probes,weights.transforms()),batch_size=96,shuffle=False,num_workers=0)
        with torch.inference_mode():
            new_embeddings = np.concatenate([encoder(batch.to(device)).float().cpu().numpy() for batch,_labels,_indices in loader])
        if not np.allclose(new_embeddings[anchor_probe],embeddings[anchor_indices],atol=1e-4,rtol=1e-4):
            raise ValueError('embedding anchor mismatch')
        embeddings[indices] = new_embeddings[changed_probe]
        embedding_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(embedding_path,sample_ids=ids,embeddings=embeddings)
        del encoder
        torch.cuda.empty_cache()
    with np.load(embedding_path) as revised, np.load(source_embedding) as old:
        embeddings = revised['embeddings']
        if not np.isfinite(embeddings).all() or not np.array_equal(embeddings[unchanged],old['embeddings'][unchanged]):
            raise ValueError('revised embedding audit failed')
    independent = []
    for name in common.CLASSES:
        positions = [i for i,r in enumerate(rows) if r['class_name'] == name]
        subset = [rows[i] for i in positions]
        order = analysis.b.kcenter_order(subset,embeddings[positions])
        independent.extend(dict(method='kcenter',class_name=name,rank=rank+1,sample_id=subset[i]['sample_id']) for rank,i in enumerate(order))
    write_csv(STUDY / 'benchmark/tables/model_independent_orders.csv',independent)
    write_json(STUDY / 'prediction_freeze.sha256.json', {str(p):sha256(p) for p in (STUDY/'benchmark').rglob('*') if p.is_file()})


def verify_amendment():
    source = common.verify_freeze()
    for name in ('amendment_input_freeze.sha256.json','prediction_freeze.sha256.json'):
        frozen = json.loads((STUDY/name).read_text())
        changed = [p for p,digest in frozen.items() if sha256(p) != digest]
        if changed:
            raise ValueError(f'{name}: changed files {changed[:5]}')
    return dict(status='pass', original_source=source, amendment_inputs='pass', amended_prediction_freeze='pass')


def configure_analysis():
    analysis.STUDY = STUDY
    analysis.SOURCE = STUDY
    analysis.job_path = lambda backend,domain: STUDY / 'annotations' / backend / domain
    analysis.schema_path = lambda domain: STUDY / 'configs' / ('old_annotation_schema.json' if domain=='old' else 'car_annotation_schema.json')
    analysis.verify_freeze = verify_amendment
    original_contexts = analysis.contexts
    def contexts(labels,smoke=False):
        # Force the technical subset to include both new targets without changing
        # the formal candidate ordering or real classifier cache identities.
        if not smoke:
            return original_contexts(labels,False)
        original_read = analysis.read_csv
        replacement_ids = {r['new_sample_id'] for r in read_csv(REPLACEMENTS/'replacement_mapping.csv')}
        def smoke_read(path):
            rows = original_read(path)
            if Path(path) == STUDY/'manifests/candidate_pool.csv':
                return sorted(rows,key=lambda row:row['sample_id'] not in replacement_ids)
            return rows
        analysis.read_csv = smoke_read
        try:
            return original_contexts(labels,True)
        finally:
            analysis.read_csv = original_read
    analysis.contexts = contexts
    def cache(context,model,seed):
        path = STUDY/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz'
        with np.load(path) as saved:
            full = [r['sample_id'] for r in read_csv(STUDY/'manifests/candidate_pool.csv')]
            if saved['sample_ids'].tolist()!=full or saved['logits'].shape!=(2400,6):
                raise ValueError('amended prediction identity mismatch')
            lookup = {key:i for i,key in enumerate(full)}
            subset = np.asarray([lookup[r['sample_id']] for r in context['rows']])
            return saved['features'][subset].astype(np.float32),saved['logits'][subset].argmax(1)
    analysis.prediction_cache = cache


def pipeline():
    STUDY.mkdir(parents=True,exist_ok=True)
    lock = (STUDY/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows,new_rows = prepare()
    infer_replacements(rows,new_rows)
    configure_analysis()
    note('analysis_smoke')
    analysis.run_analysis(STUDY/'analysis_smoke',True)
    analysis.b.BUDGETS = common.BUDGETS
    analysis.stats.BUDGETS = common.BUDGETS
    note('paired_full_analysis')
    analysis.run_analysis(STUDY/'analysis',False)
    verify_amendment()
    report = STUDY/'analysis/report_zh.md'
    original = report.read_text()
    report.write_text(original.replace('同一2400目标、','同类替换2个线上被拒目标后的同一2400目标、')+
                      '\n## 候选池修订\n\n原2个服务拒绝目标按预先记录条件同类替换；训练/权重不变，'
                      '旧2398条预测和特征保持逐元素相同，新2条在原权重下补推理，所有次序重算。'
                      '原失败记录保留。结论限修订后服务可处理候选池，详见修订协议和替换映射。\n',encoding='utf-8')
    summary = json.loads((STUDY/'analysis/summary.json').read_text())
    with (STUDY/'research_log.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n{now()}：同类替换正式分析完成，4项主检验通过{summary["primary_passed"]}/4。'
                     '新2个目标24份推理、原2398逐元素不变和完整来源冻结验证通过。\n')
    # Refresh the result manifest after adding the explicit amendment paragraph.
    output = STUDY/'analysis'
    write_json(output/'artifact_manifest.sha256.json',{str(p.relative_to(output)):sha256(p) for p in output.rglob('*')
               if p.is_file() and p.name not in ('artifact_manifest.sha256.json','progress.json')})
    write_json(STUDY/'progress.json',dict(status='complete',heartbeat=now(),pid=os.getpid(),
               targets=2400,classifier_runs=24,primary_passed=summary['primary_passed'],report=str(report)))
    print(f'[{now()}] complete: {report}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background',action='store_true')
    args = parser.parse_args()
    if args.background:
        STUDY.mkdir(parents=True,exist_ok=True)
        process_path = STUDY/'background_process.json'
        if process_path.exists():
            prior = json.loads(process_path.read_text())
            proc = Path(f'/proc/{prior["pid"]}/cmdline')
            if proc.exists() and b'run_dual_vlm_replaced_study.py' in proc.read_bytes():
                raise RuntimeError('amended study already running')
        with (STUDY/'pipeline.log').open('a',encoding='utf-8') as stream:
            child = subprocess.Popen([common.PYTHON,'-u',str(Path(__file__).resolve())],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'})
        write_json(process_path,dict(pid=child.pid,started_at=now(),log=str(STUDY/'pipeline.log')))
        print(json.dumps(dict(pid=child.pid,log=str(STUDY/'pipeline.log'))),flush=True)
        return
    try:
        pipeline()
    except Exception as error:
        write_json(STUDY/'failure.json',dict(status='failed',at=now(),error=str(error),traceback=traceback.format_exc()))
        write_json(STUDY/'progress.json',dict(status='failed',heartbeat=now(),error=str(error)))
        raise


if __name__=='__main__':
    main()
