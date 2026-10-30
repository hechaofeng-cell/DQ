#!/usr/bin/env python3
"""COCO-only, old-label five-dimensional scoring and controlled selection study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qe_coco_fine_mpl')
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

import run_category_five_dimension_pilot as pilot
import run_qe_three_strategy_full_experiment as old

POOL = ROOT/'artifacts/sixclass_independent_confirmation_20261004'
DEFAULT_OUT = ROOT/'artifacts/coco_five_dimension_fine_selection_20261015'
CLASSES = old.CLASSES
MODELS, SEEDS = old.MODELS, old.SEEDS
SCHEMES = ('equal_global', 'equal_parts')
GROUPS = ('E0', 'E1', 'E2', 'E3', 'E4')
GROUP_SCHEME = {'E1': 'equal_global', 'E2': 'equal_global', 'E3': 'equal_parts', 'E4': 'equal_parts'}
RUNS = list(itertools.product(MODELS, SEEDS))
SUPPORT = 20


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return pilot.sha256(path)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_csv(path):
    return pilot.read_csv(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty required output: {path}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def note(out, stage, **extra):
    record = dict(time=now(), stage=stage, pid=os.getpid(), complete=False)
    record.update(extra)
    dump(out/'progress.json', record)
    with (out/'research_log.md').open('a') as f:
        f.write('\n'+json.dumps(record, ensure_ascii=False)+'\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def risk_structure(scores, ids, budget):
    """Keep fractional-component uncertainty, rather than rounding width/20."""
    n = len(ids)
    assert n == 400 and budget % 5 == 0
    order = sorted(range(n), key=lambda i: (scores[i]['score_midpoint'], ids[i]))
    tiers = np.zeros(n, dtype=int)
    for tier, portion in enumerate(np.array_split(order, 5), 1):
        tiers[portion] = tier
    cells = defaultdict(list)
    for i, score in enumerate(scores):
        midpoint = int(round(score['score_midpoint']*1e8))
        width = int(round((score['score_upper']-score['score_lower'])*1e8))
        cells[(int(tiers[i]), midpoint, width)].append(i)
    quotas = {}
    for tier in range(1, 6):
        quotas.update(old.largest_remainder({k: v for k, v in cells.items() if k[0] == tier}, budget//5))
    return dict(tiers=tiers, cells=dict(cells), quota=quotas,
                group_by_idx={i: k for k, values in cells.items() for i in values})


def component_specs(features, category, scheme, area, config):
    specs = [('scale', 'target_scale', area, 1., pilot.scale_risk(area, config))]
    fields = [('occlusion', 'occlusion_level', 'occlusion_level', 1.),
              ('truncation', 'truncation_level', 'truncation_level', 1.)]
    if scheme == 'equal_parts' and category != 'car':
        fields += [('visibility', 'head_visibility', 'part_visibility', .5),
                   ('visibility', 'leg_visibility', 'leg_visibility', .5),
                   ('orientation', 'body_orientation', 'body_orientation', .5),
                   ('orientation', 'head_orientation', 'head_orientation', .5)]
    else:
        fields += [('visibility', 'body_visibility', 'body_visibility', 1.),
                   ('orientation', 'body_orientation', 'body_orientation', 1.)]
    for dimension, field, mapping, weight in fields:
        value = pilot.mapped_risk(features[field], config['maps'][mapping], config)
        specs.append((dimension, field, features[field], weight, value))
    return specs


def scale_state(area):
    return 'tiny' if area < .01 else 'small' if area < .03 else 'medium' if area < .1 else 'large'


def state_matrix(rows, features, states):
    result = np.zeros((len(rows), len(states)), dtype=np.uint8)
    for i, (row, values) in enumerate(zip(rows, features)):
        for j, (field, state) in enumerate(states):
            value = scale_state(float(row['bbox_area_ratio'])) if field == 'target_scale' else values[field]
            result[i, j] = state in value if isinstance(value, list) else state == value
    return result


def smoke_tests(config):
    features = dict(head_visibility='partial', leg_visibility='one', body_orientation='back',
                    head_orientation='front', body_visibility='partial',
                    occlusion_level='moderate', truncation_level='light')
    score = pilot.score_sample(.02, features, 'dog', 'equal_parts', config)
    assert math.isclose(score['score_midpoint'], 49.16666666666667)
    features.update(head_visibility='clear', leg_visibility='unknown')
    score = pilot.score_sample(.02, features, 'dog', 'equal_parts', config)
    assert math.isclose(score['visibility_lower'], 0) and math.isclose(score['visibility_upper'], .5)
    assert math.isclose(score['score_upper']-score['score_lower'], 10)
    scores = [dict(score_lower=0., score_midpoint=5., score_upper=10.) for _ in range(400)]
    structure = risk_structure(scores, [f'id{i:03}' for i in range(400)], 200)
    assert len(structure['cells']) == 5
    assert all(key[2] == 10*10**8 for key in structure['cells'])
    assert sum(structure['quota'].values()) == 200
    assert pilot.score_sample(.02, features, 'car', 'equal_parts', config) == pilot.score_sample(.02, features, 'car', 'equal_global', config)
    counts = np.array([2, 1, 0, 0, 0, 0])
    expected = np.mean([len(set(values))/5 for values in itertools.combinations([0, 0, 1], 2)])
    assert np.isclose(old.rarefaction(counts, np.array(2)), expected)
    assert np.isclose(old.rarefaction(np.zeros(6, dtype=int), np.array(0)), 0)
    return dict(status='pass', worked_example=49.16666666666667,
                half_weight_unknown_interval=10, quota_preserved=True,
                car_schemes_identical=True, rarefaction_enumeration_agrees=True)


def prepare(out, config, budget):
    source_rows = read_csv(POOL/'manifests/candidate_pool.csv')
    assert len(source_rows) == 2400 and len({r['sample_id'] for r in source_rows}) == 2400
    assert Counter(r['class_name'] for r in source_rows) == Counter(dict.fromkeys(CLASSES, 400))
    rows = [r for category in CLASSES for r in source_rows if r['class_name'] == category]
    inputs = {r['image_id']: r for domain in ('old', 'car') for r in read_csv(POOL/f'inputs/{domain}.csv')}
    schemas = {d: json.loads((POOL/f'configs/{d}_annotation_schema.json').read_text()) for d in ('old', 'car')}
    shared = json.loads((POOL/'configs/shared_object_schema.json').read_text())
    states = [('target_scale', v) for v in ('tiny', 'small', 'medium', 'large')]
    states += [(f['feature_id'], v) for f in shared['universal_features'] for v in f['possible_values'] if v != 'unknown']
    assert len(states) == 71 and len(set(states)) == 71
    inputs_hash = {str(p.resolve()): sha(p) for p in [Path(__file__), Path(pilot.__file__), Path(old.__file__),
                  ROOT/'configs/category_five_dimension_pilot_v1.json',
                  POOL/'manifests/candidate_pool.csv', POOL/'inputs/old.csv', POOL/'inputs/car.csv',
                  POOL/'configs/shared_object_schema.json',
                  *[POOL/f'configs/{d}_annotation_schema.json' for d in ('old', 'car')]]}
    score_rows, field_rows, originals, tier_rows = [], [], [], []
    prepared = {}
    for ci, category in enumerate(CLASSES):
        subset = rows[ci*400:(ci+1)*400]
        native, scores = [], {scheme: [] for scheme in SCHEMES}
        for row in subset:
            assert int(row['label']) == ci and row['dataset_version'] == 'COCO2017'
            image = Path(row['image_path']); crop = Path(row['crop_path'])
            assert sha(image) == row['image_sha256'] and sha(crop) == row['crop_sha256']
            with Image.open(image) as im:
                binding = pilot.validate_input_binding(row, inputs[row['sample_id']], im.size)
                assert im.size == (int(row['image_width']), int(row['image_height']))
            box = np.asarray(json.loads(row['bbox_norm_xyxy']), dtype=float)
            assert box.shape == (4,) and np.all(box[:2] >= -1e-5) and np.all(box[2:] <= 1+1e-5) and np.all(box[2:] > box[:2])
            area = float(row['bbox_area_ratio'])
            assert abs(float(np.prod(box[2:]-box[:2]))-area) < 1e-6
            domain = 'car' if category == 'car' else 'old'
            ap = POOL/f'annotations/local/{domain}/parsed/{row["sample_id"]}.json'
            annotation = json.loads(ap.read_text())
            features = pilot.validate_label(annotation, binding, schemas[domain])
            inputs_hash[str(ap.resolve())] = sha(ap)
            inputs_hash[str(image)] = row['image_sha256']; inputs_hash[str(crop)] = row['crop_sha256']
            native.append(features)
            originals.append(dict(sample_id=row['sample_id'], class_name=category, features=features))
            for scheme in SCHEMES:
                score = pilot.score_sample(area, features, category, scheme, config)
                scores[scheme].append(score)
                score_rows.append(dict(sample_id=row['sample_id'], class_name=category, scheme=scheme, **score))
                lower, upper = 0., 0.
                for dimension, field, state, weight, value in component_specs(features, category, scheme, area, config):
                    lo, hi = (0., 1.) if value is None else (value, value)
                    lower += 20*weight*lo; upper += 20*weight*hi
                    field_rows.append(dict(sample_id=row['sample_id'], class_name=category, scheme=scheme,
                        dimension=dimension, field=field, state=state, within_dimension_weight=weight,
                        lower=lo, midpoint=(lo+hi)/2, upper=hi, known=value is not None,
                        contribution_lower=20*weight*lo, contribution_midpoint=20*weight*(lo+hi)/2,
                        contribution_upper=20*weight*hi))
                assert np.allclose([lower, (lower+upper)/2, upper], [score[f'score_{k}'] for k in ('lower','midpoint','upper')], atol=1e-8)
            if category == 'car':
                assert scores['equal_global'][-1] == scores['equal_parts'][-1]
        ids = [r['sample_id'] for r in subset]
        local_fields = {'body_visibility', 'body_orientation'} if category == 'car' else {'head_visibility', 'leg_visibility', 'head_orientation'}
        local_states = [(f['feature_id'], v) for f in schemas['car' if category == 'car' else 'old']['universal_features']
                        if f['feature_id'] in local_fields for v in f['possible_values'] if v != 'unknown']
        combined_states = states + [s for s in local_states if s not in states]
        m = state_matrix(subset, native, states)
        structures = {s: risk_structure(scores[s], ids, budget) for s in SCHEMES}
        for scheme in SCHEMES:
            for i, sid in enumerate(ids):
                tier_rows.append(dict(sample_id=sid, class_name=category, scheme=scheme,
                                      tier=int(structures[scheme]['tiers'][i]), **scores[scheme][i]))
        prepared[category] = dict(rows=subset, ids=ids, sample_ids=np.asarray(ids), native=native, scores=scores, matrix=m,
            local_states=local_states, local_matrix=state_matrix(subset, native, local_states),
            condition_states=combined_states, condition_matrix=state_matrix(subset, native, combined_states), structures=structures)
        note(out, 'prepare', category=category, targets=(ci+1)*400)
    candidate_hash = {r['crop_sha256'] for r in rows}; candidate_images = {r['image_sha256'] for r in rows}
    overlaps = {}
    for name in ('train', 'eval'):
        path = POOL/f'manifests/{name}.csv'
        inputs_hash[str(path)] = sha(path)
        references = read_csv(path)
        ref_hashes = {r['crop_sha256'] for r in references if r.get('crop_sha256')}
        ref_images = {r['image_sha256'] for r in references if r.get('image_sha256')}
        overlaps[name] = dict(crop_hash_overlap=len(candidate_hash & ref_hashes), image_hash_overlap=len(candidate_images & ref_images))
        assert not candidate_hash & ref_hashes and not candidate_images & ref_images
    write_csv(out/'manifest.csv', rows)
    write_csv(out/'scores.csv', score_rows); write_csv(out/'field_scores.csv', field_rows); write_csv(out/'tiers.csv', tier_rows)
    with (out/'native_features.jsonl').open('w') as f:
        for row in originals:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    dump(out/'state_dictionary.json', dict(main_states=states, support_cap=20,
         local_states={c: z['local_states'] for c, z in prepared.items()}, unknown_counts_as_coverage=False))
    dump(out/'input_hashes.json', inputs_hash)
    dump(out/'preparation_audit.json', dict(status='pass', targets=2400, classes=6, annotations=2400,
        all_target_bindings_boxes_and_original_crop_hashes_verified=True, independently_summed_score_contributions=True,
        car_global_and_parts_identical=True, reference_overlaps=overlaps, new_vlm_calls=0))
    return prepared, rows, source_rows


def select(out, prepared, budget, repeats, seed):
    choices = {g: [] for g in GROUPS}
    traces, quota_rows = [], []
    for ci, category in enumerate(CLASSES):
        z = prepared[category]
        index, trace = old.greedy(z, budget)
        choices['E0'].append(index[None])
        traces += [dict(group='E0', class_name=category, **r) for r in trace]
        priority = np.random.default_rng(seed+ci*1009).random((repeats, 400))
        for scheme, random_group, combo_group in [('equal_global', 'E1', 'E2'), ('equal_parts', 'E3', 'E4')]:
            structure = z['structures'][scheme]
            random_parts = []
            for key in sorted(structure['cells']):
                members = np.asarray(structure['cells'][key]); quota = structure['quota'][key]
                quota_rows.append(dict(class_name=category, scheme=scheme, tier=key[0],
                    risk_midpoint=key[1]/1e8, uncertainty_width=key[2]/1e8, candidates=len(members), quota=quota))
                if quota:
                    order = np.argsort(priority[:, members], axis=1, kind='stable')[:, :quota]
                    random_parts.append(members[order])
            random = np.concatenate(random_parts, axis=1).astype(np.int16)
            combo, trace = old.greedy(z, budget, structure)
            assert random.shape == (repeats, budget) and np.all(np.diff(np.sort(random, axis=1), axis=1) > 0)
            for key, members in structure['cells'].items():
                assert np.all(np.isin(random, members).sum(1) == structure['quota'][key])
                assert int(np.isin(combo, members).sum()) == structure['quota'][key]
            choices[random_group].append(random); choices[combo_group].append(combo[None])
            traces += [dict(group=combo_group, class_name=category, **r) for r in trace]
        note(out, 'selection', category=category, completed_classes=ci+1)
    choices = {g: np.stack(parts, axis=1) for g, parts in choices.items()}
    np.savez_compressed(out/'selection.npz', sample_ids=np.array([prepared[c]['ids'] for c in CLASSES]), **choices)
    write_csv(out/'risk_cell_quotas.csv', quota_rows); write_csv(out/'greedy_trace.csv', traces)
    selected_rows = []
    for group, selections in choices.items():
        for ci, category in enumerate(CLASSES):
            z = prepared[category]
            for rank, index in enumerate(selections[0, ci], 1):
                row = z['rows'][index]
                selected_rows.append(dict(group=group, reference_repetition=0,
                    random_reference_only=group in ('E1', 'E3'), rank_within_class=rank, **row))
    write_csv(out/'selected_targets_reference.csv', selected_rows)
    dump(out/'selection_audit.json', dict(status='pass', groups=5, per_set_targets=budget*6,
         per_class=budget, random_repeats=repeats, no_repeated_targets_within_set=True,
         exact_midpoint_and_uncertainty_cell_quotas_equal=True, fractional_unknown_widths_retained=True,
         classifier_predictions_read=False, selection_sha256=sha(out/'selection.npz')))
    return choices


def selection_summary(out, prepared, choices):
    summary, by_class, state_rows, weights = [], [], [], {}
    for group, selections in choices.items():
        n = len(selections)
        presence = np.zeros(n); support = np.zeros(n); local_presence = np.zeros(n)
        profiles = {scheme: {k: np.zeros(n) for k in ('risk', 'width', 'unknown')} for scheme in SCHEMES}
        weights[group] = []
        for ci, category in enumerate(CLASSES):
            z = prepared[category]; indices = selections[:, ci]
            inclusion = np.bincount(indices.ravel(), minlength=400)/n
            assert math.isclose(float(inclusion.sum()), selections.shape[-1])
            weights[group].append(inclusion)
            counts = np.concatenate([z['matrix'][indices[a:a+64]].sum(1) for a in range(0, n, 64)])
            p, s = (counts > 0).mean(1), np.minimum(counts/SUPPORT, 1).mean(1)
            lp = np.concatenate([(z['local_matrix'][indices[a:a+64]].sum(1) > 0).mean(1) for a in range(0, n, 64)])
            presence += p/6; support += s/6; local_presence += lp/6
            by_class.append(dict(group=group, class_name=category, presence_coverage=float(p.mean()),
                                 support_completion=float(s.mean()), local_presence_coverage=float(lp.mean())))
            for j, (field, state) in enumerate(json.loads((out/'state_dictionary.json').read_text())['main_states']):
                state_rows.append(dict(group=group, class_name=category, field=field, state=state,
                                       mean_support=float(counts[:, j].mean()), missing_probability=float((counts[:, j] == 0).mean())))
            for scheme in SCHEMES:
                scores = z['scores'][scheme]
                mid = np.array([x['score_midpoint'] for x in scores]); width = np.array([x['score_upper']-x['score_lower'] for x in scores])
                profiles[scheme]['risk'] += mid[indices].mean(1)/6
                profiles[scheme]['width'] += width[indices].mean(1)/6
                profiles[scheme]['unknown'] += (width[indices] > 1e-8).mean(1)/6
        entry = dict(group=group, repetitions=n, total_targets=int(selections.shape[-1]*6),
                     presence_coverage=float(presence.mean()), support_completion=float(support.mean()),
                     local_presence_coverage=float(local_presence.mean()),
                     presence_selection_p025=float(np.quantile(presence, .025)), presence_selection_p975=float(np.quantile(presence, .975)),
                     support_selection_p025=float(np.quantile(support, .025)), support_selection_p975=float(np.quantile(support, .975)))
        for scheme in SCHEMES:
            entry.update({scheme+'_'+k: float(v.mean()) for k, v in profiles[scheme].items()})
        summary.append(entry)
    for random, combo in (('E1','E2'), ('E3','E4')):
        scheme = GROUP_SCHEME[random]
        a, b = (next(r for r in summary if r['group'] == g) for g in (random, combo))
        assert all(np.isclose(a[scheme+'_'+k], b[scheme+'_'+k], atol=1e-7) for k in ('risk', 'width', 'unknown'))
    write_csv(out/'dataset_metrics.csv', summary); write_csv(out/'dataset_class_metrics.csv', by_class)
    write_csv(out/'state_support.csv', state_rows)
    return weights


def predictions(out, rows, source_rows):
    assert json.loads((out/'selection_audit.json').read_text())['classifier_predictions_read'] is False
    original_ids = [r['sample_id'] for r in source_rows]
    target_ids = [r['sample_id'] for r in rows]
    index = {sid: i for i, sid in enumerate(original_ids)}
    reorder = [index[sid] for sid in target_ids]
    raw, hashes = [], {}
    for model, seed in RUNS:
        folder = POOL/f'benchmark/runs/{model}/{seed}'
        path = folder/'candidate_outputs.npz'
        with np.load(path, allow_pickle=False) as z:
            assert z['sample_ids'].tolist() == original_ids and z['logits'].shape == (2400, 6)
            assert np.isfinite(z['logits']).all()
            pred = z['logits'].argmax(1)
        csv_path = folder/'candidate_predictions.csv'; reference = read_csv(csv_path)
        assert [r['sample_id'] for r in reference] == original_ids
        assert all(int(r['true_label']) == int(t['label']) and int(r['predicted_label']) == int(v)
                   for r, t, v in zip(reference, source_rows, pred))
        hashes[str(path)] = sha(path); hashes[str(csv_path)] = sha(csv_path)
        raw.append(pred[reorder])
    pred = np.asarray(raw, dtype=np.int8)
    np.savez_compressed(out/'predictions.npz', sample_ids=np.array(target_ids), predictions=pred,
                        models=np.array([r[0] for r in RUNS]), seeds=np.array([r[1] for r in RUNS]))
    dump(out/'prediction_cache_audit.json', dict(status='pass', targets=2400, runs=24,
        ids_labels_original_logits_and_prediction_csv_all_match=True, new_classifier_inference=False,
        retraining=False, source_hashes=hashes))
    return pred


def evaluate(out, prepared, choices, weights, pred):
    model_rows, paired, scoring, scoring_class, tier_rows, condition_rows = [], [], [], [], [], []
    wrong_counts, metrics_raw, independent_checks = {}, {}, 0
    for group, indices in choices.items():
        global_indices = (indices + 400*np.arange(6)[None, :, None]).reshape(len(indices), -1)
        group_wrong, group_f1, group_errors = [], [], []
        for run, (model, seed) in enumerate(RUNS):
            metrics, wrong = old.model_metrics(pred[run], global_indices, indices.shape[-1])
            group_wrong.append(wrong); group_f1.append(metrics['macro_f1']); group_errors.append(metrics['errors'])
            for rep in sorted({0, len(indices)//2, len(indices)-1}):
                true = np.repeat(np.arange(6), indices.shape[-1])
                direct = pred[run, global_indices[rep]]
                assert np.isclose(f1_score(true, direct, average='macro', labels=list(range(6)), zero_division=0), metrics['macro_f1'][rep])
                assert np.count_nonzero(true != direct) == metrics['errors'][rep]
                independent_checks += 1
            model_rows.append(dict(group=group, model=model, seed=seed, repetitions=len(indices),
                **{k: float(np.mean(v)) for k, v in metrics.items()},
                macro_f1_selection_p025=float(np.quantile(metrics['macro_f1'], .025)),
                macro_f1_selection_p975=float(np.quantile(metrics['macro_f1'], .975))))
        wrong_counts[group] = np.stack(group_wrong)
        metrics_raw[group] = dict(f1=np.stack(group_f1), errors=np.stack(group_errors))
    for fixed, random in (('E2','E1'), ('E4','E3')):
        a = wrong_counts[fixed][:, 0]; b = wrong_counts[random]
        q = np.minimum(a.sum(-1)[:, None], b.sum(-1))
        left = old.rarefaction(np.broadcast_to(a[:, None], b.shape), q).mean(-1)
        right = old.rarefaction(b, q).mean(-1)
        delta = left-right
        for run, (model, seed) in enumerate(RUNS):
            paired.append(dict(contrast=fixed+'-'+random, model=model, seed=seed,
                equal_error_breadth_delta=float(delta[run].mean()),
                selection_p025=float(np.quantile(delta[run], .025)), selection_p975=float(np.quantile(delta[run], .975)),
                evaluable_class_fraction=float((q[run] > 0).mean()),
                macro_f1_delta=float(metrics_raw[fixed]['f1'][run, 0]-metrics_raw[random]['f1'][run].mean()),
                error_delta=float(metrics_raw[fixed]['errors'][run, 0]-metrics_raw[random]['errors'][run].mean())))
    for ci, category in enumerate(CLASSES):
        z = prepared[category]; p = pred[:, ci*400:(ci+1)*400]; correct = p == ci
        all_known = np.array([a['fully_known'] and b['fully_known'] for a, b in zip(z['scores']['equal_global'], z['scores']['equal_parts'])])
        for scheme in SCHEMES:
            scores = np.array([s['score_midpoint'] for s in z['scores'][scheme]])
            tiers = z['structures'][scheme]['tiers']
            for cohort, mask in [('all', np.ones(400, dtype=bool)), ('common_complete', all_known)]:
                for run, (model, seed) in enumerate(RUNS):
                    wrong = ~correct[run, mask]
                    auc = roc_auc_score(wrong, scores[mask]) if len(np.unique(wrong)) == 2 else np.nan
                    endpoints = [correct[run, mask & (tiers == t)] for t in (1, 5)]
                    gap = float(endpoints[0].mean()-endpoints[1].mean()) if all(len(a) for a in endpoints) else np.nan
                    scoring_class.append(dict(class_name=category, scheme=scheme, cohort=cohort, model=model, seed=seed,
                        targets=int(mask.sum()), error_auroc=float(auc), q5_minus_q1_error_gap=gap,
                        tiers_defined_on_full_class_pool=True))
        all_weights = dict(POOL=np.ones(400), **{g: w[ci] for g, w in weights.items()})
        for group, inclusion in all_weights.items():
            weighted_m = z['condition_matrix'] * inclusion[:, None]
            supports = weighted_m.sum(0); correct_support = correct @ weighted_m
            refs = (correct*inclusion).sum(1)/inclusion.sum()
            for j, (field, state) in enumerate(z['condition_states']):
                for run, (model, seed) in enumerate(RUNS):
                    value = float(correct_support[run, j]/supports[j]) if supports[j] > 0 else np.nan
                    condition_rows.append(dict(group=group, model=model, seed=seed, class_name=category,
                        field=field, state=state, mean_support=float(supports[j]), conditional_recall=value,
                        class_recall=float(refs[run]), recall_delta=value-float(refs[run]),
                        support_at_least_20=bool(supports[j] >= 20-1e-8),
                        estimator='pooled_across_selection_repetitions; support_is_mean_per_set'))
            schemes = SCHEMES if group in ('POOL','E0') else (GROUP_SCHEME[group],)
            for scheme in schemes:
                tiers = z['structures'][scheme]['tiers']
                for tier in range(1, 6):
                    w = inclusion*(tiers == tier); n = float(w.sum())
                    for run, (model, seed) in enumerate(RUNS):
                        tier_rows.append(dict(group=group, scheme=scheme, class_name=category, tier=tier,
                            model=model, seed=seed, mean_support=n,
                            recall=float((correct[run]*w).sum()/n) if n else np.nan))
        note(out, 'evaluation', category=category, completed_classes=ci+1)
    for scheme, cohort, model, seed in itertools.product(SCHEMES, ('all','common_complete'), MODELS, SEEDS):
        subset = [r for r in scoring_class if (r['scheme'],r['cohort'],r['model'],r['seed']) == (scheme,cohort,model,seed)]
        scoring.append(dict(scheme=scheme, cohort=cohort, model=model, seed=seed,
            error_auroc=float(np.nanmean([r['error_auroc'] for r in subset])),
            q5_minus_q1_error_gap=float(np.nanmean([r['q5_minus_q1_error_gap'] for r in subset])),
            evaluable_classes=sum(np.isfinite(r['error_auroc']) for r in subset)))
    tests = []
    architecture_values = {}
    for contrast in ('E4-E3','E2-E1'):
        architecture_values[contrast] = [float(np.mean([r['equal_error_breadth_delta'] for r in paired if r['contrast'] == contrast and r['model'] == model])) for model in MODELS]
    architecture_values['local_minus_global_auc'] = []
    for model in MODELS:
        vals = {s: np.mean([r['error_auroc'] for r in scoring if r['model'] == model and r['cohort'] == 'all' and r['scheme'] == s]) for s in SCHEMES}
        architecture_values['local_minus_global_auc'].append(float(vals['equal_parts']-vals['equal_global']))
    architecture_values['local_minus_global_selection_gain'] = (np.array(architecture_values['E4-E3'])-architecture_values['E2-E1']).tolist()
    for name, vals in architecture_values.items():
        tests.append(dict(contrast=name, mean_delta=float(np.mean(vals)), positive_architectures=sum(v > 0 for v in vals),
                          architectures=8, signflip_two_sided_p=old.exact_signflip(vals), correction_family=4))
    for row, value in zip(tests, old.holm([r['signflip_two_sided_p'] for r in tests])):
        row['holm_p'] = float(value)
    for name, data in [('model_run_metrics',model_rows), ('paired_model_run_metrics',paired),
                       ('scoring_metrics',scoring), ('scoring_class_metrics',scoring_class),
                       ('condition_metrics',condition_rows), ('tier_model_metrics',tier_rows), ('tests',tests)]:
        write_csv(out/(name+'.csv'), data)
    np.savez_compressed(out/'model_raw_arrays.npz', **{g+'_wrong': v for g,v in wrong_counts.items()})
    dump(out/'evaluation_audit.json', dict(status='pass', sklearn_independent_metric_checks=independent_checks,
        all_24_runs_retained=True, main_tests_predeclared=4, unknown_targets_retained=True,
        zero_error_rarefaction_returns_zero_and_evaluable_fraction_reported=True,
        conditional_random_recall_is_pooled_ratio_not_mean_of_repetition_ratios=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/category_five_dimension_pilot_v1.json').read_text())
    smoke = smoke_tests(config)
    if args.smoke:
        print(json.dumps(smoke)); return
    assert (args.budget, args.repeats) == (200, 1000), 'Current approved protocol is per-class200 and random1000.'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Output already contains a protocol; choose a separate run directory rather than overwrite results.')
    protocol = dict(version='coco_old_native_equal_parts_selection_v1', created_at=now(), source=str(POOL),
        annotation='local existing native labels', targets=2400, per_class=400, models=list(MODELS), seeds=list(SEEDS),
        schemes=list(SCHEMES), groups=dict(E0='unconstrained coverage', E1='global risk random', E2='global risk coverage',
            E3='local risk random', E4='local risk coverage'), per_class_budget=200, risk_tier_quota=[40]*5,
        random_repeats=1000, seed=args.seed, main_coverage='71 fixed nonunknown states per class, support20',
        risk_cells='tier, midpoint rounded1e-8, uncertainty_width rounded1e-8; exact largest remainder quotas',
        unknown='keep all targets, midpoint ranking with interval; common-complete sensitivity',
        selection_randomness='same target-level uniform priorities for both scorers within class/repetition',
        tests_predeclared=['E4-E3 breadth','E2-E1 breadth','local-global full-pool class-macro errorAUROC','local-global selection-breadth gain'],
        statistical_unit='eight architectures averaged three training seeds; two-sided exact signflip, Holm4',
        legacy_correction='old merged-pool18-test family unchanged; this is a distinct exploratory COCO protocol',
        causal_claim=False, retraining=False, new_vlm_calls=False, new_classifier_inference=False,
        rule='all selection and parameters recorded before reading model predictions; no tuning after results',
        command=[sys.executable, *sys.argv], score_config=config, python=sys.version)
    dump(out/'protocol.json', protocol); dump(out/'smoke_tests.json', smoke)
    (out/'research_log.md').write_text('# COCO local five-dimensional scoring experiment\n\n'
        'Questions: local scoring vs global scoring; coverage selection vs same-cell-quota random. '
        'Fixed factors: old COCO pool, native local labels, 71-state denominator, support20, per-class200, '
        '1000 random repetitions, 24 cached models. Success: independently validated outputs; gains reported with failures. '
        'Stop on identity, hash, field mapping or quota mismatch. No score tuning or external-source expansion in this run.\n')
    try:
        note(out, 'started')
        prepared, rows, source_rows = prepare(out, config, args.budget)
        choices = select(out, prepared, args.budget, args.repeats, args.seed)
        weights = selection_summary(out, prepared, choices)
        note(out, 'selection_complete_before_predictions')
        pred = predictions(out, rows, source_rows)
        evaluate(out, prepared, choices, weights, pred)
        note(out, 'reporting')
        subprocess.run([sys.executable, str(ROOT/'scripts/report_coco_fine_coverage_20261015.py'), '--input', str(out)], check=True)
        note(out, 'complete', complete=True, targets=2400, selected_per_set=1200, groups=5,
             random_repetitions=1000, model_runs=24, report=str(out/'report_zh.md'))
    except BaseException as exc:
        note(out, 'failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
