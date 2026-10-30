#!/usr/bin/env python3
"""Paired cached ranking sensitivity and binary official-reference validation."""
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import kendalltau, spearmanr, rankdata
from statsmodels.stats.proportion import proportion_confint

from automatic_strengthening_common import A, labels_for
from analyze_sixclass_difficulty import risk
from dual_vlm_common import CLASSES, MODELS, SEEDS, read_csv, write_csv, write_json, sha256, now

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/ranking_attribute_validation_20261005'
PROTOCOL = ROOT / 'docs/ranking_attribute_validation_protocol_20261005.md'
POOLS = {'B': ROOT / 'artifacts/sixclass_independent_confirmation_20261004',
         'C': ROOT / 'artifacts/sixclass_voc_external_confirmation_20261005',
         'D': ROOT / 'artifacts/objects365_independent_confirmation_20261005'}
CONFIGS = list(itertools.product(('local', 'online'), ('lower', 'midpoint', 'upper')))
RUNS = list(itertools.product(MODELS, SEEDS))
DRAWS = 2000
LABELS = dict(zip(CLASSES, range(6)))
OLD_REFERENCE = ROOT / 'artifacts/automatic_strengthening_20261005/automatic_audit/official_attribute_reference.csv'
METRICS = ('agreement', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'kappa')


def note(stage, **kwargs):
    value = dict(status='running', stage=stage, heartbeat=now(), **kwargs)
    write_json(OUT / 'progress.json', value)
    line = json.dumps(value, ensure_ascii=False)
    print(line, flush=True)
    with (OUT / 'run.log').open('a') as stream:
        stream.write(line + '\n')


def ranks(values):
    return rankdata(-values, axis=-1, method='average')


def top_two_shares(values):
    """Uniformly allocate remaining top-two slots at a boundary tie."""
    cutoff = np.sort(values, axis=-1)[..., -2: -1]
    above = values > cutoff
    same = values == cutoff
    return above.astype(float) + same * (2 - above.sum(-1, keepdims=True)) / same.sum(-1, keepdims=True)


def ci(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return tuple(float(x) for x in np.percentile(finite, [2.5, 97.5])) if len(finite) else (None, None)


def macro_f1(confusion):
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(-2) + confusion.sum(-1)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0).mean(-1)


def stratified_weights(y, rng):
    weights = np.zeros((DRAWS, len(y)), dtype=np.int32)
    for c in range(6):
        indices = np.flatnonzero(y == c)
        assert len(indices)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=DRAWS)
    return weights


def cache(rows, path):
    ids = [r['sample_id'] for r in rows]
    assert len(ids) == len(set(ids))
    y = np.array([LABELS[r['class_name']] for r in rows])
    assert all(int(r['label']) == LABELS[r['class_name']] for r in rows)
    predictions = []
    for model, seed in RUNS:
        with np.load(path / 'benchmark/runs' / model / str(seed) / 'candidate_outputs.npz') as saved:
            assert saved['sample_ids'].tolist() == ids
            assert np.isfinite(saved['logits']).all() and saved['logits'].shape == (len(rows), 6)
            predictions.append(saved['logits'].argmax(1))
    return ids, y, np.array(predictions)


def ranking_analysis(pool, path, rng):
    rows = read_csv(path / 'manifests/candidate_pool.csv')
    ids, y, predictions = cache(rows, path)
    labels, failures = labels_for(rows, path)
    assert not failures
    assignments = np.zeros((6, len(rows)), dtype=int)
    sample_rows = []
    masks = []
    for configuration, (source, mode) in enumerate(CONFIGS):
        score = [risk(row, labels[source][iid])[f'h_{mode}'] for row, iid in zip(rows, ids)]
        for c in range(6):
            indices = np.flatnonzero(y == c)
            assert len(indices) % 5 == 0
            ordered = sorted(indices, key=lambda i: (score[i], ids[i]))
            for tier, portion in enumerate(np.split(np.array(ordered), 5), 1):
                assignments[configuration, portion] = tier
        masks.append(assignments[configuration] == 5)
        sample_rows.extend(dict(pool=pool, source=source, unknown_mode=mode, sample_id=iid,
            class_name=rows[i]['class_name'], H=score[i], tier=int(assignments[configuration, i])) for i, iid in enumerate(ids))
    masks = np.array(masks)
    # Sparse image-to-confusion-cell mapping avoids treating shared images as independent.
    image_index, output_index = [], []
    for configuration, mask in enumerate(masks):
        indices = np.flatnonzero(mask)
        for run, prediction in enumerate(predictions):
            image_index.extend(indices.tolist())
            output_index.extend((configuration * 24 * 36 + run * 36 + y[indices] * 6 + prediction[indices]).tolist())
    matrix = coo_matrix((np.ones(len(image_index)), (output_index, image_index)),
                        shape=(6 * 24 * 36, len(rows))).tocsr()
    point_confusion = np.asarray(matrix.sum(1)).reshape(6, 24, 6, 6)
    run_scores = macro_f1(point_confusion)
    point = run_scores.reshape(6, 8, 3).mean(-1)
    # Independently compare all configurations/runs with the previous raw full-stratum metrics.
    old = pd.read_csv(path / 'confirmation/tables/full_stratum_metrics.csv')
    for j, (source, mode) in enumerate(CONFIGS):
        for run, (model, seed) in enumerate(RUNS):
            match = old[(old.source == source) & (old.unknown_mode == mode) & (old.tier == 5)
                        & (old.model == model) & (old.seed == seed)]
            assert len(match) == 1 and np.isclose(run_scores[j, run], match.macro_f1.iloc[0], atol=1e-12)
    weights = stratified_weights(y, rng)
    confusion = np.asarray(matrix @ weights.T).T.reshape(DRAWS, 6, 24, 6, 6)
    valid = (confusion.sum(-1) > 0).all(axis=(1, 2, 3))
    assert valid.sum() > DRAWS * .99
    # Verify a resampling draw against weighted direct bincount, not only the sparse implementation.
    for configuration, mask in enumerate(masks):
        for run in (0, 23):
            direct = np.bincount(y[mask] * 6 + predictions[run, mask], weights=weights[0, mask], minlength=36).reshape(6, 6)
            assert np.array_equal(direct, confusion[0, configuration, run])
    boot = macro_f1(confusion[valid]).reshape(valid.sum(), 6, 8, 3).mean(-1)
    ranked = ranks(point)
    boot_rank = ranks(boot)
    top = top_two_shares(boot)
    rank_rows, pair_rows, between_rows, seed_rows = [], [], [], []
    for j, (source, mode) in enumerate(CONFIGS):
        minimum = min(int(masks[j, y == c].sum()) for c in range(6))
        assert minimum >= 5
        for model_i, model in enumerate(MODELS):
            lo, hi = ci(boot[:, j, model_i])
            rank_lo, rank_hi = ci(boot_rank[:, j, model_i])
            rank_rows.append(dict(pool=pool, source=source, unknown_mode=mode, model=model,
                q5_n=int(masks[j].sum()), per_class_q5_n=minimum, macro_f1=float(point[j, model_i]),
                rank=float(ranked[j, model_i]), f1_ci95_low=lo, f1_ci95_high=hi,
                rank_ci95_low=rank_lo, rank_ci95_high=rank_hi,
                top2_frequency=float(top[:, j, model_i].mean()), bootstrap_valid=int(valid.sum())))
        for left, right in itertools.combinations(range(8), 2):
            delta = boot[:, j, left] - boot[:, j, right]
            lo, hi = ci(delta)
            between_rows.append(dict(pool=pool, source=source, unknown_mode=mode,
                model_left=MODELS[left], model_right=MODELS[right], mean_delta=float(point[j, left] - point[j, right]),
                ci95_low=lo, ci95_high=hi, bootstrap_positive_frequency=float((delta > 0).mean()),
                descriptive_interval_excludes_zero=bool(lo > 0 or hi < 0), no_multiplicity_confirmation=True))
        for seed_i, seed in enumerate(SEEDS):
            seed_values = run_scores[j].reshape(8, 3)[:, seed_i]
            seed_ranks = ranks(seed_values)
            seed_rows.extend(dict(pool=pool, source=source, unknown_mode=mode, seed=seed, model=model,
                macro_f1=float(seed_values[i]), rank=float(seed_ranks[i])) for i, model in enumerate(MODELS))
    for a, b in itertools.combinations(range(6), 2):
        ta = float(kendalltau(point[a], point[b]).statistic)
        rho = float(spearmanr(point[a], point[b]).statistic)
        tau_samples = [kendalltau(x[a], x[b]).statistic for x in boot]
        lo, hi = ci(tau_samples)
        a_top = set(np.flatnonzero(ranked[a] <= 2))
        b_top = set(np.flatnonzero(ranked[b] <= 2))
        intersection = int((masks[a] & masks[b]).sum())
        pair_rows.append(dict(pool=pool, source_left=CONFIGS[a][0], mode_left=CONFIGS[a][1],
            source_right=CONFIGS[b][0], mode_right=CONFIGS[b][1], kendall_tau=ta, spearman=rho,
            tau_ci95_low=lo, tau_ci95_high=hi, top2_overlap=len(a_top & b_top),
            max_rank_change=float(np.abs(ranked[a] - ranked[b]).max()), q5_overlap_n=intersection,
            q5_jaccard=intersection / int((masks[a] | masks[b]).sum())))
    summary = dict(pool=pool, configurations=6, targets=len(rows), q5_per_class=minimum,
        tau_min=min(r['kendall_tau'] for r in pair_rows), tau_max=max(r['kendall_tau'] for r in pair_rows),
        same_top2_pairs=sum(r['top2_overlap'] == 2 for r in pair_rows), configuration_pairs=15,
        max_rank_change=max(r['max_rank_change'] for r in pair_rows), bootstrap_valid=int(valid.sum()))
    np.savez_compressed(OUT / 'bootstrap' / f'{pool}_rank_scores.npz', scores=boot, ranks=boot_rank,
        models=np.array(MODELS), sources=np.array([s for s, m in CONFIGS]), modes=np.array([m for s, m in CONFIGS]),
        valid_draw_indices=np.flatnonzero(valid), point_scores=point)
    return sample_rows, rank_rows, pair_rows, between_rows, seed_rows, summary


def binary_metrics(counts):
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts)[..., :4], -1, 0)
    n = tn + fp + fn + tp
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b > 0)
    agreement = divide(tp + tn, n)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    chance = divide((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn), n * n)
    return dict(agreement=agreement, precision=divide(tp, tp + fp), recall=recall,
        f1=divide(2 * tp, 2 * tp + fp + fn), specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2, kappa=divide(agreement - chance, 1 - chance))


def attribute_analysis(rng):
    rows = read_csv(A / 'manifests/candidate_pool.csv')
    labels, failures = labels_for(rows, A)
    assert not failures
    rows = [r for r in rows if r.get('image_id') and len(r['image_id']) == 16
            and all(c in '0123456789abcdef' for c in r['image_id'].lower())
            and r.get('is_depiction') is not None and r.get('is_occluded') in ('0', '1')
            and r.get('is_truncated') in ('0', '1')]
    assert len(rows) == 2289
    assert len({r['image_id'] for r in rows}) == len(rows), 'bootstrap unit must be source image'
    y = np.array([LABELS[r['class_name']] for r in rows])
    weights = stratified_weights(y, rng)
    sample_rows, result_rows, differences = [], [], []
    old = pd.read_csv(OLD_REFERENCE)
    pooled = {}
    for field, official in [('occlusion_level', 'is_occluded'), ('truncation_level', 'is_truncated')]:
        for source in ('local', 'online'):
            true = np.array([int(r[official]) for r in rows])
            states = [labels[source][r['sample_id']][field] for r in rows]
            codes = np.array([4 + a if state == 'unknown' else 2 * a + int(state != 'none') for a, state in zip(true, states)])
            sample_rows.extend(dict(sample_id=r['sample_id'], image_id=r['image_id'], class_name=r['class_name'],
                source=source, field=field, official_reference=int(a), vlm_state=state,
                mapped_binary=None if state == 'unknown' else int(state != 'none')) for r, a, state in zip(rows, true, states))
            class_points, class_boot = [], []
            for c, name in enumerate(CLASSES):
                mask = y == c
                counts = np.bincount(codes[mask], minlength=6)
                onehot = np.eye(6)[codes[mask]]
                boot_counts = weights[:, mask] @ onehot
                expected = old[(old.pool == 'A') & (old.source == source) & (old.class_name == name) & (old.feature == field)].iloc[0]
                for value, key in zip(counts[:4], ('tn', 'fp', 'fn', 'tp')):
                    assert int(value) == int(expected[key])
                assert int(counts[4:].sum()) == int(expected.unknown)
                result_rows.append(attribute_row(source, field, name, 'class', counts, boot_counts))
                class_points.append(counts)
                class_boot.append(boot_counts)
            point_total = np.sum(class_points, axis=0)
            boot_total = np.sum(class_boot, axis=0)
            result_rows.append(attribute_row(source, field, 'all', 'target_weighted', point_total, boot_total))
            pooled[source, field] = binary_metrics(boot_total)
            metrics_per_class = [binary_metrics(c) for c in class_points]
            boots_per_class = [binary_metrics(c) for c in class_boot]
            macro = dict(source=source, field=field, class_name='all', aggregation='class_equal',
                planned=int(point_total.sum()), known=int(point_total[:4].sum()), unknown=int(point_total[4:].sum()))
            for metric in METRICS:
                point = float(np.mean([m[metric] for m in metrics_per_class]))
                sample = np.mean([m[metric] for m in boots_per_class], axis=0)
                lo, hi = ci(sample)
                macro.update({metric:point, metric+'_ci95_low':lo, metric+'_ci95_high':hi})
            result_rows.append(macro)
        for metric in METRICS:
            delta = pooled['online', field][metric] - pooled['local', field][metric]
            lo, hi = ci(delta)
            differences.append(dict(field=field, metric=metric, contrast='online_minus_local',
                aggregation='target_weighted', mean_bootstrap_delta=float(np.nanmean(delta)),
                ci95_low=lo, ci95_high=hi, bootstrap_valid=int(np.isfinite(delta).sum()), descriptive=True))
    return sample_rows, result_rows, differences


def attribute_row(source, field, name, aggregation, counts, boot):
    known = int(counts[:4].sum())
    lo, hi = proportion_confint(int(counts[0] + counts[3]), known, method='wilson')
    row = dict(source=source, field=field, class_name=name, aggregation=aggregation,
        planned=int(counts.sum()), known=known, unknown=int(counts[4:].sum()),
        tn=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]), tp=int(counts[3]),
        wilson_agreement_ci95_low=float(lo), wilson_agreement_ci95_high=float(hi))
    point = binary_metrics(counts)
    intervals = binary_metrics(boot)
    for metric in METRICS:
        lo, hi = ci(intervals[metric])
        row.update({metric:float(point[metric]), metric+'_ci95_low':lo, metric+'_ci95_high':hi})
    return row


def scale_check():
    from run_multiclass_object_selection_benchmark import scale_state
    details, summary = [], []
    for pool, path in {'A':A, **POOLS}.items():
        errors, changes = [], []
        rows = read_csv(path / 'manifests/candidate_pool.csv')
        for row in rows:
            x1, y1, x2, y2 = json.loads(row['bbox_norm_xyxy'])
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
            area = (x2 - x1) * (y2 - y1)
            saved = float(row['bbox_area_ratio'])
            error = abs(area - saved)
            changed = scale_state(area) != scale_state(saved)
            errors.append(error); changes.append(changed)
            details.append(dict(pool=pool, sample_id=row['sample_id'], recomputed_area=area,
                manifest_area=saved, abs_difference=error, scale_recomputed=scale_state(area),
                scale_manifest=scale_state(saved), scale_changed=changed))
        summary.append(dict(pool=pool, targets=len(rows), max_abs_difference=max(errors),
            scale_changes=sum(changes), area_difference_over_1e6=sum(e > 1e-6 for e in errors)))
    assert all(r['scale_changes'] == 0 and r['area_difference_over_1e6'] == 0 for r in summary)
    return details, summary


def main():
    for directory in ('tables', 'bootstrap', 'figures'):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    inputs = [PROTOCOL, Path(__file__), OLD_REFERENCE]
    for path in [A, *POOLS.values()]:
        inputs += [path / 'manifests/candidate_pool.csv'] + sorted((path / 'configs').glob('*.json'))
        inputs += sorted((path / 'annotations').glob('*/*/parsed/*.json'))
        if path != A:
            inputs += sorted((path / 'benchmark').glob('**/candidate_outputs.npz'))
            inputs += [path / 'confirmation/tables/full_stratum_metrics.csv']
    freeze = {str(p):sha256(p) for p in inputs}
    frozen = OUT / 'input_freeze.sha256.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == freeze, 'frozen input changed'
    else:
        write_json(frozen, freeze)
    write_json(OUT / 'registration.json', dict(created_at=now(), posthoc=True, draws=DRAWS,
        configs=CONFIGS, ranking_metric='Q5 mean-seed Macro-F1', no_new_training=True,
        no_new_inference=True, manual_review=False, bootstrap_seed=202610052))
    rng = np.random.default_rng(202610052)
    gathered = [[], [], [], [], [], []]
    for pool, path in POOLS.items():
        note('ranking', pool=pool)
        result = ranking_analysis(pool, path, rng)
        for destination, records in zip(gathered, result):
            destination.extend(records if isinstance(records, list) else [records])
    for name, rows in zip(('risk_assignments', 'model_ranks', 'configuration_pairs', 'model_pair_differences', 'seed_ranks', 'ranking_summary'), gathered):
        write_csv(OUT / 'tables' / (name + '.csv'), rows)
    note('official_attribute_reference')
    for name, rows in zip(('attribute_samples', 'attribute_metrics', 'attribute_source_differences'), attribute_analysis(rng)):
        write_csv(OUT / 'tables' / (name + '.csv'), rows)
    for name, rows in zip(('scale_details', 'scale_summary'), scale_check()):
        write_csv(OUT / 'tables' / (name + '.csv'), rows)
    assert all(sha256(p) == digest for p, digest in freeze.items())
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    audit = dict(status='pass', completed_at=now(), input_files_unchanged=len(freeze),
        configurations=18, model_rank_rows=len(gathered[1]), configuration_pair_rows=len(gathered[2]),
        prior_q5_metrics_verified=432, prior_reference_matrices_verified=24,
        fixed_panel_source_bootstrap=DRAWS, posthoc=True, no_new_inference=True,
        no_new_training=True, manual_review=False, accuracy_is_binary_reference_only=True)
    write_json(OUT / 'analysis_audit.json', audit)
    write_json(OUT / 'progress.json', dict(**audit, stage='analysis_complete'))
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'progress.json', dict(status='failed', heartbeat=now(), traceback=traceback.format_exc()))
        raise
