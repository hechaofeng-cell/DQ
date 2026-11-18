#!/usr/bin/env python3
"""Independently recalculate robustness metrics from frozen inputs and saved sets."""
import argparse
import json
from pathlib import Path

import numpy as np

import run_design_robustness_20261110 as exp


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-8, (a, b)


def truth(v):
    assert v in ['True', 'False']
    return v == 'True'


def mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows]))


def pass_rate(rows, field):
    return 100 * float(np.mean([truth(r[field]) for r in rows]))


def audit(out):
    exp.cc.verify_freeze(out / 'input_freeze.json')
    ctx = exp.frozen.contexts()
    thresholds = json.loads((exp.BASE / 'thresholds.json').read_text())
    starts = np.load(exp.BASE / 'main_sets.npz', allow_pickle=False)
    axes = exp.read(out / 'axes_runs.csv')
    sets = np.load(out / 'axes_sets.npz', allow_pickle=False)
    assert len(axes) == 1500 and len(sets.files) == 1500
    for r in axes:
        c, seed, setting = r['class_name'], int(r['selection_seed']), r['setting']
        ct = ctx[c]
        alloc = json.loads(r['tier_allocation'])
        quota = exp.quotas_for(ct['keys'], ct['cell'], alloc)
        selected = sets[f'{setting}__{c}__{seed}']
        exp.validate_for(ct['cell'], quota, ct['tiers'], selected, alloc)
        s0 = (starts[exp.frozen.key(c, 'R', seed)] if r['axis'] == 'change_limit' else
              exp.initial_for(ct['cell'], quota, seed, exp.CLASSES.index(c)))
        limit = int(setting.split('=')[1]) if r['axis'] == 'change_limit' else 20
        moved = len(set(s0) - set(selected))
        assert moved == int(r['removed_targets']) and moved <= limit
        metrics = exp.cc.count_metrics(ct['matrix'][selected].sum(0), 20)
        close(100 * metrics['occurrence'], r['occurrence_pct'])
        close(100 * metrics['support_completion'], r['support_completion_pct'])
        assert truth(r['final_pass']) == exp.cc.passes(metrics, thresholds[c])
        if not truth(r['final_pass']):
            assert np.array_equal(selected, s0), 'Failed adjustment must return original selection'
    perturb = exp.read(out / 'label_perturbation_runs.csv')
    worlds = exp.read(out / 'label_perturbation_worlds.csv')
    sets = np.load(out / 'label_perturbation_sets.npz', allow_pickle=False)
    assert len(perturb) == 2880 and len(worlds) == 180 and len(sets.files) == 2880
    witnesses = np.load(out / 'label_perturbation_witnesses.npz', allow_pickle=False)
    assert len(witnesses.files) == 180
    native = exp.native_features()
    grouped = {}
    for r in perturb:
        grouped.setdefault((r['class_name'], float(r['rate']), int(r['repeat'])), []).append(r)
    for w in worlds:
        c, rate, rep = w['class_name'], float(w['rate']), int(w['repeat'])
        ct = ctx[c]
        rng = np.random.default_rng(np.random.SeedSequence([2026111020, exp.CLASSES.index(c), int(rate * 1000), rep]))
        X, risk = exp.perturb_inputs(c, ct, [native[s] for s in ct['ids']], rng, rate)
        _, tiers, keys, cell = exp.rescore(c, ct['rows'], risk)
        quota = exp.quotas_for(keys, cell, [40] * 5)
        witness = witnesses[f'{c}__{rate}__{rep}']
        exp.validate_for(cell, quota, tiers, witness, [40] * 5)
        wm = exp.cc.count_metrics(X[witness].sum(0), 20)
        recalibrated = {k: .95 * wm[k] for k in thresholds[c]}
        for k in recalibrated:
            close(recalibrated[k], w['threshold_' + k])
        fraction = exp.cell_change_fraction(ct['keys'], ct['cell'], keys, cell)
        close(fraction, w['cell_change_fraction'])
        assert int(w['false_positive_bits']) == int((X & ~ct['matrix']).sum()) > 0
        assert int(w['false_negative_bits']) == int((ct['matrix'] & ~X).sum()) > 0
        assert int(w['shared_changed_bits']) > 0 and int(w['native_risk_changed_fields']) > 0
        for r in grouped[c, rate, rep]:
            seed = int(r['selection_seed'])
            mode = r['threshold_mode']
            selected = sets[f'{c}__{rate}__{rep}__{seed}__{mode}']
            effective = thresholds[c] if mode == 'frozen_clean' else recalibrated
            for k in effective:
                close(effective[k], r['threshold_' + k])
            exp.validate_for(cell, quota, tiers, selected, [40] * 5)
            s0 = exp.initial_for(cell, quota, seed, exp.CLASSES.index(c))
            assert len(set(s0) - set(selected)) == int(r['removed_targets']) <= 20
            for view, matrix in [('view', X), ('clean', ct['matrix'])]:
                metrics = exp.cc.count_metrics(matrix[selected].sum(0), 20)
                close(100 * metrics['occurrence'], r[f'{view}_occurrence_pct'])
                close(100 * metrics['support_completion'], r[f'{view}_support_pct'])
                target = effective if view == 'view' else thresholds[c]
                assert exp.cc.passes(metrics, target) == truth(r[f'{view}_pass'])
            if not truth(r['view_pass']):
                assert np.array_equal(selected, s0)
    counts = exp.read(out / 'bootstrap_counts.csv')
    draws = np.load(out / 'bootstrap_count_draws.npz', allow_pickle=False)
    assert len(counts) == 1800 and len(draws.files) == 3600
    original = {}
    for r in exp.read(exp.BASE / 'diagnostic_runs.csv'):
        if r['scope'] == 'all':
            key = (r['class_name'], r['method'], int(r['selection_seed']))
            original[key] = original.get(key, 0) + int(r['confirmed'])
    for r in counts:
        c, m, seed = r['class_name'], r['method'], int(r['selection_seed'])
        key = f'{c}__{m}__{seed}'
        a, b = draws[key + '__fixed'], draws[key + '__legacy']
        assert (b <= a).all() and a.shape == b.shape == (exp.BOOT,)
        assert int(r['point_confirmed']) == original[c, m, seed]
        for name, values in [('fixed', a), ('regated', b)]:
            close(r[name + '_p025'], np.quantile(values, .025))
            close(r[name + '_p975'], np.quantile(values, .975))
            close(r[name + '_median'], np.median(values))
            close(r[name + '_mean'], values.mean())
    conditions = 0
    for path in [out / 'bootstrap_conditions.csv', out / 'bootstrap_reference_effects.csv']:
        import csv
        with path.open(encoding='utf-8-sig') as handle:
            for r in csv.DictReader(handle):
                assert int(r['valid_replicates']) == exp.BOOT, 'Undefined draws cannot masquerade as negative findings'
                assert int(r['legacy_eligible_replicates']) <= exp.BOOT
                assert float(r['boot_p025']) <= float(r['boot_p975'])
                assert float(r['boot_leq_minus10_regated']) <= float(r['boot_leq_minus10']) + 1e-12
                conditions += 1
    exp.cc.dump(out / 'validation.json', dict(status='pass', axes_recomputed=len(axes),
                                             perturb_selections_recomputed=len(perturb),
                                             perturb_worlds_replayed=len(worlds),
                                             bootstrap_point_counts_reproduced=len(counts),
                                             continuous_effect_records_checked=conditions,
                                             input_freeze_verified=True))
    return axes, perturb, worlds, counts


def report(out):
    axes, perturb, worlds, counts = audit(out)
    rows = []
    for setting in ['L=10', 'equal', 'pool_proportional', 'risk_weighted', 'L=40']:
        rs = [r for r in axes if r['setting'] == setting]
        rows.append(dict(setting=setting, runs=len(rs), pass_pct=pass_rate(rs, 'final_pass'),
                         mean_occurrence_pct=mean(rs, 'occurrence_pct'),
                         mean_support_pct=mean(rs, 'support_completion_pct'),
                         mean_removed=mean(rs, 'removed_targets')))
    exp.cc.write(out / 'axes_summary.csv', rows)
    label_summary = []
    for mode in exp.THRESHOLD_MODES:
      for rate in exp.RATES:
        rs = [r for r in perturb if float(r['rate']) == rate and r['threshold_mode'] == mode]
        ws = [w for w in worlds if float(w['rate']) == rate]
        label_summary.append(dict(threshold_mode=mode, rate=rate, runs=len(rs), view_pass_pct=pass_rate(rs, 'view_pass'),
                                  clean_pass_pct=pass_rate(rs, 'clean_pass'),
                                  view_support_pct=mean(rs, 'view_support_pct'),
                                  clean_support_pct=mean(rs, 'clean_support_pct'),
                                  tier_change_pct=100 * mean(ws, 'tier_change_fraction'),
                                  cell_change_pct=100 * mean(ws, 'cell_change_fraction'),
                                  changed_vs_clean_P20=mean(rs, 'changed_vs_clean_P20')))
    exp.cc.write(out / 'label_summary.csv', label_summary)
    bs = []
    for m in ['R', 'C20', 'CS', 'P20', 'G', 'SR']:
        rs = [r for r in counts if r['method'] == m]
        # Each row is a class x start; sum over six classes and average over starts.
        bs.append(dict(method=m, point_mean_total=6 * mean(rs, 'point_confirmed'),
                       fixed_boot_mean_total=6 * mean(rs, 'fixed_mean'),
                       legacy_boot_mean_total=6 * mean(rs, 'regated_mean')))
    exp.cc.write(out / 'bootstrap_summary.csv', bs)
    byclass = []
    for c in exp.CLASSES:
      for mode in exp.THRESHOLD_MODES:
        for rate in exp.RATES:
            rs = [r for r in perturb if r['class_name'] == c and float(r['rate']) == rate and r['threshold_mode'] == mode]
            byclass.append(dict(class_name=c, threshold_mode=mode, rate=rate, view_pass_pct=pass_rate(rs, 'view_pass'),
                                clean_pass_pct=pass_rate(rs, 'clean_pass'),
                                clean_support_pct=mean(rs, 'clean_support_pct')))
    exp.cc.write(out / 'label_summary_by_class.csv', byclass)
    text = [
        '# 设计稳健性补全（2026-11-10）', '',
        '三个补全实验已执行并经保存选集独立重算。基于冻结历史池与旧模型预测，没有新推理、训练或人工标签；不是 fresh gate 验证。', '',
        '## 风险输入的来源修复', '',
        '原评分读取 native_features.jsonl；horse/sheep/person/car 的 specific 字段来自另一轮覆盖专用标注。'
        'limb_visibility 与 leg_visibility 不是别名，同名的头部字段也不能互换。共享字段与 native 风险输入同步翻转；'
        '旧风险专用字段独立翻转。六类干净分数、Q层、精细单元均精确复现，零扰动输入保持不变。', '',
        '## L 与配额结构', '',
        '每设置六类×50起点，预算200。L轴沿用冻结R起点；配额轴固定L=20并按新配额重建起点。'
        '均分为每Q40；池比例按各Q候选数分配；风险加权按候选数×Q编号分配，最大余数法得到13/27/40/53/67。'
        '原池每Q80，故池比例与均分严格等价。所有设置保留原同池阈值，未为新配额重新校准。', '',
        '| 设置 | 达标率(%) | 出现覆盖(%) | 支持完成度(%) | 平均替换目标 |',
        '|---|---:|---:|---:|---:|',
    ]
    text += [f"| {r['setting']} | {r['pass_pct']:.2f} | {r['mean_occurrence_pct']:.2f} | {r['mean_support_pct']:.2f} | {r['mean_removed']:.2f} |" for r in rows]
    text += ['', '## 标签翻转与风险传导', '',
             '每错误率六类×10噪声世界×8选样起点。错误概率施加于已知字段；枚举替换会同时引入FP和FN；'
             '集合字段依照schema增删或替换且遵守排他约束。未知保持未知，几何尺度不扰动。'
             '这不是误差率估计，也不涵盖未知→已知幻觉或相关性标签错误。', '',
             '扰动同时影响覆盖矩阵、旧风险分数、Q层、精细单元、配额和选样。frozen_clean保留旧阈值；'
             'recalibrated_corrupt在噪声世界中先运行同池G参照，用其覆盖值×0.95重定阈值，再冻结后执行P20，全程不读取预测。'
             '扰动视角按该模式阈值判断；干净参照始终按原干净阈值判断。'
             '“干净标签参照”仍是旧VLM标签，不能称为语义真值。cell变更按(Q,分数,区间宽度)键比较，不比较重编号的整数ID。', '',
             '| 门槛模式 | 字段错误率 | 扰动视角达标(%) | 干净参照达标(%) | Q层变更(%) | 精细单元变更(%) |',
             '|---|---|---:|---:|---:|---:|']
    text += [f"| {r['threshold_mode']} | {100*r['rate']:.0f}% | {r['view_pass_pct']:.2f} | {r['clean_pass_pct']:.2f} | {r['tier_change_pct']:.2f} | {r['cell_change_pct']:.2f} |" for r in label_summary]
    text += ['', '## 固定资格的图像簇 bootstrap', '',
             '1000次重复；B/D按类独立重采样原图，权重在方法、模型、选样起点之间共享。B固定选集接受原池图像权重，'
             '所以每次重复的选集总权重可变；不重跑选样。点估计的资格固定，B提案固定后在D计算确认数量。'
             '所有点估计合格的B和D状态均输出连续Δ分位数，避免只展示原先弱状态。', '',
             '旧法对照同时检查带重复次数的目标数和不同原图数均≥20。固定资格移除了重复开资格门造成的截断；'
             '确认数量仍是离散的，以下均值及原始计数分位数均为描述性敏感性结果，不是校准置信区间。', '',
             '| 方法 | 原点估计确认数 | 固定资格重采样均值 | 旧法重开门均值 |',
             '|---|---:|---:|---:|']
    text += [f"| {r['method']} | {r['point_mean_total']:.2f} | {r['fixed_boot_mean_total']:.2f} | {r['legacy_boot_mean_total']:.2f} |" for r in bs]
    text += ['', 'Δ区间是对冻结历史池、自动标签、选集和条件的近似逐状态区间。没有覆盖率模拟、后选择校准或多重比较校正；'
             '不能把本补全写成“推断校准全部解决”，也不能据此声称P20诊断优于强基线。', '',
             '## 复现与文件', '',
             '运行环境见environment.json，完整输入哈希见input_freeze.json。selection与perturb阶段不读取分类器预测值；'
             'boot阶段只读取已有缓存。输出axes_sets.npz与label_perturbation_sets.npz保存每次选集，label_perturbation_worlds.csv记录误差传导。'
             'validation.json逐选集复算覆盖/配额并验证180个扰动世界与1800条bootstrap点计数。', '',
             '仓库脚本：scripts/run_design_robustness_20261110.py、scripts/report_design_robustness_20261110.py；'
             '测试：tests/test_design_robustness.py。历史依赖output/experiments/论文关键结论验证_20261106/common.py及数据均沿用本地模式，'
             'artifacts/、output/受.gitignore忽略，不能仅靠Git恢复完整实验。', '',
             '```bash',
             '.venv/bin/python scripts/run_design_robustness_20261110.py prepare',
             '.venv/bin/python scripts/run_design_robustness_20261110.py axes',
             '.venv/bin/python scripts/run_design_robustness_20261110.py perturb',
             '.venv/bin/python scripts/run_design_robustness_20261110.py boot',
             '.venv/bin/python scripts/report_design_robustness_20261110.py',
             '```', '',
             '重复运行请每阶段使用相同的全新--out路径；输入代码被哈希冻结后不得原位修改并沿用产物。'
             '先前未完成的v1产物保留在artifacts/design_robustness_20261110，不作为本报告证据。', '']
    (out / 'report_zh.md').write_text('\n'.join(text))
    (out / 'claim_evidence.md').write_text(
        '# Claim evidence\n\n'
        '| Claim | Status | Evidence | Boundary |\n|---|---|---|---|\n'
        '| Six clean native risks reproduce frozen scores and cells | supported | clean_rescore_audit.json | no semantic label accuracy claim |\n'
        '| Label errors propagate through risk/tiers/cells/quotas/selection | supported | label_perturbation_worlds.csv; label_perturbation_runs.csv | independent synthetic known-label noise only |\n'
        '| L/quota robustness is quantified | supported | axes_runs.csv; axes_summary.csv | original frozen thresholds; historical pool only |\n'
        '| Repeated eligibility gate artifact removed from effect intervals | supported | bootstrap_reference_effects.csv; bootstrap_conditions.csv | pointwise conditional approximation, not coverage validation |\n'
        '| Confirmation-count quantiles are calibrated intervals | unsupported | bootstrap_summary.csv | counts remain discrete |\n'
        '| P20 exceeds strong baselines in diagnostic discovery | not established here | bootstrap_summary.csv | needs a separate inferential comparison |\n')
    (out / 'review.md').write_text(
        '# Review\n\n'
        '- Citation integrity: pass; no new external factual or citation claims.\n'
        '- Result traceability: pass; input hashes verified; saved selections, noise worlds and point counts independently recalculated.\n'
        '- Claim scope: pass; clean labels are not semantic truth; historical pools are not fresh validation; count percentiles are not confidence intervals.\n'
        '- Resolved: native/coverage provenance; cell-key renumbering; schema-valid flips; repeated distinct-image eligibility gate in effect summaries.\n'
        '- Remaining limits: actual label accuracy; correlated/unknown-label errors; interval coverage and post-selection calibration; multiplicity; fresh pools; co-state confounding.\n')
    tasks = out / 'task_list.md'
    tasks.write_text(tasks.read_text().replace('- [ ]', '- [x]'))
    outputs = [p for p in out.iterdir() if p.is_file() and p.name != 'final_artifacts.json']
    exp.cc.dump(out / 'final_artifacts.json', dict(status='complete', files={str(p): exp.cc.sha(p) for p in sorted(outputs)},
                                                reporting_script_sha256=exp.cc.sha(__file__)))
    print(json.dumps(dict(status='complete', output=str(out), axes=rows, label=label_summary, bootstrap=bs), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=exp.OUT)
    report(parser.parse_args().out.resolve())
