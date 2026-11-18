"""Factorial contrasts, independent scalar diagnostic audit, and bounded report."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import run_label_factorial_20261112 as run

OUT, cc, study = run.OUT, run.cc, run.study


def contrasts():
    rr = study.read(OUT / 'summary.csv')
    index = {tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'evaluator', 'reference', 'method']): r for r in rr}
    def difference(risk, cover, mode, ev, ref, metric, comparator='Greedy20'):
        return float(index[risk, cover, mode, ev, ref, 'P20'][metric]) - float(index[risk, cover, mode, ev, ref, comparator][metric])
    metrics = ['confirmed_per_start', 'support_completion_pct', 'error_pct']
    effects, measurement, gains = [], [], []
    for mode in run.MODES:
        for ev in study.SOURCES:
            for ref in study.SOURCES:
                for risk in study.SOURCES:
                    for metric in metrics:
                        da, db = [difference(risk, cov, mode, ev, ref, metric) for cov in study.SOURCES]
                        effects.append(dict(factor='coverage', fixed_source=risk, threshold_mode=mode,
                            evaluator=ev, reference=ref, metric=metric, A_P20_minus_Greedy20=da,
                            B_P20_minus_Greedy20=db, change=db-da, strict_sign_reversal=da*db < -1e-12))
                for cover in study.SOURCES:
                    for metric in metrics:
                        da, db = [difference(risk, cover, mode, ev, ref, metric) for risk in study.SOURCES]
                        effects.append(dict(factor='risk_bundle', fixed_source=cover, threshold_mode=mode,
                            evaluator=ev, reference=ref, metric=metric, A_P20_minus_Greedy20=da,
                            B_P20_minus_Greedy20=db, change=db-da, strict_sign_reversal=da*db < -1e-12))
        for risk in study.SOURCES:
            for cover in study.SOURCES:
                aa, ab, ba, bb = [difference(risk, cover, mode, ev, ref, 'confirmed_per_start')
                                  for ev, ref in [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]]
                measurement.append(dict(risk_source=risk, coverage_source=cover, threshold_mode=mode,
                    AA=aa, AB=ab, BA=ba, BB=bb, discovery_only_change=ba-aa,
                    reference_only_change=ab-aa, interaction=bb-ba-ab+aa, total_change=bb-aa,
                    strict_sign_reversal=aa*bb < -1e-12))
                for ev in study.SOURCES:
                    for ref in study.SOURCES:
                        p, g, r = [index[risk, cover, mode, ev, ref, m] for m in ['P20', 'Greedy20', 'R']]
                        gains.append(dict(risk_source=risk, coverage_source=cover, threshold_mode=mode,
                            evaluator=ev, reference=ref,
                            P20_minus_R_confirmed=float(p['confirmed_per_start'])-float(r['confirmed_per_start']),
                            P20_minus_Greedy20_confirmed=float(p['confirmed_per_start'])-float(g['confirmed_per_start']),
                            P20_minus_R_support_pp=float(p['support_completion_pct'])-float(r['support_completion_pct']),
                            P20_minus_R_error_pp=float(p['error_pct'])-float(r['error_pct']),
                            P20_minus_Greedy20_error_pp=float(p['error_pct'])-float(g['error_pct'])))
    run.write('factor_effects.csv', effects); run.write('measurement_effects.csv', measurement); run.write('gains.csv', gains)
    z = np.load(OUT / 'selections.npz', allow_pickle=False)
    overlap = []
    for risk in study.SOURCES:
        for mode in run.MODES:
            for method in run.METHODS:
                retained = []
                for c in run.CLASSES:
                    for seed in run.SEEDS:
                        a, b = [z[run.key(risk, cov, mode, c, method, seed)] for cov in study.SOURCES]
                        retained.append(len(set(a) & set(b)) / 200)
                assert min(retained) >= .8 - 1e-12
                if method == 'R':
                    assert min(retained) == 1
                overlap.append(dict(risk_source=risk, threshold_mode=mode, method=method,
                                    mean_retained_fraction=float(np.mean(retained)), minimum_retained_fraction=min(retained)))
    run.write('coverage_switch_overlap.csv', overlap)
    # Exact additive decomposition of ranking change of the same selected sets.
    byfield = defaultdict(lambda: defaultdict(float))
    for r in study.read(OUT / 'by_field.csv'):
        if r['method'] not in ['P20', 'Greedy20']:
            continue
        k = tuple(r[f] for f in ['risk_source', 'coverage_source', 'threshold_mode', 'class_name', 'namespace', 'field'])
        byfield[k][r['evaluator'], r['reference']] += (1 if r['method'] == 'P20' else -1) * float(r['confirmed_per_start'])
    fields = []
    for k, v in byfield.items():
        risk, cover, mode, c, ns, field = k
        aa, ab, ba, bb = [v[e, r] for e, r in [('A','A'),('A','B'),('B','A'),('B','B')]]
        fields.append(dict(risk_source=risk, coverage_source=cover, threshold_mode=mode,
                           class_name=c, namespace=ns, field=field,
                           label_kind='fine' if c not in ['dog', 'cat'] and ns == 'specific' else 'native',
                           AA_P20_minus_Greedy20=aa, BB_P20_minus_Greedy20=bb,
                           discovery_only_change=ba-aa, reference_only_change=ab-aa,
                           interaction=bb-ba-ab+aa, total_change=bb-aa))
    run.write('field_measurement_decomposition.csv', fields)
    for r in measurement:
        relevant = [f for f in fields if all(f[k] == r[k] for k in ['risk_source','coverage_source','threshold_mode'])]
        assert abs(sum(f['total_change'] for f in relevant) - r['total_change']) < 1e-9
    return index, effects, measurement, gains, fields


def scalar_audit():
    rows, risks, matrices = run.contexts()
    selections = np.load(OUT / 'selections.npz', allow_pickle=False)
    seed = run.SEEDS[0]
    records = [r for r in study.read(OUT / 'diagnostic.csv') if int(r['selection_seed']) == seed]
    correctness = {}
    for phase in rows:
        pred, hashes = cc.old.predictions(phase, rows[phase])
        offsets = {r['sample_id']: i for i, r in enumerate(rows[phase])}
        for ci, c in enumerate(run.CLASSES):
            pos = [offsets[s] for s in risks[phase, 'A', c]['ids']]
            for model in run.old.MODELS:
                correctness[phase, c, model] = (pred[model][:, pos] == ci).mean(0)
    def masks(phase, source, c):
        x = matrices[phase, source, c]
        states = risks[phase, source, c]['states']
        result = []
        for j, state in enumerate(states):
            group = [k for k, other in enumerate(states) if tuple(other[:2]) == tuple(state[:2])]
            known = np.logical_or.reduce(x[:, group], axis=1)
            result.append((x[:, j], known & ~x[:, j]))
        return result
    mask_cache = {(phase, source, c): masks(phase, source, c) for phase in rows for source in study.SOURCES for c in run.CLASSES}
    def scalar(left, right, correct):
        n1, n0 = int(left.sum()), int(right.sum())
        eligible = n1 >= 20 and n0 >= 20
        gap = 100 * (float(correct[left].sum()) / n1 - float(correct[right].sum()) / n0) if n1 and n0 else 0
        return eligible, eligible and gap <= -10 + 1e-12
    ref = {}
    for source in study.SOURCES:
        for c in run.CLASSES:
            for model in run.old.MODELS:
                ref[source, c, model] = [scalar(a, b, correctness['D', c, model]) for a,b in mask_cache['D', source, c]]
    for r in records:
        risk, cover, mode, c, method, evaluator, reference, model = [r[k] for k in
            ['risk_source','coverage_source','threshold_mode','class_name','method','evaluator','reference','model']]
        ix = selections[run.key(risk, cover, mode, c, method, seed)]
        correct = correctness['B', c, model][ix]
        counters = np.zeros(4, int)
        for j, (a,b) in enumerate(mask_cache['B', evaluator, c]):
            eligible, proposed = scalar(a[ix], b[ix], correct)
            re, rw = ref[reference, c, model][j]
            counters += [int(eligible), int(proposed), int(proposed and re), int(proposed and rw)]
        assert list(counters) == [int(r[k]) for k in ['eligible','proposed','reference_evaluable','confirmed']], r
        assert abs(100 * (1-float(correct.mean())) - float(r['error_pct'])) < 1e-10
    run.dump('independent_diagnostic_audit.json', dict(status='pass', independently_recomputed_rows=len(records),
        scope='First predeclared selection seed, all 8 scenarios x 6 classes x 4 methods x 2 discovery labels x 2 reference labels x 5 classifiers',
        method='Scalar per-state same-field known masks and explicit means; no reference_stats function used'))


def report(index, effects, measurement, gains, fields):
    def get(risk, cover, mode, ev, ref, method):
        return index[risk, cover, mode, ev, ref, method]
    reversals = [r for r in effects if r['factor']=='coverage' and r['threshold_mode']=='original' and
                 r['metric']=='confirmed_per_start' and r['strict_sign_reversal']]
    risk_reversals = [r for r in effects if r['factor']=='risk_bundle' and r['threshold_mode']=='original' and
                      r['metric']=='confirmed_per_start' and r['strict_sign_reversal']]
    measured = [r for r in measurement if r['strict_sign_reversal']]
    positive_r = sum(g['P20_minus_R_confirmed'] > 0 for g in gains)
    positive_g = sum(g['P20_minus_Greedy20_confirmed'] > 0 for g in gains)
    lines = ['# 固定风险与覆盖标签来源的2×2实验\n',
        '本轮回答：跨VLM结果变化来自选样依据、风险分层，还是诊断的评价标签？A为Qwen，B为SenseNova；两者都不是真值。所有工作自动完成，没有人工核验或新增VLM调用。\n',
        '## 设计与核验\n',
        '6类B候选共2400目标，历史D参考1200目标；每类选200、替换最多20、支持上限20，50个固定起点。2风险来源×2覆盖来源×2门槛，共9600组选集。其中4800组复用历史选集（AA两门槛相同，重复列示），4800组为缺失的混合组合；这些并非9600个独立数据集。',
        '同风险的覆盖比较保持完全相同的风险细胞、整数配额及初始选集。风险来源切换同时改变细胞、配额及起点，只解释为整套风险构造的影响。原数值门槛与组合内G×0.95的预测盲重标定门槛分开报告；后者比较同时包含目标门槛的变化。',
        '选集冻结后读取5分类器×3预测种子的缓存。两套标签分别用于B发现和D参考，四个评价组合全部保留；原AA/BB的64条摘要逐项复现。所有选集通过配额/替换预算核验，19200行覆盖结果重算；首个预定起点的3840行诊断另用逐状态标量算法独立核算。\n',
        '## 主结果\n',
        f'固定原数值门槛后，只切换覆盖标签时，8种固定风险/评价组合中有{len(reversals)}种出现P20−Greedy20严格正负翻转；切换风险构造时有{len(risk_reversals)}种。对于完全固定的选集，仅切换发现与参考标签，在8个场景中有{len(measured)}个翻转。',
        f'所有32种组合/评价口径中，P20相对R的确认条件均值为正的有{positive_r}种，相对Greedy20为正的有{positive_g}种。这里的组合高度相关，不能当作独立重复或显著性检验。\n',
        '| 风险 | 覆盖 | 门槛 | A发现/A参考：P−G | B发现/B参考：P−G | A/A：P−R | B/B：P−R |',
        '|---|---|---|---:|---:|---:|---:|']
    for mode in run.MODES:
        for risk in study.SOURCES:
            for cover in study.SOURCES:
                vals = []
                for comparator in ['Greedy20', 'R']:
                    for ev in study.SOURCES:
                        p, g = [get(risk, cover, mode, ev, ev, method) for method in ['P20', comparator]]
                        vals.append(float(p['confirmed_per_start'])-float(g['confirmed_per_start']))
                lines.append(f'| {risk} | {cover} | {"原数值" if mode=="original" else "组合内重标定"} | ' + ' | '.join(f'{v:+.2f}' for v in vals) + ' |')
    lines += ['\n表中是每起点确认条件数的差：先跨6类×5分类器求和，再对50起点平均；它不是独立根因数，也不是正确标签数。',
              '\n## 固定选集：发现标签与参考标签各自的影响\n',
              '| 风险/覆盖/门槛 | A/A | A发现/B参考 | B发现/A参考 | B/B |', '|---|---:|---:|---:|---:|']
    for r in measurement:
        lines.append(f'| {r["risk_source"]}/{r["coverage_source"]}/{r["threshold_mode"]} | ' +
                     ' | '.join(f'{r[f]:+.2f}' for f in ['AA','AB','BA','BB']) + ' |')
    lines += ['\n每行保持同一组选集，表内仅更换诊断标签。发现单独变化、参考单独变化及交互的精确分解见measurement_effects.csv，按类/字段的可加分解见field_measurement_decomposition.csv。\n',
              '## 原AA选集的字段分解（原数值门槛）\n',
              '| 类 | 命名空间/字段 | A/A下P−G | B/B下P−G | 标签切换带来的变化 |', '|---|---|---:|---:|---:|']
    ff = sorted([f for f in fields if (f['risk_source'],f['coverage_source'],f['threshold_mode'])==('A','A','original')],
                key=lambda f: -abs(f['total_change']))
    for f in ff[:12]:
        lines.append(f'| {f["class_name"]} | {f["namespace"]}/{f["field"]} | {f["AA_P20_minus_Greedy20"]:+.2f} | {f["BB_P20_minus_Greedy20"]:+.2f} | {f["total_change"]:+.2f} |')
    lines += ['\n按绝对变化列前12项，其余均保留在CSV中。这是历史条件计数的算术归因，不能确定哪个VLM判断正确，不能当作这些字段的真实语义因果效应。\n',
              '## 门槛、覆盖与实际错误率\n',
              '| 风险/覆盖/门槛 | B标签下P20支持完成率 | P20原门槛通过/300 | P20重标定门槛通过/300 | P20错误率 | Greedy20错误率 |',
              '|---|---:|---:|---:|---:|---:|']
    for mode in run.MODES:
        for risk in study.SOURCES:
            for cover in study.SOURCES:
                p,g = [get(risk, cover, mode,'B','B',m) for m in ['P20','Greedy20']]
                lines.append(f'| {risk}/{cover}/{mode} | {float(p["support_completion_pct"]):.3f}% | {p["pass_original_count"]} | {p["pass_recalibrated_count"]} | {float(p["error_pct"]):.3f}% | {float(g["error_pct"]):.3f}% |')
    lines += ['\n固定风险配额不等于固定实际错误率。重标定使门槛符合该组合的G基线，但不等于跨来源统一语义或统一绝对目标。\n',
              '## 决策与边界\n',
              '本轮应按factor_effects.csv中的固定因素比较判断原假设，而不凭全来源AA→BB的混合变化归因。当前证据不支持“P20在VLM切换下稳定优于Greedy20”的强表述。相对R的收益须保留具体标签、门槛、风险构造和实际错误率条件，不能称为普遍优越。',
              '不把提示词修订小试验的负面结果改写为成功，不据本轮历史敏感性分析全量重标或挑选有利VLM。后续若要恢复强结论，需要事先固定语义协议与决策指标，并在真正未使用数据上验证。人工不参与本轮核验。',
              '\n复现：.venv/bin/python scripts/run_label_factorial_20261112.py prepare / select / evaluate（依次执行），随后 .venv/bin/python scripts/analyze_label_factorial_20261112.py。完成目录不可覆盖；更改设计须另起版本。\n']
    text = '\n'.join(lines)
    (OUT / 'report_zh.md').write_text(text)
    doc = study.ROOT / 'docs/实验方案与结果/风险与覆盖标签来源拆分实验_20261112.md'
    doc.write_text(text)
    run.dump('decision.json', dict(coverage_only_original_threshold_rank_reversals=len(reversals),
        coverage_only_comparisons=8, risk_bundle_original_threshold_rank_reversals=len(risk_reversals),
        measurement_rank_reversals=len(measured), measurement_comparisons=8,
        P20_above_R_scenarios=positive_r, P20_above_Greedy20_scenarios=positive_g, total_scenarios=32,
        stable_P20_superiority_supported=False, semantic_accuracy_established=False,
        stop='Bounded factorial and field decomposition complete; do not tune historical prompts to rescue rank'))
    (OUT / 'claim_evidence.md').write_text(
        '# 主张—证据\n\n固定因素下的覆盖/风险敏感性：factor_effects.csv，分门槛/评价口径表述。\n\n'
        '固定选集的评价口径敏感性：measurement_effects.csv。字段贡献：field_measurement_decomposition.csv，只作算术分解。\n\n'
        'P20相对R和Greedy20：gains.csv与paired_differences.csv，不将50个固定起点当作50个独立数据集。\n\n'
        '不支持：稳定优于Greedy20、VLM语义准确率、独立根因增多、真正新数据泛化、配额控制等于实际错误率控制。\n')
    (OUT / 'review.md').write_text(
        '# 自动审查\n\n引用完整性：本轮未新增外部文献事实，不适用。\n\n'
        '结果追溯：通过。全部覆盖/门槛/配额重算，历史对角组合复现，3840条首起点诊断逐状态独立重算，字段分解严格守恒。\n\n'
        '主张边界：通过。仅历史自动标签敏感性，明确评价口径与风险构造差异、重标定的混杂和非独立重复，不宣称语义正确。\n')
    run.event('followup_complete', coverage_rank_reversals=len(reversals), risk_rank_reversals=len(risk_reversals))
    outputs = {str(p): study.sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['final_outputs_freeze.json','completion.json']}
    outputs[str(doc)] = study.sha(doc)
    cc.freeze(OUT / 'final_outputs_freeze.json', dict(files=outputs))
    run.dump('completion.json', dict(status='complete', time=cc.old.now(), new_VLM_calls=0,
        output_freeze_sha256=study.sha(OUT / 'final_outputs_freeze.json'), report=str(doc)))


if __name__ == '__main__':
    assert not (OUT / 'completion.json').exists(), 'Completed experiment is immutable'
    cc.freeze(OUT / 'followup_code_freeze.json', dict(files={str(Path(__file__)): study.sha(Path(__file__))}))
    cc.verify_freeze(OUT / 'input_freeze.json'); cc.verify_freeze(OUT / 'selection_freeze.json')
    scalar_audit()
    report(*contrasts())
