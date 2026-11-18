#!/usr/bin/env python3
"""Build the Chinese verification report only from completed, audited outputs."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_cross_vlm_20261112 as study

ROOT = Path(__file__).resolve().parents[1]


def run():
    out = study.OUT
    validation = json.loads((out / 'validation.json').read_text())
    assert validation['status'] == 'pass'
    native = json.loads((out / 'native_summary.json').read_text())
    completeness = json.loads((out / 'specific_completeness.json').read_text())
    assert completeness['completed'] == completeness['expected'] == 2400
    summary = study.read(out / 'summary.csv')
    fields = study.read(out / 'native_field_agreement.csv')
    specific = study.read(out / 'specific_field_agreement.csv')
    specific_n = sum(int(r['n']) for r in specific)
    specific_agree = sum(float(r['exact_agreement_pct']) * int(r['n']) for r in specific) / specific_n
    native_known = sum(int(r['both_known_n']) for r in fields)
    specific_known = sum(int(r['both_known_n']) for r in specific)
    native_known_agree = sum(float(r['both_known_agreement_pct']) * int(r['both_known_n'])
                             for r in fields if int(r['both_known_n'])) / native_known
    specific_known_agree = sum(float(r['both_known_agreement_pct']) * int(r['both_known_n'])
                               for r in specific if int(r['both_known_n'])) / specific_known
    lookup = {(r['selection_source'], r['threshold_mode'], r['evaluator'],
               r['reference_label_source'], r['method']): r for r in summary}
    original_p = lookup['A', 'original', 'A', 'A', 'P20']
    cross_p = lookup['A', 'original', 'B', 'B', 'P20']
    original_g = lookup['A', 'original', 'A', 'A', 'Greedy20']
    cross_g = lookup['A', 'original', 'B', 'B', 'Greedy20']
    lines = [
        '# 跨VLM核验结果（2026-11-12）', '',
        '核验按用户要求全自动执行。A为Qwen3-VL 30B，B为SenseNova 6.8 Flash Lite。'
        '两套答案独立保存，任一模型都不作为语义真值；本轮检验一致性及下游结论对标注来源的敏感性。', '',
        '**核验结论：标签来源对结果有明显影响，现有证据不能支持“方法结论不依赖VLM来源”。**'
        f"原Qwen P20选集在Qwen标签下{original_p['pass_original_count']}/300次达到冻结门槛，"
        f"改用SenseNova标签核对同一门槛后仅{cross_p['pass_original_count']}/300次达标；"
        f"即使用SenseNova自身重标定的评价门槛，也只有{cross_p['pass_evaluator_count']}/300次。"
        '重新按SenseNova标签选样和重标定后，P20相对随机选样仍有收益，但不能据此宣布两套标签可互换或一致优于Greedy20。', '',
        '## 数据与执行范围', '',
        '- 当前COCO候选池2,400个目标（每类400），历史Objects365参考池1,200个目标（每类200）。',
        '- 校验并复用7,200份原生字段历史标注；另复用2,400份Qwen细粒度标注，新增2,400份SenseNova细粒度覆盖标注，逐目标配对马、羊、人、车四类。',
        '- 新标注使用原冻结字典、原语义提示及逐目标原证据图；不提供Qwen答案、分类器输出或选样结果。',
        '- 格式不合规时只重试原目标；最后10个车类枚举违规目标使用严格schema请求及原字典“其他主色”判据重述，实际模型答案保留，不在本地映射标签值。',
        '- 每类选200，最多移出20个原成员，保持各自标注来源所定义的精细风险单元配额，沿用50个历史选样起点。',
        '- 分类器沿用5模型×3种子冻结预测；没有重新训练或新增分类器推理，也没有人工核验。',
        '- 原生字段缓存使用历史模型别名和提示，包含旧版截断措辞；新增覆盖字段在本轮真实调用。不是所有字段都在同一天重新标注。', '',
        '## 1. 状态标签是否可互换', '',
        f"原生字段共{native['field_instances']:,}条配对记录，精确一致率为"
        f"**{native['micro_agreement_pct']:.2f}%**（包含unknown）。"
        f'新增细粒度字段共{specific_n:,}条配对记录，精确一致率为**{specific_agree:.2f}%**。'
        '这些比例不等于标签准确率。', '',
        f'双方均已知时，原生字段一致率为{native_known_agree:.2f}%（{native_known:,}条），'
        f'细粒度字段为{specific_known_agree:.2f}%（{specific_known:,}条）。'
        '这两个子集由模型输出决定，不能当作全体标签准确率，也不能忽略被排除的未知记录。', '',
        '| 原生字段 | 目标数 | 精确一致率 |', '|---|---:|---:|',
    ]
    cn = {'body_visibility': '整体可见性', 'head_visibility': '头部可见性',
          'leg_visibility': '腿部可见性', 'head_orientation': '头部朝向',
          'occlusion_level': '遮挡程度', 'truncation_level': '截断程度',
          'body_orientation': '身体朝向', 'visual_representation': '视觉呈现形式'}
    for f, name in cn.items():
        rr = [r for r in fields if r['field'] == f]
        n = sum(int(r['n']) for r in rr)
        pct = sum(float(r['exact_agreement_pct']) * int(r['n']) for r in rr) / n
        lines.append(f'| {name} | {n:,} | {pct:.2f}% |')
    lines += ['', '分字段表同时报告双方unknown比例、双方均已知时的一致率、Cohen κ，以及集合字段的Jaccard。'
        '不因两模型一致就标记“正确”，不把分歧直接裁成SenseNova答案。', '',
        '来源：`native_field_agreement.csv`、`specific_field_agreement.csv`、`native_value_transitions.csv`。', '',
        '## 2. 风险分层是否稳定', '',
        f"候选池更换标注来源后，**{native['B_Q_changed_pct']:.2f}%**的目标改变Q层，"
        f"**{native['B_cell_changed_pct']:.2f}%**改变精细风险单元。"
        '后者按单元含义（Q、分数、不确定性宽度）比较，不按独立编号比较。', '',
        '| 类别 | 风险分数平均绝对差（0–100） | 分数Spearman相关 | Q层变化 |',
        '|---|---:|---:|---:|',
    ]
    names = dict(dog='狗', cat='猫', horse='马', sheep='羊', person='人', car='车')
    for r in native['results']:
        if r['phase'] == 'B':
            lines.append(f"| {names[r['class_name']]} | {r['score_MAE']:.2f} | {r['score_spearman']:.3f} | {r['Q_changed_pct']:.2f}% |")
    lines += ['', 'Q层是类内相对排序，不是错误概率。单元变化不等于相同比例的配额或最终选集变化。'
        '来源：`risk_summary.csv`、`risk_per_target.csv`。', '',
        '固定分类器下的风险—错误关联另列，避免把Q层变化直接等同于风险代理失效：', '',
        '| 数据池 | 风险标签来源 | Q5−Q1平均错误率差 | 正向类别×模型／30 |',
        '|---|---|---:|---:|',
    ]
    gradient = study.read(out / 'risk_error_gradient.csv')
    for phase in ['B', 'D']:
        for source in ['A', 'B']:
            rr = [r for r in gradient if r['phase'] == phase and r['source'] == source]
            delta = float(np.mean([float(r['Q5_minus_Q1_pp']) for r in rr]))
            positive = sum(float(r['Q5_minus_Q1_pp']) > 0 for r in rr)
            lines.append(f"| {'COCO候选' if phase == 'B' else 'Objects365历史参考'} | "
                         f"{'Qwen' if source == 'A' else 'SenseNova'} | {delta:+.2f}个百分点 | {positive} |")
    lines += ['', '这是历史冻结分类器上的描述性关联，不能解释为错误概率校准或因果关系。'
        '来源：`risk_error_gradient.csv`。', '',
        '最终成员重合率按同类别、同方法、同起点计算：|A集合∩B集合|/200，再平均六类×50起点。', '',
        '| 方法 | SenseNova保留原门槛：与Qwen集合重合 | SenseNova重标定：与Qwen集合重合 |',
        '|---|---:|---:|',
    ]
    overlap = study.read(out / 'selection_overlap.csv')
    for method in study.METHODS:
        values = []
        for mode in ['original', 'recalibrated']:
            rr = [r for r in overlap if r['selection_source'] == 'B' and r['threshold_mode'] == mode and r['method'] == method]
            assert len(rr) == 300
            values.append(f"{100 * float(np.mean([float(r['retained_fraction_vs_A']) for r in rr])):.2f}%")
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines += ['', '20个成员改动上限约束各自风险组织下的初始集合到最终集合；'
        '它不约束两套风险标签重新组织配额并重新选样后的跨来源成员差异。来源：`selection_overlap.csv`。', '',
        '## 3. 方法收益是否能在另一个VLM下重复', '',
        '以下同时用该行评价来源的COCO候选池标签计算覆盖、Objects365历史参考池标签复核弱条件。'
        '覆盖为六类×50起点宏平均；弱条件数先汇总六类×五分类模型，再平均50起点。'
        '条件数不是独立失效根因数。', '',
        '| 选样来源／门槛 | 评价来源 | 方法 | 支持完成度 | 历史重复弱条件数 | 实际错误率 | 按评价源门槛达标／300 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for s, mode, label in [('A', 'original', 'Qwen／原门槛'), ('B', 'original', 'SenseNova／原门槛'),
                            ('B', 'recalibrated', 'SenseNova／重标定')]:
        for e in ['A', 'B']:
            for method in study.METHODS:
                r = lookup[s, mode, e, e, method]
                lines.append(f"| {label} | {'Qwen' if e == 'A' else 'SenseNova'} | {method} | "
                             f"{float(r['support_completion_pct']):.2f}% | {float(r['confirmed_per_start']):.2f} | "
                             f"{float(r['error_pct']):.2f}% | {r['pass_evaluator_count']} |")
    lines += ['', '“原门槛”保留原Qwen设置的数值要求；“重标定”用对应标注来源、相同预算与配额的G参照值×0.95，'
        '在读取本轮分类器预测前确定。重标定不能证明哪个VLM更正确；不同来源的精细配额随风险输入变化，'
        '跨来源对比是完整流程敏感性，不是仅替换一个覆盖字段的因果效应。', '',
        '来源：`summary.csv`、`coverage_cross_evaluation.csv`、`diagnostic_cross_evaluation.csv`、`source_thresholds.csv`。', '',
        '本轮重新求解的P20另记录停止状态，避免将可行解或原集合回退描述成已证明最优：', '',
        '| SenseNova选样门槛 | 停止状态 | 次数／300 |', '|---|---|---:|',
    ]
    solver_runs = study.read(out / 'selection_runs.csv')
    for mode in ['original', 'recalibrated']:
        rr = [r for r in solver_runs if r['source'] == 'B' and
              r['threshold_mode'] == mode and r['method'] == 'P20']
        assert len(rr) == 300
        for reason, count in sorted(Counter(r['stop_reason'] for r in rr).items()):
            lines.append(f'| {mode} | {reason} | {count} |')
    lines += ['', 'initial_pass表示初始集合已经达标；lexicographic_optimal表示两阶段均证明最优；'
        'infeasible表示门槛约束不可行并返回原集合；其他状态保留原求解器含义。'
        'Qwen选集直接复用冻结历史结果，未把它们计为本轮新求解。来源：`selection_runs.csv`。', '',
        '若保持原Qwen门槛数值完全相同，P20的跨标签达标计数如下；它与按各自参照重标定的结果分开解释：', '',
        '| 选样来源／策略门槛 | 用Qwen标签核对原门槛 | 用SenseNova标签核对同一原门槛 |',
        '|---|---:|---:|',
    ]
    for s, mode, label in [('A', 'original', 'Qwen／原门槛'), ('B', 'original', 'SenseNova／原门槛'),
                            ('B', 'recalibrated', 'SenseNova／重标定')]:
        aa = lookup[s, mode, 'A', 'A', 'P20']
        bb = lookup[s, mode, 'B', 'B', 'P20']
        lines.append(f"| {label} | {aa['pass_original_count']}/300 | {bb['pass_original_count']}/300 |")
    lines += ['',
        '逐类保留P20达标情况（以各自评价源的门槛判断），不以总达标率掩盖失败类别：', '',
        '| 类别 | Qwen选样／Qwen评价 | SenseNova原门槛选样／自身评价 | SenseNova重标定选样／自身评价 |',
        '|---|---:|---:|---:|',
    ]
    byclass = study.read(out / 'by_class.csv')
    for c in study.CLASSES:
        values = []
        for s, mode, evaluator in [('A', 'original', 'A'), ('B', 'original', 'B'), ('B', 'recalibrated', 'B')]:
            r = next(r for r in byclass if (r['selection_source'], r['threshold_mode'], r['evaluator'],
                     r['class_name'], r['method']) == (s, mode, evaluator, c, 'P20'))
            values.append(r['pass_evaluator_count'] + '/50')
        lines.append(f"| {names[c]} | {' | '.join(values)} |")
    lines += ['', '来源：`by_class.csv`。', '',
        '### 同约束增量与50起点的方向', '',
        '| 选样／评价／历史参考标签 | 门槛 | 比较 | 重复条件数均值差 | 正／平／负起点 |',
        '|---|---|---|---:|---:|',
    ]
    paired = study.read(out / 'paired_diagnostic_differences.csv')
    for r in paired:
        if r['selection_source'] == r['evaluator'] == r['reference_label_source'] and (
            (r['selection_source'], r['threshold_mode']) in [('A', 'original'), ('B', 'recalibrated')]):
            lines.append(f"| {'Qwen' if r['selection_source'] == 'A' else 'SenseNova'} | {r['threshold_mode']} | "
                f"{r['method']} − {r['comparator']} | {float(r['mean_confirmed_difference']):+.2f} | "
                f"{r['positive_starts']}／{r['ties']}／{r['negative_starts']} |")
    lines += ['', '50个起点来自固定数据池，不当作50个独立数据集；本轮不据此宣称统计显著或来源等价。'
        '交叉来源提案／历史复核四种组合均保留在明细中，避免只展示同源评价。', '',
        '## 4. 可支持的结论与边界', '',
        '- 原生字段的一致性与风险分层结果表明，当前自动标签来源具有明显敏感性，不能把两套标签当作可互换真值。',
        f"- 固定原Qwen选集时，P20相对Greedy20的历史重复弱条件数差，从Qwen评价下的"
        f"{float(original_p['confirmed_per_start']) - float(original_g['confirmed_per_start']):+.2f}条，"
        f"变为SenseNova评价下的{float(cross_p['confirmed_per_start']) - float(cross_g['confirmed_per_start']):+.2f}条。"
        '因此，本轮不支持P20在不同VLM评价下都优于同改动预算Greedy20的强结论。',
    ]
    for s, mode, name in [('A', 'original', 'Qwen'), ('B', 'recalibrated', 'SenseNova')]:
        r = lookup[s, mode, s, s, 'R']
        p = lookup[s, mode, s, s, 'P20']
        g = lookup[s, mode, s, s, 'Greedy20']
        support_delta = float(p['support_completion_pct']) - float(r['support_completion_pct'])
        confirmed_delta = float(p['confirmed_per_start']) - float(r['confirmed_per_start'])
        pg = float(p['confirmed_per_start']) - float(g['confirmed_per_start'])
        lines.append(f'- {name}标签下，P20相对R的支持完成度变化为{support_delta:+.2f}个百分点，'
                     f'历史重复弱条件变化为{confirmed_delta:+.2f}条；P20相对Greedy20为{pg:+.2f}条。')
    lines += [
        '- 本轮验证的是历史数据上的跨VLM稳健性；没有语义真值，所以不报告准确率，也不宣布分歧已经裁决。',
        '- Objects365已用于历史分析；本轮重新标注或重算不能使其成为未见数据验证。',
        '- 风险配额受控不保证实际错误率相同；重复弱条件更多也可能受错误数量和可评价支持变化影响，不证明失效根因种类增加。',
        '- 标签压力测试、跨VLM复核、新数据验证回答不同问题；本轮只补齐跨VLM证据。', '',
        '## 复算与检查', '',
        '```bash',
        '.venv/bin/python scripts/verify_cross_vlm_20261112.py prepare',
        '.venv/bin/python scripts/verify_cross_vlm_20261112.py annotate --workers 8',
        '.venv/bin/python scripts/repair_cross_vlm_format_20261112.py',
        '.venv/bin/python scripts/repair_cross_vlm_constrained_20261112.py',
        '.venv/bin/python scripts/verify_cross_vlm_20261112.py analyze',
        '.venv/bin/python scripts/verify_cross_vlm_20261112.py audit',
        '.venv/bin/python scripts/report_cross_vlm_20261112.py',
        '```', '',
        'prepare与annotate依赖本地冻结缓存、原图及已配置接口；已完成的独立标注会通过哈希和字典校验后复用。'
        '分析仅使用保存的标签和分类器预测。API限流及格式失败保留在原始响应和研究日志中。', '',
        f"检查通过：{validation['coverage_rows_independently_recomputed']:,}条覆盖记录独立复算，"
        f"{validation['diagnostic_aggregates_recomputed']}组诊断汇总复算，原Qwen历史指标与计数复现，输入／标签矩阵／选集／预测哈希核对通过。", '',
        '产物目录：`artifacts/cross_vlm_verification_20261112/`。',
    ]
    text = '\n'.join(lines) + '\n'
    (out / 'report_zh.md').write_text(text)
    doc = ROOT / 'docs/实验方案与结果/跨VLM核验结果_20261112.md'
    doc.write_text(text)
    (out / 'claim_evidence.md').write_text(
        '# 主张与证据\n\n'
        '| 主张 | 证据 | 边界 |\n|---|---|---|\n'
        '| 跨VLM字段一致性及风险分层敏感性 | native_field_agreement.csv、specific_field_agreement.csv、risk_summary.csv | 不等于准确率 |\n'
        '| 同约束方法在两套标签下的覆盖与诊断变化 | summary.csv、paired_diagnostic_differences.csv、by_class.csv | 固定历史池、描述性对照 |\n'
        '| 数值可追溯 | validation.json、result_traceability.json、各freeze清单 | 不证明语义正确或新数据泛化 |\n')
    (out / 'review.md').write_text(
        '# 核验审查\n\n'
        '- 引用完整性：通过；本轮未新增外部事实或文献主张。\n'
        '- 数值追溯：通过；原历史指标复现、覆盖逐行独立复算、诊断汇总复算、冻结输入核对均通过。\n'
        '- 主张范围：通过；仅报告自动标签跨来源一致性与历史流程敏感性，不声称语义准确率、因果作用或fresh泛化。\n'
        '- 语义真值：没有建立；按用户要求全自动执行，不把人工参与列为本任务未完成项。\n'
        '- 仍需独立研究：未见数据泛化、模型别名版本与历史提示语义的影响。\n')
    print(doc)


if __name__ == '__main__':
    run()
