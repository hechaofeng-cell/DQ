#!/usr/bin/env python3
"""Summarize development and locked results without changing the experiment."""
from collections import Counter
import json
from pathlib import Path

import run_label_protocol_pilot_20261112 as run


def fmt(value):
    return '不可评价' if value is None else f'{value:.2f}'


def make_report():
    out=run.OUT
    run.verify(out/'input_freeze.json');run.verify(out/'execution_freeze.json')
    run.verify(out/'analysis_code_freeze.json');run.verify(out/'final_prompt_freeze.json')
    run.verify(out/'report_code_freeze.json')
    data={}
    for split in ('development','locked_repair_check'):
        p=out/'analysis'/split
        assert json.loads((p/'validation.json').read_text())['status']=='pass'
        run.verify(p/'labels_freeze.json')
        data[split]=json.loads((p/'summary.json').read_text())
    dev=data['development'];locked=data['locked_repair_check']
    evaluable=[r for r in locked['primary_fields'] if r['engineering_screen'] not in
               ('near_ceiling_descriptive_only','insufficient_known_support')]
    promising=[r for r in evaluable if r['engineering_screen']=='promising_for_expansion']
    all_passed=bool(evaluable) and len(promising)==len(evaluable)
    lead=('本轮可评价优先字段全部达到预定工程筛查要求，值得后续扩展验证。' if all_passed else
          '本轮未满足全部可评价优先字段的预定改善要求，不建议直接进入全量重标。')
    complete=sum(s['completed_base_jobs'] for s in data.values())
    attempts=sum(s['actual_attempts'] for s in data.values())
    lines=['# 跨VLM协议修订小试验结果（2026-11-12）','',f'**{lead}**','',
        '本轮按固定240个历史目标执行，两套VLM分别使用旧语义协议和统一修订协议。开发组120个，锁定复核组120个；'
        '候选新版在开发组后保持不变，未用复核结果调提示、字典或评分。全程无人工核验，也没有让一个模型查看另一个模型的答案。','',
        f'960项基本任务均已执行，其中 **{complete}/960** 份答案有效，{960-complete}份在预定尝试上限内未通过校验，保留为缺失；'
        f'实际请求 **{attempts}** 次（包含失败重试）。'
        '模型为Qwen3-VL 30B与SenseNova 6.8 Flash Lite；两组保留相同字段和合法枚举，原始答案不在本地改值。','',
        '## 1. 完整一致率','',
        '| 数据组 | 命名空间 | 配对字段数 | 旧协议 | 修订协议 | 变化 |',
        '|---|---|---:|---:|---:|---:|']
    for split,name in [('development','开发'),('locked_repair_check','锁定复核')]:
        for ns,r in data[split]['micro_agreement'].items():
            delta=r['revised_exact_pct']-r['old_exact_pct']
            lines.append(f"| {name} | {ns} | {r['paired_field_instances']} | {r['old_exact_pct']:.2f}% | "
                         f"{r['revised_exact_pct']:.2f}% | {delta:+.2f}个百分点 |")
    lines+=['','native覆盖六类原生字段；specific为马、羊、人、车原冻结细粒度覆盖字段。'
        '这些完整一致率包含unknown，不等于语义准确率。只在四路答案完整的目标上作配对比较，缺失数量另列。'
        '本轮240目标分层样本、现时重新调用及统一拼图，与上轮3600目标历史缓存不同；'
        '不能把本轮比例相对历史61.12%的差异归因于提示修订，只比较本轮同图旧／新配对差。','',
        '## 2. 锁定复核组：主要字段','',
        '| 字段 | 旧／新双方已知数 | 旧／新已知一致率 | 变化及95%配对bootstrap区间（百分点） | 双方已知比例变化 | 筛查结果 |',
        '|---|---:|---:|---|---:|---|']
    for r in locked['primary_fields']:
        lines.append(f"| {r['field']} | {r['old_both_known_n']}／{r['revised_both_known_n']} | "
            f"{fmt(r['old_known_agreement_pct'])}%／{fmt(r['revised_known_agreement_pct'])}% | "
            f"{fmt(r['known_agreement_pct_delta_pp'])} [{fmt(r['known_agreement_pct_delta_ci_low'])}, "
            f"{fmt(r['known_agreement_pct_delta_ci_high'])}] | {fmt(r['both_known_pct_delta_pp'])} | {r['engineering_screen']} |")
    lines+=['',f'可评价优先字段{len(evaluable)}个，其中{len(promising)}个满足工程增幅与支持比例要求且描述性区间下限大于0。',
        '调用前已固定：双方已知一致率至少增加5个百分点，双方已知比例下降不超过5个百分点。'
        '旧双方已知少于30条或旧已知一致率至少95%时仅作描述。区间来自2,000次按源图配对、类别／来源分层bootstrap；'
        '没有把多个字段筛查当作经多重比较校正的科学优势检验。','',
        '## 3. unknown与同一已知子集','',
        '| 字段 | Qwen unknown旧／新 | SenseNova unknown旧／新 | 四路均已知数 | 同一已知子集一致率旧／新 |',
        '|---|---:|---:|---:|---:|']
    for r in locked['primary_fields']:
        lines.append(f"| {r['field']} | {fmt(r['old_A_unknown_pct'])}%／{fmt(r['revised_A_unknown_pct'])}% | "
            f"{fmt(r['old_B_unknown_pct'])}%／{fmt(r['revised_B_unknown_pct'])}% | {r['all_four_known_n']} | "
            f"{fmt(r['old_agreement_on_all_four_known_pct'])}%／{fmt(r['revised_agreement_on_all_four_known_pct'])}% |")
    lines+=['','同一已知子集仍由四路输出决定，不能当作无选择偏差的真值样本。全部分歧保留；没有用第三模型投票替代真值。','',
        '## 4. 风险评分与执行完整性','',
        '| 数据组 | 两模型分数平均绝对差：旧 | 两模型分数平均绝对差：新 | 四路完整目标 | 有效基本任务 |',
        '|---|---:|---:|---:|---:|']
    for split,name in [('development','开发'),('locked_repair_check','锁定复核')]:
        s=data[split]
        lines.append(f"| {name} | {fmt(s['risk_score_MAE']['old'])} | {fmt(s['risk_score_MAE']['revised'])} | "
                     f"{s['four_way_paired_targets']}/120 | {s['completed_base_jobs']}/480 |")
    lines+=['','分数采用原冻结规则（0–100），未依据分类器结果调权重；分数差缩小也不证明哪套标签正确。'
        '逻辑标志、失败次数、全部字段与逐类结果均在明细中保留。每个基本任务总尝试不超过3次，烟测计入开发组预算。','',
        '## 5. 一项具体的失败线索','',
        '整体可见性字段在两模型中的分布变化不同。以下只统计锁定组四路答案齐全的相同目标：','',
        '| 模型 | 协议 | full | most | partial | minimal | unknown |',
        '|---|---|---:|---:|---:|---:|---:|']
    paired=run.previous.read(out/'analysis/locked_repair_check/paired_values.csv')
    for source,name in [('A','Qwen'),('B','SenseNova')]:
        for arm,arm_name in [('old','旧'),('revised','新')]:
            counts=Counter(json.loads(r[source+'_value']) for r in paired
                if r['namespace']=='native' and r['field']=='body_visibility' and r['arm']==arm)
            cells=' | '.join(str(counts[k]) for k in ['full','most','partial','minimal','unknown'])
            lines.append(f'| {name} | {arm_name} | {cells} |')
    lines+=['', '这说明相同新版文字没有让两模型形成相同的可见性判定分布；不是靠统一提示就解决了歧义。'
        '它不能证明哪套分布更正确。下一步若继续研究，应把整体可见性的参照部位、缺失原因及等级判据拆成单字段的小对照，'
        '而非继续同时加长全部字段提示。头部可见性只在本轮锁定组达到筛查要求，不应据单个字段、多个筛查中的结果宣称整套协议已修复。','',
        '## 6. 可以怎样使用这轮结果','',
        '- 只能陈述统一提示在这240个历史目标上的字段一致性、未知率及风险分数敏感性变化。',
        '- 达标字段可作为下一步扩大验证的候选；未达标字段保留原失败结果，不能以合并类别、增加unknown或选择一致子集宣称修订成功。',
        '- 本轮没有执行P20／Greedy20重选或新分类器推理，不能宣称先前的方法优势或跨来源达标率已经恢复。',
        '- 后续方法复验仍须分别固定选集换评价标签、固定风险组织换覆盖标签、以及换完整来源；同数值门槛和来源独立重标定分开报告。',
        '- 这不是标签准确率验证。历史图像不因划分锁定组而变成未见数据；真正的新数据验证须另立冻结数据池。','',
        '## 7. 复查产物','',
        '- 样本与输入：manifest.csv、input_freeze.json、evidence_manifest.json。',
        '- 协议与模型：experiment_protocol.json、runtime.json、execution_freeze.json、final_prompt_freeze.json。',
        '- 开发／复核：analysis/development/、analysis/locked_repair_check/中的summary.json、field_comparison.csv、field_by_class.csv、paired_values.csv、logical_flags.csv及validation.json。',
        '- 实际响应：raw/；完整合法答案：labels/；失败记录：failures/。',
        '- 原图／目标、schema、prompt、实际响应与保存答案已核对并保留哈希；完整一致率与已知比例已从逐目标表独立复算。','',
        '复算命令：','', '```bash',
        '.venv/bin/python scripts/analyze_label_protocol_pilot_20261112.py --split development',
        '.venv/bin/python scripts/analyze_label_protocol_pilot_20261112.py --split locked_repair_check',
        '.venv/bin/python scripts/report_label_protocol_pilot_20261112.py','```','']
    text='\n'.join(lines)
    path=run.ROOT/'docs/实验方案与结果/跨VLM协议修订小试验结果_20261112.md'
    path.write_text(text);(out/'report_zh.md').write_text(text)
    run.dump(out/'pilot_decision.json',dict(lead=lead,eligible_primary_fields=len(evaluable),
        promising_primary_fields=len(promising),all_eligible_passed=all_passed,
        completed_base_jobs=complete,actual_attempts=attempts,
        automatically_expand_full_pool=False,method_advantage_reestablished=False,
        semantic_accuracy_established=False,fresh_generalization_test=False))
    (out/'claim_evidence.md').write_text(
        '# Claim evidence\n\n'
        '| Claim | Evidence | Boundary |\n|---|---|---|\n'
        '| Paired protocol consistency and unknown changes | analysis/*/field_comparison.csv and paired_values.csv | Historical balanced pilot; no semantic truth |\n'
        '| Engineering-screen decision | experiment_protocol.json and pilot_decision.json | Predetermined operational screen; no accuracy or universal advantage claim |\n'
        '| Traceability | analysis/*/validation.json, labels_freeze.json, source freezes | Exact raw-answer integrity and numeric aggregation, not semantic correctness |\n')
    (out/'review.md').write_text(
        '# Review\n\n'
        '- Citation integrity: pass; no new external factual claims or literature citations introduced.\n'
        '- Result traceability: pass; raw answer identity, source freezes, field agreement and support denominators audited.\n'
        '- Claim scope: pass; historical automatic consistency only, no semantic accuracy, fresh generalization or restored P20 superiority.\n'
        '- Negative/mixed fields and missing answers retained; no repeated tuning on locked outputs.\n')
    print(path)


if __name__=='__main__':make_report()
