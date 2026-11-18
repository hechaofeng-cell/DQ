"""Bounded interpretation of pooled, standardized, and outcome-matched checks."""
import json
from collections import defaultdict
import numpy as np
import run_risk_error_control_20261112 as r

OUT,cc,study=r.OUT,r.cc,r.study


def run():
    assert not (OUT/'completion.json').exists()
    for name in ['input_freeze.json','primary_output_freeze.json','matched_output_freeze.json','prediction_freeze.json']:
        cc.verify_freeze(OUT/name)
    primary=study.read(OUT/'primary_summary.csv');matched=study.read(OUT/'matched_summary.csv')
    groups=['risk_source','coverage_source','threshold_mode','evaluator','reference']
    p={tuple(v[f] for f in groups+['path','method','comparator']):v for v in primary}
    a={tuple(v[f] for f in groups+['phase','method','comparator']):v for v in matched}
    errors=defaultdict(list)
    for v in study.read(OUT/'actual_error.csv'):
        if v['Q']=='0':
            errors[tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','method'])].append(float(v['error_pct']))
    error_means={k:float(np.mean(v)) for k,v in errors.items()}
    r.write('actual_error_summary.csv',[dict(zip(['risk_source','coverage_source','threshold_mode','method'],k),mean_error_pct=v) for k,v in error_means.items()])
    def primary_at(risk,cover,mode,ev,path,method='P20',comparator='R'):
        return p[risk,cover,mode,ev,ev,path,method,comparator]
    def aux_at(risk,cover,mode,ev,phase,method='P20',comparator='R'):
        return a[risk,cover,mode,ev,ev,phase,method,comparator]
    verdict={}
    for method,comparator in r.COMPARISONS:
        verdict[method+'_minus_'+comparator]={path:sum(float(v['mean_difference'])>1e-10 for v in primary
            if (v['method'],v['comparator'],v['path'])==(method,comparator,path)) for path in r.PATHS}
    pv=json.loads((OUT/'primary_validation.json').read_text());av=json.loads((OUT/'matched_validation.json').read_text())
    lines=['# 风险构成与实际错误率敏感性复验','',
        '本轮接续共同可评价条件分解，检验剩余确认条件增益是否随风险层重加权、或对齐实际错误数与正确数而改变。所有工作使用历史冻结选集、两套VLM/共识标签及缓存分类器预测，没有新VLM调用、重新选样训练或人工核验。','',
        '## 主分析：同一共同条件集合上的风险调整','',
        '8个冻结风险/覆盖/门槛场景，6类、50个固定起点、R/Greedy20/P20、5分类器、3预测种子平均正确率。每个场景采用其选样风险来源的B/D类内Q1–Q5。','',
        '每状态只保留三方法都至少有5个状态阳性和5个已知同字段阴性的Q层；各方法保留阳性/阴性均≥20且保留≥80%原阳性。D参考独立满足相同规则。共同可评价状态取三方法与D的交集；这些支持规则不看正确率。调整前后使用完全相同的状态集合。','',
        '“原池化”使用完整选集中每侧的正确率；“保留后池化”仅使用保留Q中的目标、维持实际组内构成；“风险标准化”在每Q计算两侧差，再用完整候选池的状态阳性Q分布加权。保留Q和权重在三方法间相同；两侧也使用相同权重。B与D各有自己的固定权重。','',
        '从原池化到保留后池化会排除样本；从保留后池化到标准化才是同一保留目标上的权重变化。主分析仍保留全部冻结选集；状态特定风险层筛选只用于诊断指标。低支持条件可能被排除，不能把调整前后的本轮数值与上一轮更宽状态集合的均值直接相减。','',
        '| 比较 | 原池化净差为正 | 保留后池化净差为正 | 风险标准化净差为正 |','|---|---:|---:|---:|']
    for name,v in verdict.items():
        lines.append(f'| {name} | {v["pooled"]}/40 | {v["retained"]}/40 | {v["standardized"]}/40 |')
    lines+=['','40个口径包含5种发现/参考标签组合与8个场景，彼此高度相关且AA两门槛重复；这不是独立实验胜率或显著性检验。','',
        '| 锚点 | 评价标签 | 共同条件数 | P−R原池化 | P−R保留后池化 | P−R标准化 | P−G原池化 | P−G标准化 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for risk,cover,mode in r.ANCHORS:
        for ev in r.VIEWS:
            pp=[primary_at(risk,cover,mode,ev,path) for path in ['pooled','retained','standardized']]
            gg=[primary_at(risk,cover,mode,ev,path,'P20','Greedy20') for path in ['pooled','standardized']]
            lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | {float(pp[0]["common_n"]):.2f} | '+
                ' | '.join(f'{float(v["mean_difference"]):+.2f}' for v in pp+gg)+' |')
    lines+=['','表中净差为每起点确认条件数：先跨6类×5分类器求和，再对50起点平均。条件数与根因数、标注正确数不是同一概念。发现端或参考端单独标准化的结果也完整保留在primary_summary.csv。','',
        '## 原选集的实际错误率','',
        '| 风险/覆盖/门槛 | R错误率 | Greedy20错误率 | P20错误率 |','|---|---:|---:|---:|']
    for risk in ['A','B']:
        for cover in ['A','B']:
            for mode in r.previous.MODES:
                vals=[error_means[risk,cover,mode,m] for m in r.METHODS]
                lines.append(f'| {risk}/{cover}/{mode} | '+' | '.join(f'{v:.3f}%' for v in vals)+' |')
    lines+=['','原选集拥有相同的精确风险配额，但实际错误率仍可能不同。按Q的错误率及每组样本数保存在actual_error.csv。Q标准化并未对齐实际错误率，亦不能消除Q内部未观察难度差异或标签与风险构造的关联。','',
        '## 辅助检查：Q层内对齐实际错误与正确数量','',
        '固定AA原门槛与BB重标定两个锚点；全部50起点、6类、5分类器、3个预测种子，各20次确定性重抽。在每个类别/分类器/预测种子/Q内，分别取三方法中最少的错误数和最少的正确数，按共享随机图像排序各自抽样。因此每个Q内三方法的错误数、正确数和总样本数完全相同。','',
        '这一步使用已知分类器结果进行抽样，会减少N并改变细风险配额，Q间份额也可能变化。诊断改用单个预测种子的正确率，D参考保持同一预测种子与全部原D目标。它不是主分析的三种子平均诊断规则。只能在辅助检查内部比较原选集与匹配子集，不能把两类分析的数值直接相减，更不能把匹配结果解释为因果识别。','',
        '每次重抽只比较三方法原选集、三方法匹配子集及D都可评价的状态ID；参考弱条件定义不变。该共同集合在“原选集”和“匹配子集”之间一致，但因抽样依赖模型输出，辅助检查的状态集合也并非预测盲。所有抽样与负结果保留，不把20次重抽视为20个独立数据集。','',
        '| 锚点 | 评价 | P−R原选集 | P−R匹配后 | P−G原选集 | P−G匹配后 | 每类/模型匹配N | 匹配错误率（三方法相同） |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for risk,cover,mode in r.ANCHORS:
        for ev in r.VIEWS:
            pp=[aux_at(risk,cover,mode,ev,phase) for phase in ['original','matched']]
            gg=[aux_at(risk,cover,mode,ev,phase,'P20','Greedy20') for phase in ['original','matched']]
            lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | '+
                ' | '.join(f'{float(v["mean_difference"]):+.2f}' for v in pp+gg)+
                f' | {float(pp[1]["mean_selected_n"]):.2f} | {float(pp[1]["method_error_pct"]):.3f}% |')
    lines+=['','辅助指标先在每起点/预测种子/重抽内跨6类×5分类器求和，再依次平均3预测种子与20次重抽、50起点。matched_per_draw.csv保留重抽、预测种子与起点维度，matched_per_start.csv保留50个配对起点；共同条件数及支持量损失均在matched_summary.csv中。','',
        '## 核验与结论边界','',
        f'主分析输出{pv["primary_rows"]}行，复现上一轮原诊断{pv["original_raw_diagnostic_rows_reproduced"]}行及原A历史风险调整{pv["legacy_A_adjusted_rows_reproduced"]}行。{pv["scalar_gap_bundles_checked"]}组差值另用逐状态均值核算。辅助检查{av["balance_units"]}个匹配单元均验证Q×正确/错误数量、成员来源与三方法错误率一致；空匹配{av["empty_matches"]}个，均未静默删除。','',
        '该实验能说明历史诊断结论对风险构成和结果条件化抽样是否敏感。它不能确定哪个VLM更准确，不能证明实际错误率是唯一原因，也不能恢复跨VLM稳定优于Greedy20或新数据泛化的强主张。','',
        '后续若进入新数据验证，应预先固定自动标签协议、选样策略与两个分开的指标：可评价性收益，以及同一共同条件集合上的诊断收益。当前研究主线未因本轮结果被擅自改写。','',
        '复现：依次运行 scripts/run_risk_error_control_20261112.py 的prepare、primary、matched阶段，再运行 scripts/report_risk_error_control_20261112.py，均使用.venv/bin/python。完成目录不可覆盖。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text)
    doc=r.ROOT/'docs/实验方案与结果/风险分层与实际错误率敏感性复验_20261112.md';doc.write_text(text)
    r.dump('decision.json',dict(primary_positive_scenarios=verdict,primary_scenarios=40,
        stable_P20_superiority_supported=False,causal_error_independence_established=False,
        semantic_accuracy_established=False,auxiliary_is_outcome_conditioned=True))
    (OUT/'claim_evidence.md').write_text('风险调整的同集合差值：primary_summary.csv与primary_per_start.csv。实际错误率：actual_error_summary.csv。结果条件化匹配：matched_summary.csv、matched_per_start.csv、matched_balance.csv。所有结论限于历史敏感性，不将重抽当独立实验，不把同Q当实际错误率相同，不把匹配当因果识别。\n')
    (OUT/'review.md').write_text('引用完整性：未新增外部文献事实。结果追溯：历史原诊断和原A调整记录复现、逐状态标量核算、Q×结果匹配与成员检查通过。主张边界：主/辅分析、不同诊断估计量、支持量/样本群体变化、历史重复使用、非因果/非准确率/非新数据明确。\n')
    cc.log(OUT,'report_complete')
    outputs={str(p):study.sha(p) for p in OUT.glob('*') if p.is_file() and p.name not in ['completion.json','final_outputs_freeze.json']}
    outputs[str(doc)]=study.sha(doc)
    cc.freeze(OUT/'final_outputs_freeze.json',dict(files=outputs))
    r.dump('completion.json',dict(status='complete',time=cc.old.now(),new_VLM_calls=0,new_classifier_inference=False,
        output_freeze_sha256=study.sha(OUT/'final_outputs_freeze.json'),report=str(doc)))


if __name__=='__main__':run()
