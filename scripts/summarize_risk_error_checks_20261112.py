"""Independent audit and concise synthesis including the size-only control."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import run_risk_error_control_20261112 as r

OUT,cc,study=r.OUT,r.cc,r.study
CONTROL=OUT/'size_control'


def audit():
    cc.freeze(OUT/'synthesis_code_freeze.json',dict(files={str(Path(__file__)):study.sha(Path(__file__))}))
    cc.verify_freeze(OUT/'size_control_output_freeze.json')
    cc.verify_freeze(CONTROL/'report_doc_freeze.json')
    rows=study.read(CONTROL/'per_draw.csv');grouped=defaultdict(list);same={};sizes={};outcomes={}
    base=['risk_source','coverage_source','threshold_mode','selection_seed','prediction_seed','repeat','evaluator','reference']
    fields=['risk_source','coverage_source','threshold_mode','selection_seed','evaluator','reference','phase','method']
    for v in rows:
        unit=tuple(v[f] for f in base);n=float(v['common_n'])
        if unit in same:assert same[unit]==n
        else:same[unit]=n
        if v['phase']!='original':
            nn=float(v['mean_selected_n'])
            if unit in sizes:assert abs(sizes[unit]-nn)<1e-10
            else:sizes[unit]=nn
        if v['phase']=='outcome_matched':
            e=float(v['error_pct'])
            if unit in outcomes:assert abs(outcomes[unit]-e)<1e-10
            else:outcomes[unit]=e
        grouped[tuple(v[f] for f in fields)].append(v)
    averages={}
    for key,rr in grouped.items():
        assert len(rr)==3*r.REPEATS
        averages[key]={f:sum(float(v[f]) for v in rr)/len(rr) for f in ['confirmed','common_n','mean_selected_n','error_pct']}
    for v in study.read(CONTROL/'per_start.csv'):
        key=tuple(v[f] for f in fields)
        for f in averages[key]:assert abs(float(v[f])-averages[key][f])<1e-9
    summary=study.read(CONTROL/'summary.csv')
    for v in summary:
        difference=[]
        for seed in r.previous.SEEDS:
            key=(v['risk_source'],v['coverage_source'],v['threshold_mode'],str(seed),v['evaluator'],v['reference'],v['phase'])
            a=averages[key+(v['method'],)];b=averages[key+(v['comparator'],)]
            difference.append(a['confirmed']-b['confirmed'])
        d=np.array(difference)
        assert abs(d.mean()-float(v['mean_difference']))<1e-9
        assert int((d>1e-10).sum())==int(v['positive_starts'])
        assert int((np.abs(d)<=1e-10).sum())==int(v['ties'])
        assert int((d<-1e-10).sum())==int(v['negative_starts'])
    cc.dump(OUT/'size_control_independent_audit.json',dict(status='pass',per_draw_rows=len(rows),
        per_start_groups=len(grouped),summary_rows=len(summary),shared_universe_units=len(same),
        same_N_in_size_and_outcome_controls=True,same_error_rate_in_outcome_control=True))
    return summary


def run():
    assert not (OUT/'completion.json').exists()
    summary=audit()
    idx={tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','evaluator','reference','phase','method','comparator']):v for v in summary}
    primary=study.read(OUT/'primary_summary.csv')
    pidx={tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','evaluator','reference','path','method','comparator']):v for v in primary}
    lines=['# 风险构成与错误率核验：阶段结论','',
        '本轮自动核验完成风险标准化、Q层内错误/正确数匹配，以及错误匹配的同样本量对照。只使用历史冻结选集、VLM标签和分类器预测；没有人工核验、新模型调用或训练。结果属于探索性敏感性分析。','',
        '## 风险层构成不能统一解释剩余收益','',
        '主分析在相同的共同可评价状态上比较原池化、保留风险层后池化及风险标准化。各方法使用相同保留风险层与候选池状态权重，原A历史调整记录全部复现。','',
        '| 锚点/评价 | P20−R原池化 | 保留层后池化 | 风险标准化 |','|---|---:|---:|---:|']
    for risk,cover,mode,ev in [('A','A','original','A'),('A','A','original','B'),('B','B','recalibrated','B'),('B','B','recalibrated','C')]:
        vv=[pidx[risk,cover,mode,ev,ev,path,'P20','R'] for path in ['pooled','retained','standardized']]
        lines.append(f'| {risk}/{cover}/{mode}，{ev}/{ev} | '+' | '.join(f'{float(v["mean_difference"]):+.2f}' for v in vv)+' |')
    lines+=['','调整既可能增大也可能减小差异，因此不支持“剩余收益都只是Q层构成”的解释。主分析的风险Q控制并非实际错误率匹配，也不是因果控制。','',
        '## 错误数量约束与单纯样本量减少的对照','',
        '辅助检查在每类/分类器/预测种子/Q内对齐三方法的错误数、正确数。新增同人数对照按同一随机图像排序仅抽取同样Q内人数，不按成员本人的正确与否筛选。三种选集和三种方法在同一组可评价状态上比较。','',
        '| 锚点 | 评价 | P20−R原选集 | 同人数随机子集 | 错误/正确数匹配 |','|---|---|---:|---:|---:|']
    effects=[]
    for risk,cover,mode in r.ANCHORS:
        for ev in r.VIEWS:
            vv=[idx[risk,cover,mode,ev,ev,phase,'P20','R'] for phase in ['original','size_only','outcome_matched']]
            vals=[float(v['mean_difference']) for v in vv];effects.append(vals)
            lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | '+' | '.join(f'{v:+.2f}' for v in vals)+' |')
    reduced=sum(v[2]<v[1]-1e-10 for v in effects)
    negative=sum(v[2]<-1e-10 for v in effects)
    lines+=['',f'在上表6种同源/共识口径中，增加错误数量约束后相对同人数对照的差值下降{reduced}/6；错误匹配后的P20−R差值为负{negative}/6。六种口径共享数据与模型，不能解释为独立重复、胜率或显著性证据。','',
        '## P20相对Greedy20仍应保留口径限制','',
        '| 锚点 | 评价 | 原选集P−G | 同人数对照P−G | 错误匹配P−G |','|---|---|---:|---:|---:|']
    for risk,cover,mode in r.ANCHORS:
        for ev in r.VIEWS:
            vv=[idx[risk,cover,mode,ev,ev,phase,'P20','Greedy20'] for phase in ['original','size_only','outcome_matched']]
            lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | '+' | '.join(f'{float(v["mean_difference"]):+.2f}' for v in vv)+' |')
    lines+=['','所有表格的净差都是平均每起点确认条件数，非独立根因数。主分析使用三预测种子平均正确率；辅助检查使用单预测种子诊断，再平均3种子×20抽样×50起点。两类数字不直接相减。','',
        '## 允许的结论与下一步','',
        '现有确认条件收益不能直接解读为独立于支持量、风险构成或错误数量的更强诊断能力。继续保留提高可评价性的限定观察，以及每种口径下的负结果；目前不恢复“P20稳定优于Greedy20”的强主张。','',
        '错误匹配按模型结果抽样，改变了样本成员、N和细配额；同人数对照的N本身也来自原错误数差异。即使两者不同，也只能说明该条件化抽样约束的敏感性，不能证明错误率是唯一原因。标签共识不等于语义真值，历史D不是新数据。','',
        '下一阶段应先固定自动标签协议和主指标，再进入未使用数据的验证。可评价状态数与共同状态上的诊断收益须分开，随机与Greedy20同时作为对照，并预先报告实际错误率；不再通过调整历史提示词或挑状态来争取有利排序。','',
        '详细记录：', '',
        f'- [风险/错误匹配主报告]({r.ROOT}/docs/实验方案与结果/风险分层与实际错误率敏感性复验_20261112.md)',
        f'- [同人数对照]({r.ROOT}/docs/实验方案与结果/错误数匹配的样本量对照_20261112.md)',
        f'- [原始结果目录]({OUT})']
    text='\n'.join(lines)+'\n';(OUT/'interpretation_zh.md').write_text(text)
    doc=r.ROOT/'docs/实验方案与结果/风险与错误率核验阶段结论_20261112.md';doc.write_text(text)
    cc.freeze(OUT/'synthesis_doc_freeze.json',dict(files={str(doc):study.sha(doc)}))
    cc.dump(OUT/'synthesis_decision.json',dict(size_control_comparisons=6,outcome_constraint_lowers_P20_minus_R=reduced,
        matched_P20_minus_R_negative=negative,causal_error_independence_established=False,stable_P20_superiority_supported=False))


if __name__=='__main__':run()
