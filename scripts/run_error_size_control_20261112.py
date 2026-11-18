"""Same-N-within-Q control for the outcome-matched historical auxiliary check."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import run_risk_error_control_20261112 as r

ROOT,cc,study=r.ROOT,r.cc,r.study
OUT=r.OUT/'size_control'


def take_sizes(original,q,sizes,ranks):
    chosen={}
    for method,ix in original.items():
        result=[]
        for tier in range(1,6):
            members=ix[q[ix]==tier];n=sizes[tier]
            result.extend(members[np.argsort(ranks[members],kind='stable')[:n]].tolist())
        chosen[method]=np.array(result,dtype=int)
        assert set(chosen[method])<=set(ix)
        assert [int((q[chosen[method]]==tier).sum()) for tier in range(1,6)]==[sizes[tier] for tier in range(1,6)]
    return chosen


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'completion.json').exists()
    for name in ['input_freeze.json','primary_output_freeze.json','matched_output_freeze.json']:
        cc.verify_freeze(r.OUT/name)
    cc.freeze(OUT/'protocol.json',dict(
        hypothesis='A loss of diagnostic gain under outcome matching may arise from fewer samples; separate it with equal-N-Q uniform subsampling',
        timing='Added after observing outcome-matching results; exploratory control, not preregistered confirmation',
        scope='Same two anchors,50starts,6classes,5models,3prediction seeds,20draws; existing selections and caches only',
        phases=['original','size_only','outcome_matched'],
        size_control='Use exactly the N per Q imposed by outcome matching, rank members uniformly within Q without consulting their individual correctness; shared ranks with outcome matching',
        note='N per Q itself still derives from original outcome counts; size-only control is not completely outcome-blind',
        universe='All3 original + all3 size-only + all3 outcome-matched methods and D reference evaluable, per label axis and draw',
        diagnostic='Single prediction seed, support20 each arm, gap<=-10pp in B and unchanged same-seed D; not primary three-seed mean diagnostic',
        aggregation='Sum class/model counts within each start,predseed,draw; average3predseeds x20draws inside start, then50starts; no independent-draw inference',
        RNG='Same SeedSequence and random image ranks as previous outcome matching',
        stop='Complete fixed budget; no thresholds, labels, seeds or sample sizes tuned; preserve all unfavorable results',
        caution='Conditional subsampling changes target populations and fine quotas; cannot prove causal independence or semantic truth'))
    paths=[Path(__file__),ROOT/'tests/test_error_size_control.py',Path(r.__file__),
        r.OUT/'matched_output_freeze.json',r.OUT/'input_freeze.json',r.previous.OUT/'selections.npz']
    cc.freeze(OUT/'input_freeze.json',dict(files={str(p):study.sha(p) for p in paths}))
    (OUT/'research_log.md').write_text('在错误数匹配结果已知后增加固定Q样本量对照；先冻结对照规则，所有锚点/起点/标签结果保留。\n')
    risks,raw=r.load();z=np.load(r.previous.OUT/'selections.npz',allow_pickle=False)
    totals=defaultdict(lambda:defaultdict(float));units=0
    for ai,(risk,cover,mode) in enumerate(r.ANCHORS):
        for ci,c in enumerate(r.previous.CLASSES):
            q=risks['B',risk,c]['tiers'];correctness=raw['B','A',c][2]
            for seed in r.previous.SEEDS:
                original={m:z[r.previous.key(risk,cover,mode,c,m,seed)] for m in r.METHODS}
                for mi,model in enumerate(r.old.MODELS):
                    for pi,pseed in enumerate(cc.old.SEEDS):
                        correct=correctness[model][pi];original_stats={};d_stats={}
                        for view in r.VIEWS:
                            x,y,_=raw['B',view,c];dx,dy,dc=raw['D',view,c]
                            d_stats[view]=r.previous.prior.reference_stats(dx,dy,dc[model][pi].astype(float))
                            for m,ix in original.items():
                                original_stats[view,m]=r.previous.prior.reference_stats(x[ix],y[ix],correct[ix].astype(float))
                        for repeat in range(r.REPEATS):
                            rng=np.random.default_rng(np.random.SeedSequence([2026111207,ai,seed,ci,mi,pi,repeat]))
                            ranks=rng.random(len(q));matched,cells=r.match_outcomes(original,q,correct,ranks)
                            sizes={tier:sum(n for qq,truth,n in cells if qq==tier) for tier in range(1,6)}
                            sized=take_sizes(original,q,sizes,ranks)
                            sets=dict(original=original,size_only=sized,outcome_matched=matched)
                            stats=dict(original=original_stats)
                            for phase in ['size_only','outcome_matched']:
                                stats[phase]={}
                                for view in r.VIEWS:
                                    x,y,_=raw['B',view,c]
                                    for m,ix in sets[phase].items():
                                        stats[phase][view,m]=r.previous.prior.reference_stats(x[ix],y[ix],correct[ix].astype(float))
                            for ev,ref in r.AXES:
                                universe=np.logical_and.reduce([stats[phase][ev,m][3] for phase in sets for m in r.METHODS])&d_stats[ref][3]
                                for phase in sets:
                                    for m,ix in sets[phase].items():
                                        confirmed=universe&stats[phase][ev,m][4]&d_stats[ref][4]
                                        key=(risk,cover,mode,seed,pseed,repeat,ev,ref,phase,m)
                                        t=totals[key];t['confirmed']+=int(confirmed.sum());t['common_n']+=int(universe.sum())
                                        t['mean_selected_n']+=len(ix)/30
                                        t['error_pct']+=100*float((~correct[ix]).mean())/30 if len(ix) else 0
                            units+=1
            cc.log(r.OUT,'size_control_class',risk=risk,coverage=cover,threshold_mode=mode,class_name=c,units=units)
    names=['risk_source','coverage_source','threshold_mode','selection_seed','prediction_seed','repeat','evaluator','reference','phase','method']
    draw=[dict(zip(names,key),**v) for key,v in totals.items()]
    cc.write(OUT/'per_draw.csv',draw)
    grouped=defaultdict(list)
    base_names=['risk_source','coverage_source','threshold_mode','selection_seed','evaluator','reference','phase','method']
    for row in draw:grouped[tuple(row[f] for f in base_names)].append(row)
    averages={}
    for key,rr in grouped.items():
        assert len(rr)==3*r.REPEATS
        averages[key]={f:float(np.mean([v[f] for v in rr])) for f in ['confirmed','common_n','mean_selected_n','error_pct']}
    cc.write(OUT/'per_start.csv',[dict(zip(base_names,key),**v) for key,v in averages.items()])
    summary=[]
    for risk,cover,mode in r.ANCHORS:
        for ev,ref in r.AXES:
            for phase in ['original','size_only','outcome_matched']:
                for method,comparator in r.COMPARISONS:
                    aa=[averages[risk,cover,mode,s,ev,ref,phase,method] for s in r.previous.SEEDS]
                    bb=[averages[risk,cover,mode,s,ev,ref,phase,comparator] for s in r.previous.SEEDS]
                    for a,b in zip(aa,bb):
                        assert abs(a['common_n']-b['common_n'])<1e-9
                        if phase=='outcome_matched':assert abs(a['error_pct']-b['error_pct'])<1e-9
                    delta=np.array([a['confirmed']-b['confirmed'] for a,b in zip(aa,bb)])
                    summary.append(dict(risk_source=risk,coverage_source=cover,threshold_mode=mode,evaluator=ev,reference=ref,
                        phase=phase,method=method,comparator=comparator,mean_difference=float(delta.mean()),
                        common_n=float(np.mean([v['common_n'] for v in aa])),mean_selected_n=float(np.mean([v['mean_selected_n'] for v in aa])),
                        method_error_pct=float(np.mean([v['error_pct'] for v in aa])),comparator_error_pct=float(np.mean([v['error_pct'] for v in bb])),
                        positive_starts=int((delta>1e-10).sum()),ties=int((np.abs(delta)<=1e-10).sum()),negative_starts=int((delta<-1e-10).sum())))
    cc.write(OUT/'summary.csv',summary)
    cc.dump(OUT/'validation.json',dict(status='pass',matched_units=units,Q_sizes_equal_in_size_and_outcome_controls=True,
        all_members_from_original_sets=True,shared_state_universe_all3phases_and_methods=True,
        exact_error_rate_match_only_claimed_for_outcome_phase=True))
    make_report(summary)
    cc.verify_freeze(OUT/'input_freeze.json')
    cc.freeze(r.OUT/'size_control_output_freeze.json',dict(files={str(p):study.sha(p) for p in OUT.glob('*') if p.is_file()}))
    cc.dump(OUT/'completion.json',dict(status='complete',time=cc.old.now(),new_VLM_calls=0))


def make_report(summary):
    idx={tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','evaluator','reference','phase','method','comparator']):v for v in summary}
    lines=['# 错误数匹配的相同样本量对照','',
        '在先前错误数匹配结果已知后增加的探索性控制：原选集、仅按Q减少到相同人数的均匀随机子集、Q内正确数/错误数都匹配的子集。后两者逐Q人数完全相同。使用同一图像随机排序，样本量对照不按单个成员是否正确来筛选；但每Q目标人数本身仍来自原错误数差异。','',
        '两锚点、50起点、6类、5分类器、3个预测种子、20次抽样，预算与原辅助检查一致；全部使用单预测种子B/D诊断。条件集合取三方法×三种选集与D均可评价的交集，因此本轮数值与先前较宽共同集合不能直接相减。','',
        '| 锚点 | 评价 | 比较 | 原选集 | 仅匹配Q内人数 | 匹配Q内错误/正确数 |','|---|---|---|---:|---:|---:|']
    for risk,cover,mode in r.ANCHORS:
        for ev in r.VIEWS:
            for method,comp in [('P20','R'),('P20','Greedy20')]:
                values=[idx[risk,cover,mode,ev,ev,phase,method,comp] for phase in ['original','size_only','outcome_matched']]
                lines.append(f'| {risk}/{cover}/{mode} | {ev}/{ev} | {method}−{comp} | '+
                    ' | '.join(f'{float(v["mean_difference"]):+.2f}' for v in values)+' |')
    lines+=['','表内为平均每起点确认条件净差，先跨6类×5分类器求和，再平均3预测种子与20次重抽、50起点。不是独立根因数。完整交叉标签、配对起点、错误率与N均保存在CSV。','',
        '原选集→人数对照检验减少样本及组内随机波动的影响；人数对照→结果匹配对比在相同N/Q构成下增加结果匹配约束的变化。后者仍是模型结果条件化抽样，不能证明错误率是唯一原因或建立因果独立性，也不能将共识当作真值。','',
        '复现：.venv/bin/python scripts/run_error_size_control_20261112.py；使用已冻结历史输入，不修改原选集。']
    text='\n'.join(lines)+'\n';(OUT/'report_zh.md').write_text(text)
    doc=ROOT/'docs/实验方案与结果/错误数匹配的样本量对照_20261112.md';doc.write_text(text)
    cc.freeze(OUT/'report_doc_freeze.json',dict(files={str(doc):study.sha(doc)}))


if __name__=='__main__':run()
