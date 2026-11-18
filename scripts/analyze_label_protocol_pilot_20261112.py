#!/usr/bin/env python3
"""Paired source-image bootstrap and auditable report for the bounded pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import run_label_protocol_pilot_20261112 as run

OUT = run.OUT
UNKNOWN = {'unknown','not_observable','not_applicable','missing'}


def canonical(value):
    return tuple(sorted(value)) if isinstance(value,list) else value


def known(value):
    return not bool(set(value)&UNKNOWN) if isinstance(value,list) else value not in UNKNOWN


def field_vectors(records):
    arrays = {}
    for arm in ('old','revised'):
        a = [r[arm]['A'] for r in records]; b = [r[arm]['B'] for r in records]
        arrays[arm] = dict(same=np.array([canonical(x)==canonical(y) for x,y in zip(a,b)]),
            both=np.array([known(x) and known(y) for x,y in zip(a,b)]),
            A_unknown=np.array([not known(x) for x in a]),
            B_unknown=np.array([not known(x) for x in b]))
    return arrays


def weighted_metrics(v, weights):
    total=weights.sum(-1)
    both=weights@v['both']; matched=weights@(v['both'] & v['same'])
    return dict(exact_pct=100*(weights@v['same'])/total,
        both_known_pct=100*both/total,
        known_agreement_pct=np.divide(100*matched,both,out=np.full_like(both,np.nan,dtype=float),where=both>0),
        A_unknown_pct=100*(weights@v['A_unknown'])/total,
        B_unknown_pct=100*(weights@v['B_unknown'])/total)


def bootstrap_weights(records, replicates=2000):
    rng=np.random.default_rng(20261112)
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['class_name'],r['phase']].append(i)
    w=np.zeros((replicates,len(records)),dtype=np.int16)
    for ix in groups.values():
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=replicates)
    assert (w.sum(1)==len(records)).all()
    return w


def summarize_field(records, namespace, field, category='ALL'):
    n=len(records);v=field_vectors(records)
    ones=np.ones((1,n),dtype=np.int16)
    point={a:weighted_metrics(v[a],ones) for a in v}
    weights=bootstrap_weights(records)
    boot={a:weighted_metrics(v[a],weights) for a in v}
    result=dict(namespace=namespace,field=field,class_name=category,paired_targets=n)
    for arm in v:
        for key,value in point[arm].items():
            result[arm+'_'+key]=float(value[0]) if np.isfinite(value[0]) else None
        result[arm+'_both_known_n']=int(v[arm]['both'].sum())
        values_a=[r[arm]['A'] for r in records];values_b=[r[arm]['B'] for r in records]
        ca,cb=Counter(map(canonical,values_a)),Counter(map(canonical,values_b))
        chance=sum(ca[k]*cb[k] for k in ca)/n**2;obs=v[arm]['same'].mean()
        result[arm+'_cohen_kappa']=float((obs-chance)/(1-chance)) if chance<1-1e-15 else None
        if isinstance(values_a[0],list):
            result[arm+'_mean_jaccard']=float(np.mean([len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(values_a,values_b)]))
        else:result[arm+'_mean_jaccard']=None
    for key in point['old']:
        delta=point['revised'][key][0]-point['old'][key][0]
        values=boot['revised'][key]-boot['old'][key]
        valid=values[np.isfinite(values)]
        result[key+'_delta_pp']=float(delta) if np.isfinite(delta) else None
        lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
        result[key+'_delta_ci_low']=float(lo) if np.isfinite(lo) else None
        result[key+'_delta_ci_high']=float(hi) if np.isfinite(hi) else None
        result[key+'_bootstrap_valid']=len(valid)
    common=v['old']['both']&v['revised']['both']
    result['all_four_known_n']=int(common.sum())
    for arm in v:
        result[arm+'_agreement_on_all_four_known_pct']=100*float(v[arm]['same'][common].mean()) if common.any() else None
    for source in ('A','B'):
        result[source+'_old_to_revised_agreement_pct']=100*float(np.mean([
            canonical(r['old'][source])==canonical(r['revised'][source]) for r in records]))
    gate=result['known_agreement_pct_delta_pp']
    coverage=result['both_known_pct_delta_pp']
    if result['old_both_known_n']<30 or result['old_known_agreement_pct'] is None:
        decision='insufficient_known_support'
    elif result['old_known_agreement_pct']>=95:
        decision='near_ceiling_descriptive_only'
    elif gate is None or coverage < -5 or gate < 5:
        decision='screen_not_passed'
    elif result['known_agreement_pct_delta_ci_low']<=0:
        decision='point_gate_passed_uncertain_direction'
    else:decision='promising_for_expansion'
    result['engineering_screen']=decision
    return result


def consistency_flags(answer):
    f=answer['native'];flags=[]
    level=f.get('occlusion_level');occluders=set(f.get('occluder_type',[]))
    if level=='none' and occluders!={'none'}:flags.append('no_occlusion_but_occluder_not_none')
    if level not in ('none','unknown',None) and 'none' in occluders:flags.append('positive_occlusion_but_occluder_none')
    if level not in ('none','unknown',None) and occluders=={'unknown'}:flags.append('positive_occlusion_but_source_unknown')
    if 'image_boundary' in occluders:flags.append('image_boundary_used_as_occluder')
    if 'truncation_parts' in f:
        parts=set(f['truncation_parts']);level=f.get('truncation_level')
        if level=='none' and parts!={'none'}:flags.append('no_truncation_but_parts_not_none')
        if level not in ('none','unknown',None) and parts=={'none'}:flags.append('positive_truncation_but_parts_none')
    return flags


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=columns);w.writeheader();w.writerows(rows)


def analyze(split):
    run.verify(OUT/'input_freeze.json');run.verify(OUT/'execution_freeze.json')
    run.verify(OUT/'analysis_code_freeze.json')
    if split=='locked_repair_check':run.verify(OUT/'final_prompt_freeze.json')
    destination=OUT/'analysis'/split;destination.mkdir(parents=True,exist_ok=True)
    rows=[r for r in run.previous.read(OUT/'manifest.csv') if r['pilot_split']==split]
    labels={};provenance=[];completion=[];violations=[];freezes={}
    field_records=defaultdict(list);attempts=[];risk_rows=[]
    for row in rows:
        sid=row['sample_id'];labels[sid]={a:{} for a in ('old','revised')}
        for arm in ('old','revised'):
            for source in run.MODELS:
                path=run.job_path(row,source,arm)
                raw_files=list((OUT/'raw'/split/source/arm).glob(sid+'_*.json'))
                raw_files=[p for p in raw_files if not p.name.endswith('.meta.json')]
                failure=OUT/'failures'/split/source/arm/f'{sid}.json'
                error_attempts=[e['attempt'] for e in json.loads(failure.read_text())['errors']] if failure.exists() else []
                actual_attempts=max([int(p.stem.rsplit('_',1)[1]) for p in raw_files]+error_attempts+[0])
                completion.append(dict(sample_id=sid,class_name=row['class_name'],phase=row['phase'],
                    source=source,arm=arm,valid=path.exists(),actual_attempts=actual_attempts))
                if not path.exists():continue
                saved=json.loads(path.read_text());run.validate(saved['answer'],row)
                raw_path=Path(saved['raw_path']);assert run.sha(raw_path)==saved['raw_sha256']
                raw=json.loads(raw_path.read_text())
                assert raw['model']==run.MODELS[source]
                if source=='A':assert raw['done_reason']=='stop';content=raw['message']['content']
                else:assert raw['choices'][0]['finish_reason']=='stop';content=raw['choices'][0]['message']['content']
                assert run.unique_json_object(content)==saved['answer'], 'label_value_rewrite'
                prompt=(OUT/'prompts'/f"{row['class_name']}_{arm}.txt").read_text()
                prompt+='\n本次目标ID：'+sid+'\n指定类别：'+row['class_name']
                assert hashlib.sha256(prompt.encode()).hexdigest()==saved['prompt_sha256']
                assert run.sha(OUT/'evidence'/f'{sid}.jpg')==saved['evidence_sha256']
                labels[sid][arm][source]=saved['answer']
                freezes.update({str(path):run.sha(path),str(raw_path):run.sha(raw_path)})
                provenance.append(dict(sample_id=sid,source=source,arm=arm,path=str(path),
                    sha256=run.sha(path),raw_path=str(raw_path),raw_sha256=run.sha(raw_path)))
                flags=consistency_flags(saved['answer'])
                violations.append(dict(sample_id=sid,class_name=row['class_name'],source=source,arm=arm,
                    violations=len(flags),any_violation=bool(flags),flags='|'.join(flags)))
        if not all(len(labels[sid][a])==2 for a in labels[sid]):continue
        for namespace in ('native','specific'):
            fields=list(labels[sid]['old']['A'][namespace])
            for f in fields:
                record=dict(sample_id=sid,class_name=row['class_name'],phase=row['phase'],
                    old={s:labels[sid]['old'][s][namespace][f] for s in run.MODELS},
                    revised={s:labels[sid]['revised'][s][namespace][f] for s in run.MODELS})
                field_records[namespace,f].append(record)
        for arm in ('old','revised'):
            scores={s:run.previous.robustness.score_sample(float(row['bbox_area_ratio']),
                    labels[sid][arm][s]['native'],row['class_name'],'equal_parts',run.previous.robustness.CONFIG)
                    for s in run.MODELS}
            risk_rows.append(dict(sample_id=sid,class_name=row['class_name'],phase=row['phase'],arm=arm,
                A_score=scores['A']['score_midpoint'],B_score=scores['B']['score_midpoint'],
                absolute_score_difference=abs(scores['A']['score_midpoint']-scores['B']['score_midpoint'])))
    comparisons=[];byclass=[];values=[]
    for (namespace,f),records in sorted(field_records.items()):
        comparisons.append(summarize_field(records,namespace,f))
        for c in run.previous.CLASSES:
            rr=[r for r in records if r['class_name']==c]
            if rr:byclass.append(summarize_field(rr,namespace,f,c))
        for r in records:
            for arm in ('old','revised'):
                values.append(dict(sample_id=r['sample_id'],class_name=r['class_name'],phase=r['phase'],
                    namespace=namespace,field=f,arm=arm,A_value=json.dumps(r[arm]['A']),
                    B_value=json.dumps(r[arm]['B']),same=canonical(r[arm]['A'])==canonical(r[arm]['B']),
                    both_known=known(r[arm]['A']) and known(r[arm]['B'])))
    for name,records in [('completion',completion),('provenance',provenance),('field_comparison',comparisons),
                         ('field_by_class',byclass),('paired_values',values),
                         ('logical_flags',violations),('risk_scores',risk_rows)]:
        write_csv(destination/(name+'.csv'),records)
    primary=json.loads((OUT/'experiment_protocol.json').read_text())['primary_native_fields']
    summary=dict(split=split,targets=len(rows),expected_base_jobs=len(rows)*4,
        completed_base_jobs=sum(r['valid'] for r in completion),actual_attempts=sum(r['actual_attempts'] for r in completion),
        four_way_paired_targets=sum(all(len(labels[r['sample_id']][a])==2 for a in ('old','revised')) for r in rows),
        primary_fields=[r for r in comparisons if r['namespace']=='native' and r['field'] in primary],
        micro_agreement={},logical_consistency={},risk_score_MAE={},
        accuracy_established=False,fresh_generalization_test=False,predictions_read=False,
        uncertainty='2000 paired source-image bootstrap draws stratified by class and historical pool; descriptive field intervals, no multiplicity-adjusted superiority test')
    for namespace in ('native','specific'):
        rr=[r for r in comparisons if r['namespace']==namespace]
        total=sum(r['paired_targets'] for r in rr)
        summary['micro_agreement'][namespace]=dict(paired_field_instances=total)
        for arm in ('old','revised'):
            summary['micro_agreement'][namespace][arm+'_exact_pct']=sum(r['paired_targets']*r[arm+'_exact_pct'] for r in rr)/total
    for source in run.MODELS:
        for arm in ('old','revised'):
            rr=[r for r in violations if r['source']==source and r['arm']==arm]
            summary['logical_consistency'][source+'_'+arm]=dict(valid_answers=len(rr),
                any_flag=sum(r['any_violation'] for r in rr),flag_rate_pct=100*sum(r['any_violation'] for r in rr)/len(rr) if rr else None)
    for arm in ('old','revised'):
        rr=[r for r in risk_rows if r['arm']==arm]
        summary['risk_score_MAE'][arm]=float(np.mean([r['absolute_score_difference'] for r in rr])) if rr else None
    run.dump(destination/'summary.json',summary)
    run.freeze(destination/'labels_freeze.json',dict(files=freezes))
    audit(destination)
    report(destination,summary,comparisons,completion)
    run.event('analysis_complete',split=split,valid=summary['completed_base_jobs'],paired=summary['four_way_paired_targets'])
    return summary


def audit(destination):
    run.verify(OUT/'input_freeze.json');run.verify(OUT/'execution_freeze.json')
    run.verify(OUT/'analysis_code_freeze.json');run.verify(destination/'labels_freeze.json')
    comparisons=run.previous.read(destination/'field_comparison.csv')
    values=run.previous.read(destination/'paired_values.csv')
    groups=defaultdict(list)
    for r in values:groups[r['namespace'],r['field'],r['arm']].append(r)
    for r in comparisons:
        for arm in ('old','revised'):
            rr=groups[r['namespace'],r['field'],arm]
            assert len(rr)==int(r['paired_targets'])
            equal=[];known_both=[]
            for v in rr:
                a,b=json.loads(v['A_value']),json.loads(v['B_value'])
                equal.append(set(a)==set(b) if isinstance(a,list) else a==b)
                ua=bool(set(a)&UNKNOWN) if isinstance(a,list) else a in UNKNOWN
                ub=bool(set(b)&UNKNOWN) if isinstance(b,list) else b in UNKNOWN
                known_both.append(not ua and not ub)
            n=len(rr);n_known=sum(known_both)
            assert abs(100*sum(equal)/n-float(r[arm+'_exact_pct']))<1e-10
            assert abs(100*n_known/n-float(r[arm+'_both_known_pct']))<1e-10
            assert n_known==int(r[arm+'_both_known_n'])
            if n_known:
                agreement=100*sum(e and k for e,k in zip(equal,known_both))/n_known
                assert abs(agreement-float(r[arm+'_known_agreement_pct']))<1e-10
    completion=run.previous.read(destination/'completion.csv')
    summary=json.loads((destination/'summary.json').read_text())
    assert sum(r['valid']=='True' for r in completion)==summary['completed_base_jobs']
    assert max(int(r['actual_attempts']) for r in completion)<=3
    for ns in ('native','specific'):
        for arm in ('old','revised'):
            rr=[r for r in values if r['namespace']==ns and r['arm']==arm]
            pct=100*sum(r['same']=='True' for r in rr)/len(rr)
            assert abs(pct-summary['micro_agreement'][ns][arm+'_exact_pct'])<1e-10
    run.dump(destination/'validation.json',dict(status='pass',field_comparisons_recomputed=len(comparisons)*2,
        saved_answers_match_raw_responses=True,source_freezes_verified=True,attempt_budget_verified=True,
        semantic_accuracy_established=False,predictions_used=False))


def fmt(x):
    return '不可评价' if x is None else f'{x:.2f}'


def report(destination,summary,comparisons,completion):
    split=summary['split'];lines=[f'# 协议修订小试验：{split}', '',
        '比较同一目标、同一证据图上的Qwen与SenseNova独立答案。旧／新协议均重新真实调用，不复用历史答案。'
        '新版保留字段与枚举，只澄清语义；模型答案不在本地改值。', '',
        f"固定目标{summary['targets']}个，基本任务{summary['completed_base_jobs']}/{summary['expected_base_jobs']}完成，"
        f"实际请求{summary['actual_attempts']}次，四路配对齐全{summary['four_way_paired_targets']}个目标。", '',
        '| 字段命名空间 | 配对字段数 | 旧协议一致率 | 新协议一致率 |', '|---|---:|---:|---:|']
    for ns,r in summary['micro_agreement'].items():
        lines.append(f"| {ns} | {r['paired_field_instances']} | {r['old_exact_pct']:.2f}% | {r['revised_exact_pct']:.2f}% |")
    lines+=['', '## 优先字段：已知一致性与有效支持同时检查', '',
        '| 字段 | 旧／新双方已知数 | 旧已知一致率 | 新已知一致率 | 变化及95%配对bootstrap区间（百分点） | 双方已知比例变化 | 工程筛查 |',
        '|---|---:|---:|---:|---|---:|---|']
    for r in summary['primary_fields']:
        lines.append(f"| {r['field']} | {r['old_both_known_n']}／{r['revised_both_known_n']} | "
            f"{fmt(r['old_known_agreement_pct'])}% | {fmt(r['revised_known_agreement_pct'])}% | "
            f"{fmt(r['known_agreement_pct_delta_pp'])} [{fmt(r['known_agreement_pct_delta_ci_low'])}, "
            f"{fmt(r['known_agreement_pct_delta_ci_high'])}] | {fmt(r['both_known_pct_delta_pp'])} | {r['engineering_screen']} |")
    lines+=['', '筛查规则在调用前冻结：已知一致率增幅至少5个百分点，双方已知比例下降不超过5个百分点；'
        '旧双方已知少于30、或旧已知一致率至少95%时只做描述。promising还要求描述性区间下限大于0。'
        '这些是工程扩展筛查，不是语义准确率或经多重比较校正的优势证明。', '',
        '## unknown及子集变化', '',
        '完整表同时给出双方各自unknown率、包含unknown的一致率、四路均已知的同一子集一致率、'
        '集合Jaccard、κ及逐类结果。不能仅展示双方已知子集或跨来源一致子集。', '',
        '## 风险分数与逻辑标志', '',
        f"固定旧评分规则下，两模型风险分数平均绝对差：旧{fmt(summary['risk_score_MAE']['old'])}，"
        f"新{fmt(summary['risk_score_MAE']['revised'])}（0–100）。未在这个小样本上重选方法或调风险权重。", '',
        '| 模型／协议 | 有效答案 | 至少一个逻辑标志 |', '|---|---:|---:|']
    for k,r in summary['logical_consistency'].items():lines.append(f"| {k} | {r['valid_answers']} | {r['any_flag']} |")
    lines+=['', '逻辑标志只检查字段内部兼容性与原图边界被当作遮挡物等情况，不裁定图像语义；'
        '解释中提到模糊／烟雾不会自动拒绝答案。', '',
        '## 执行与结论边界', '',
        '- 同一标记原图＋目标裁剪拼图用于全部四路调用；历史原生提示原本使用三张图，因此旧协议新跑并非历史缓存的完全复现。',
        '- 两模型接口的结构化约束能力不同：Qwen使用schema format，SenseNova使用JSON模式并将同一schema写入提示；失败率按模型／协议保留。',
        '- 字段含义仍有主观性。两模型一致、解释合规或工程筛查通过都不能证明准确率。',
        '- 固定旧Q分层抽取，源图全局去重；该240图及其历史输出已使用过，锁定复核不是未见数据泛化验证。',
        '- 本轮最多一次语义修订，实际保留候选v1，不根据复核结果继续调规则；不会自动全量扩展。',
        '- 这是标签协议小试验，未重跑P20／Greedy20，不据此宣布方法优势恢复。', '',
        '明细：field_comparison.csv、field_by_class.csv、paired_values.csv、completion.csv、logical_flags.csv、risk_scores.csv。',
        '原始响应与保存答案已逐条核对；输入及答案哈希均冻结。', '']
    (destination/'report_zh.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['development','locked_repair_check'],required=True)
    args=p.parse_args();analyze(args.split)
