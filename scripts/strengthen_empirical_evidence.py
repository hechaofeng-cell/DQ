#!/usr/bin/env python3
"""Audit measured costs, select reproducible diagnostic cases, package all outcomes."""
from __future__ import annotations
import argparse
import csv
import html
import itertools
import json
import re
import shutil
import time
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from voc_external_common import ROOT,BASE,COCO_B,EVIDENCE,PROTOCOL,common

def cached_policy_costs(studies):
    """Representative CPU timing; uses existing labels/embeddings/predictions only."""
    from voc_external_common import activate
    a,selection,analysis=activate();output=[]
    for study in studies:
        n=len(common.read_csv(study/'manifests/candidate_pool.csv'))//6
        step=20 if n==400 else 10;budgets=tuple(step*t for t in range(1,6))
        for module in (a,selection,analysis):module.__dict__.update(N=n,TIER_N=n//5,STEP=step,TOTAL=n*6,BUDGETS=budgets)
        common.BASE=study;selection.BASE=study;analysis.BASE=study
        _,contexts,_=common.context()
        with np.load(study/'embeddings/vit_b16_candidates.npz') as z:emb=z['embeddings']
        model='resnet18';seed=common.SEEDS[0]
        features,logits,pred=analysis.prediction(model,seed)
        with np.load(study/'benchmark/runs'/model/str(seed)/'train_features.npz') as z:train=z['features'];train_labels=z['labels']
        score_times={};scores={}
        begin=time.perf_counter();prob=a.b.softmax(logits);scores['deepgini']=1-(prob**2).sum(1);score_times['deepgini']=time.perf_counter()-begin
        begin=time.perf_counter();scores['dsa']=a.b.dsa_scores(train,train_labels,features,pred);score_times['dsa']=time.perf_counter()-begin
        for source,ctx in contexts.items():
            for method in ('coverage','coverage_without_H','deepgini','dsa','kcenter','random'):
                for repeat in range(2):
                    begin=time.perf_counter()
                    for c in common.CLASSES:
                        section=ctx['by_class'][c];tiers,cells,quotas=selection.risk_structure(source,c,contexts)
                        if method=='random':
                            rng=np.random.default_rng(20261005+repeat)
                            for _ in range(1000):
                                for t in range(5):
                                    for k in sorted(cells[t]):
                                        if quotas[t][k]:rng.choice(cells[t][k],quotas[t][k],replace=False)
                            continue
                        kwargs={}
                        if method in ('coverage','coverage_without_H'):
                            matrix=section['matrix']
                            if method=='coverage_without_H':matrix=matrix[:,np.asarray([f not in selection.EXCLUDE for f,v in section['states']])]
                            kwargs['matrix']=matrix
                        elif method=='kcenter':kwargs['embeddings']=emb[section['global_indices']]
                        else:kwargs['scores']=scores[method][section['global_indices']]
                        selection.constrained(section,tiers,cells,quotas,**kwargs)
                    elapsed=time.perf_counter()-begin
                    output.append(dict(study=study.name,source=source,method=method,repeat=repeat+1,model=model,seed=seed,
                        candidates=n*6,endpoint_budget_per_class=budgets[-1],cpu_selection_seconds=elapsed,
                        cached_score_seconds=score_times.get(method,0),random_replicates=1000 if method=='random' else None,
                        excludes='VLM annotation, image embedding extraction, classifier prediction, data loading before timer',
                        scope='representative resnet18/first-seed cached CPU policy benchmark; not all-model mean or end-to-end method ranking'))
    return output

def cost(study):
    records=[];summaries=[]
    for backend in ('local','online'):
        successful=[];raw_count=0;prompt=0;completion=0;missing_usage=0;service_seconds=0
        for domain in ('old','car'):
            folder=study/'annotations'/backend/domain
            for path in (folder/'parsed').glob('*.json'):
                value=json.loads(path.read_text())
                field='elapsed_seconds' if backend=='local' else 'online_elapsed_seconds'
                if value.get(field) is not None:successful.append(float(value[field]))
            for path in (folder/'raw').glob('*.json'):
                raw_count+=1
                try:value=json.loads(path.read_text())
                except (OSError,ValueError):value={}
                usage=value.get('usage',{}) if backend=='online' else dict(prompt_tokens=value.get('prompt_eval_count'),completion_tokens=value.get('eval_count'))
                valid=usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None
                if valid:prompt+=int(usage['prompt_tokens']);completion+=int(usage['completion_tokens'])
                else:missing_usage+=1
                duration=float(value['total_duration'])/1e9 if backend=='local' and value.get('total_duration') is not None else None
                if duration is not None:service_seconds+=duration
                records.append(dict(study=study.name,backend=backend,raw_path=str(path),prompt_tokens=usage.get('prompt_tokens'),
                  completion_tokens=usage.get('completion_tokens'),usage_available=valid,local_server_duration_seconds=duration,
                  response_has_success_payload='choices' in value or 'message' in value))
        ledger=study/'cost/request_attempts.jsonl';attempts=[]
        if ledger.exists():attempts=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        attempts=[r for r in attempts if r['backend']==backend]
        def span(entries):
            if not entries:return None
            starts=[datetime.fromisoformat(r['started_at']) for r in entries];ends=[datetime.fromisoformat(r['finished_at']) for r in entries]
            return (max(ends)-min(starts)).total_seconds()
        log=study/'pipeline.log';stage_times=[]
        if log.exists():
            for line in log.read_text().splitlines():
                if any(marker in line for marker in ('smoke_'+backend,'"round":',' freeze_selection ')):
                    matched=re.match(r'\[([^]]+)\]',line)
                    if matched:stage_times.append(datetime.fromisoformat(matched.group(1)))
        estimates=(max(stage_times)-min(stage_times)).total_seconds() if len(stage_times)>1 else None
        count=sum(1 for domain in ('old','car') for _ in (study/'annotations'/backend/domain/'parsed').glob('*.json'))
        summaries.append(dict(study=study.name,backend=backend,successful_targets=count,raw_response_files=raw_count,
          extra_responses_over_targets=max(0,raw_count-count),raw_prompt_tokens=prompt,raw_completion_tokens=completion,
          raw_responses_missing_usage=missing_usage,successful_latency_n=len(successful),
          successful_latency_median_seconds=float(np.median(successful)) if successful else None,
          successful_latency_p95_seconds=float(np.percentile(successful,95)) if successful else None,
          successful_latency_sum_seconds=float(sum(successful)),local_raw_server_seconds=service_seconds if backend=='local' else None,
          attempt_ledger_count=len(attempts),request_span_seconds=span(attempts),
          ledger_elapsed_sum_seconds=sum(r['elapsed_seconds'] for r in attempts) if attempts else None,
          pipeline_annotation_window_upper_bound_seconds=estimates,
          monetary_cost=None,billing_status='unknown: no historical rate or actual invoice supplied',
          historical_attempt_latency_complete=bool(attempts),
          note='Raw server time and successful request duration are not parallel wall time or measured GPU active time. Raw token sums include retries when usage exists.'))
    return records,summaries

def cases(study):
    labels={};rows=common.read_csv(study/'manifests/candidate_pool.csv');byid={r['sample_id']:r for r in rows}
    for backend in ('local','online'):
        labels[backend]={}
        for domain in ('old','car'):
            for path in (study/'annotations'/backend/domain/'parsed').glob('*.json'):
                d=json.loads(path.read_text());labels[backend][d['image_id']]=d['features']
        from run_multiclass_object_selection_benchmark import scale_state
        for row in rows:labels[backend][row['sample_id']]['target_scale']=scale_state(float(row['bbox_area_ratio']))
    frame=pd.read_csv(study/'transfer/tables/state_validation.csv')
    keys=['source','model','class_name','feature','value']
    agg=frame.groupby(keys,as_index=False).agg(support=('B_support','first'),evaluable=('B_evaluable','first'),
         class_recall=('B_class_recall','mean'),state_recall=('B_recall','mean'),gap=('B_gap','mean'),
         gap_min=('B_gap','min'),gap_max=('B_gap','max'),risk_matched_gap=('risk_matched_gap','mean'),risk_matchable_n=('risk_matchable_n','first'))
    agg['study']=study.name
    candidate=agg[agg.evaluable].copy();chosen=[]
    for c in common.CLASSES:
        for mode,condition,ascending in [('replicated_weakness',candidate.gap<=-.10,True),('not_replicated_direction',candidate.gap>=0,False)]:
            part=candidate[(candidate.class_name==c)&condition]
            if len(part):
                selected=part.sort_values(['gap','model','source','feature','value'],ascending=[ascending,True,True,True,True]).iloc[0].to_dict()
                selected['case_type']=mode;chosen.append(selected)
    predictions={}
    def getpred(model):
        if model not in predictions:
            values=[]
            for seed in common.SEEDS:
                with np.load(study/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as z:
                    assert z['sample_ids'].tolist()==[r['sample_id'] for r in rows]
                    values.append(z['logits'].argmax(1))
            predictions[model]=np.asarray(values)
        return predictions[model]
    globals_={r['sample_id']:i for i,r in enumerate(rows)};images=[]
    def member(value,expected):return expected in value if isinstance(value,list) else value==expected
    for item in chosen:
        candidates=sorted([r for r in rows if r['class_name']==item['class_name'] and member(labels[item['source']][r['sample_id']][item['feature']],item['value'])],key=lambda r:r['sample_id'])
        p=getpred(item['model'])
        for correct,kind in [(False,'state_failure'),(True,'state_success_counterexample')]:
            available=[r for r in candidates if bool(np.all(p[:,globals_[r['sample_id']]]==common.LABELS[r['class_name']]))==correct
                       and (correct or np.all(p[:,globals_[r['sample_id']]]!=common.LABELS[r['class_name']]))]
            if available:
                r=available[0];images.append(dict(**item,sample_id=r['sample_id'],image_kind=kind,**{k:r[k] for k in ('image_path','crop_path','marked_image_path')},
                    predicted_classes=json.dumps([common.CLASSES[int(v)] for v in p[:,globals_[r['sample_id']]]])))
    risk=pd.read_csv(study/'selection/risk_scores.csv');model='convnext_tiny';p=getpred(model)
    for c in common.CLASSES:
        high=sorted(risk[(risk.source=='local')&(risk.class_name==c)&(risk.tier==5)].sample_id)
        good=[k for k in high if np.all(p[:,globals_[k]]==common.LABELS[c])]
        if good:
            r=byid[good[0]];images.append(dict(study=study.name,case_type='high_risk_success',source='local',model=model,class_name=c,
               sample_id=r['sample_id'],image_kind='high_risk_success',image_path=r['image_path'],crop_path=r['crop_path'],marked_image_path=r['marked_image_path']))
        disagreement=sorted(r['sample_id'] for r in rows if r['class_name']==c and any(labels['local'][r['sample_id']][f]!=labels['online'][r['sample_id']][f]
           for f in ('body_visibility','occlusion_level','truncation_level','body_orientation')))
        if disagreement:
            k=disagreement[0];r=byid[k];differences={f:{b:labels[b][k][f] for b in ('local','online')} for f in labels['local'][k]
                  if f in labels['online'][k] and labels['local'][k][f]!=labels['online'][k][f]}
            images.append(dict(study=study.name,case_type='VLM_disagreement',class_name=c,sample_id=k,image_kind='annotation_disagreement',
                image_path=r['image_path'],crop_path=r['crop_path'],marked_image_path=r['marked_image_path'],differences=json.dumps(differences,ensure_ascii=False)))
    return agg.to_dict('records'),chosen,images

def tables(studies):
    records=[]
    for study in studies:
        for family,path in [('main','comparison/tables/comparisons.csv'),('uniform_vit','robustness/tables/vit_cluster_comparisons.csv')]:
            for r in common.read_csv(study/path):
                field='holm_primary4_p' if r['baseline']=='random' else 'holm_ablation4_p' if r['baseline']=='coverage_without_H' else 'holm_strong12_p'
                effect=float(r['mean_delta_auc']);p=float(r[field])
                level='primary' if family=='main' and r['baseline']=='random' else 'sensitivity' if family=='uniform_vit' else 'secondary'
                records.append(dict(study=study.name,result_level=level,representation=r['representation'],source=r['source'],baseline=r['baseline'],metric=r['metric'],
                    delta_auc=effect,ci95_low=float(r['ci95_low']),ci95_high=float(r['ci95_high']),family_holm_p=p,
                    all20_holm_p=float(r['holm_all20_p']),positive_runs=int(r['positive_runs']),negative_runs=int(r['negative_runs']),
                    interpretation='positive_and_family_supported' if effect>0 and p<.05 else 'negative_and_family_supported' if effect<0 and p<.05 else 'not_supported_in_prespecified_family'))
    return records

def paper(studies,summary,method_records):
    external=BASE in studies
    lines=['# 可解释视觉状态与风险分层的图像分类测试诊断：六类实证研究','',
      '本稿更新主次结果与适用范围。旧20261004论文保留；新增来源只有执行完成后才报告。研究不要求覆盖—F1曲线单调，不提出训练增强或全面优势。','',
      '## 摘要','',
      '本研究在固定六类分类任务、1800训练目标、八架构及三个训练种子下，用线上/本地VLM描述可见状态并构造类别内视觉风险层。COCO新池的24次运行均出现高风险层F1低于低风险层；在共同风险配额和逐类别等错误数控制下，Coverage相对Random提供混淆类型诊断收益。统一表征错误簇和强基线比较呈现条件性结果，不能概括为普遍优势。自动标签不是人工真值，风险评分不代表所有困难因素。',
      '', '## 问题与贡献','',
      'RQ1：预定义风险层能否区分固定模型的测得性能？RQ2：旧池冻结薄弱状态是否在新池/新来源复现？RQ3：控制预定义风险、预算及错误数后，Coverage是否增加独立混淆类型/错误簇广度？',
      '贡献是可复算的可解释状态测试组织协议、同风险等错误数诊断对照，以及跨池复现和条件性负结果的证据。没有文献核验支持“首创”；不把显著性当新颖性。','',
      '## 数据、协议与统计','',
      '原COCO A为发现池、COCO B为2400独立样本确认池，各类400；B的弱状态清单在预测前由A冻结，外部来源沿用清单不重选。分类权重、预处理、五维H、unknown规则和支持阈值固定。状态标注三视图且类别可见，分类器只用固定目标框裁剪。',
      'VOC2012因完整历史候选排除后dog仅47张停止，未开展新标注/分类预测。改用VOC2007 trainval+test合并，六类100/class，预算10–50/class，各层20候选选10。COCO B选样比例25%、VOC50%；各来源内部方法预算相同，不能把不同池原始覆盖分母直接合并。',
      '全部24运行来自8架构×3种子。主要风险6项Holm、主要随机诊断4项Holm、冻结状态迁移2项Holm；强基线12项、删字段4项及统一ViT和全20敏感性不省略。12/20族存在精确架构检验分辨率限制，未通过不能当零效应；固定面板图像bootstrap不能代表新增架构或重新选样。','',
      '## 主要风险确认','', '| 数据池 | 来源 | Q5−Q1 Macro-F1 (pp) | 95%区间(pp) | 六项Holm p |','|---|---|---:|---|---:|']
    for study in studies:
        for r in common.read_csv(study/'confirmation/tables/difficulty_contrasts.csv'):
            if r['unknown_mode']!='midpoint':continue
            lines.append(f"| {study.name} | {r['source']} | {float(r['mean_delta_auc'])*100:+.2f} | [{float(r['ci95_low'])*100:+.2f},{float(r['ci95_high'])*100:+.2f}] | {float(r['holm_six_p']):.5f} |")
    lines+=['','## 外部来源确认状态','', 'VOC2007已完成；以下与COCO B分开报告。' if external else 'VOC2007执行中；当前没有外部来源结果，不写复现成功。','',
       '## 全部方法结果与负结果','', '| 池 | 表征 | 来源 | 对照 | 指标 | AUC差 | 家族Holm p | 全20 p |','|---|---|---|---|---|---:|---:|---:|']
    for r in method_records:
        lines.append(f"| {r['study']} | {r['representation']} | {r['source']} | {r['baseline']} | {r['metric']} | {r['delta_auc']:+.4f} | {r['family_holm_p']:.5f} | {r['all20_holm_p']:.5f} |")
    lines+=['','COCO B分类器表征下Random主四项支持，统一ViT错误簇收益未通过主家族校正。强基线与删字段优势没有稳定确认；不能只展示正向平均值。全部区间和逐预算/运行结果见full_outcome_index.csv及原始表。','',
       '## 冻结状态迁移','', '| 池 | 来源 | 状态减类别Recall(pp) | 两项Holm p |','|---|---|---:|---:|']
    for study in studies:
        for r in common.read_csv(study/'transfer/tables/comparisons.csv'):
            lines.append(f"| {study.name} | {r['source']} | {float(r['mean_delta_auc'])*100:+.2f} | {float(r['holm_two_p']):.5f} |")
    lines+=['','## 成本与实际案例','',
       '成本明细见cost_summary.csv、cost_raw_response_ledger.csv与stage_durations.csv；未提供实际账单或历史费率，金额未知。已知token包括有usage的失败/重试响应；缺失usage不能当0。局部服务器时长、成功请求时长、并行墙钟与GPU活跃时间不能混用。cached_policy_costs.csv为ResNet18首种子的两次代表性CPU缓存选样实测，不是全部模型均值；Random1000次与单次固定策略分别标明，不直接作速度排名。',
       'diagnostic_cases.html同时展示原A冻结状态在新池的复现/未复现、同状态正确样本、高风险成功和VLM分歧。全状态支持、三种子差异和风险匹配差见state_case_metrics.csv。案例经结果筛选用于解释，不是独立假设检验或失败根因证明。','',
       '## 有效性边界与结论','',
       '相对于当前五维风险评分的额外信息不等于控制全部难度；自动状态的可解释性不等于语义准确率。有限类别、8架构、数据来源与单目标分类限制泛化；旧VOC发现过程、预训练数据可能重叠均说明。',
       '研究支持在所测固定模型与候选池中用视觉风险组织能力测试，并用状态诊断定位可复现的薄弱条件。Coverage相对风险匹配Random的混淆类型广度提供条件性证据；对强基线、统一表征错误簇的普遍优势仍未确立。',
       '', '## 与旧完整论文的关系','',
       '本稿为增补后的证据与口径稿，旧完整论文保留文献和详细方法：'+str(ROOT/'docs/visual_state_coverage_difficulty_stratified_paper_20261004.md'),
       '未增加未经核验的新文献、作者、资助或发表承诺。投稿前仍需把两稿合并、验证引用、补作者信息及目标期刊模板。']
    target=ROOT/'docs/visual_state_coverage_external_evidence_revision_20261005.md';target.write_text('\n'.join(lines)+'\n')
    from export_empirical_paper_docx import export
    export(target,target.with_suffix('.docx'))
    return target

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-external',action='store_true');args=parser.parse_args()
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    studies=[COCO_B]
    if args.include_external:
        assert json.loads((BASE/'final_audit.json').read_text())['status']=='pass';studies.append(BASE)
    raw=[];summaries=[];metrics=[];chosen=[];images=[]
    for study in studies:
        rr,ss=cost(study);raw+=rr;summaries+=ss
        mm,cc,ii=cases(study);metrics+=mm;chosen+=cc;images+=ii
    for name,rows in [('cost_raw_response_ledger',raw),('cost_summary',summaries),('state_case_metrics',metrics),('selected_state_cases',chosen),('case_images',images)]:common.write_csv(EVIDENCE/f'{name}.csv',rows)
    outcomes=tables(studies);common.write_csv(EVIDENCE/'full_outcome_index.csv',outcomes)
    stage_rows=[]
    for study in studies:
        for marker in (study/'stages').glob('*.json'):
            process=study/'processes'/f'{marker.stem}.json'
            if not process.exists():continue
            first=json.loads(process.read_text());last=json.loads(marker.read_text())
            if first.get('started_at') and last.get('finished_at'):
                duration=(datetime.fromisoformat(last['finished_at'])-datetime.fromisoformat(first['started_at'])).total_seconds()
                stage_rows.append(dict(study=study.name,stage=marker.stem,wall_seconds=duration,
                    note='Coordinator wall time includes up to 20s polling; stage may include cache reuse or retries.'))
    common.write_csv(EVIDENCE/'stage_durations.csv',stage_rows)
    common.write_csv(EVIDENCE/'cached_policy_costs.csv',cached_policy_costs(studies))
    page=['<!doctype html><meta charset="utf-8"><title>诊断案例与反例</title>',
       '<style>body{font-family:sans-serif;max-width:1100px;margin:32px auto}article{border-top:1px solid #bbb;padding:18px 0}img{max-width:48%;max-height:280px}pre{white-space:pre-wrap}</style>',
       '<h1>诊断案例、反例与标注分歧</h1><p>确定性ID选例；状态案例为探索性展示，关联不等于错误因果解释。支持不足和未复现条件保存在全状态表。</p>']
    for item in images:
        page.append('<article><h2>'+html.escape(f"{item['study']} / {item['class_name']} / {item['case_type']} / {item['sample_id']}")+'</h2>')
        page.append('<img src="'+html.escape(Path(item['marked_image_path']).as_uri(),quote=True)+'"><img src="'+html.escape(Path(item['crop_path']).as_uri(),quote=True)+'">')
        page.append('<pre>'+html.escape(json.dumps({k:v for k,v in item.items() if not k.endswith('_path')},ensure_ascii=False,indent=2))+'</pre></article>')
    (EVIDENCE/'diagnostic_cases.html').write_text('\n'.join(page))
    target=paper(studies,summaries,outcomes)
    (EVIDENCE/'paper_brief.md').write_text('# 范围\n\n固定模型的状态测试诊断实证；COCO B及完成后的VOC2007。旧完整稿保留，新证据稿更新MD/Word。\n')
    (EVIDENCE/'claim_evidence.md').write_text('# 结论—证据边界\n\n风险差异→各池confirmation/tables/difficulty_contrasts.csv；额外诊断→full_outcome_index.csv完整家族结果；状态复现→state_case_metrics.csv；成本→raw账本与缺失项；案例是探索性关联。外部结果仅include-external完成后报告。全面优势、因果、人工标签准确与首次声明禁止。\n')
    with (EVIDENCE/'research_log.md').open('a') as stream:stream.write(f'\n[{common.now()}] 生成成本/案例/全结果索引与新稿，外部完成={args.include_external}；原始结果不修改。\n')
    (EVIDENCE/'review.md').write_text('# 证据审阅\n\n结果可追踪：原始表和全raw账本，金额及历史缺失延迟不虚构。范围：明确关联、条件性收益和所有负结果；外部执行状态如实。引用：本增补稿未引入新外部事实引用，旧完整稿引用仍需投稿合稿审计；不声称整篇引用门禁已完成。\n')
    if args.include_external:
        primary=[r for r in outcomes if r['study']==BASE.name and r['result_level']=='primary']
        supported=sum(r['interpretation']=='positive_and_family_supported' for r in primary)
        (BASE/'claim_evidence.md').write_text('# 外部确认结论—证据\n\n'
          f'Coverage对风险配额Random：主四项正向且家族通过{supported}/4；全20、强基线、删字段与统一ViT完整见../empirical_evidence_strengthening_20261005/full_outcome_index.csv。\n\n'
          '风险差异→confirmation/tables/difficulty_contrasts.csv；原A状态迁移→transfer/tables/state_validation.csv，保留不可评价状态。\n\n'
          '成本和案例→增补目录；金额未知、案例为探索性关联。仅支持所测模型与来源的条件性结论，不声明全面优势、因果、人工准确或首创。\n')
        (BASE/'review.md').write_text('# 外部执行与证据审阅\n\n'
          '结果可追踪：输入/权重冻结、完整性audit及原始表；执行通过不等于假设支持。\n\n'
          '范围：全部正负结果报告，VOC100/class预算10–50与COCO比例差明确，自动标签非人工真值。\n\n'
          '引用：VOC来源与校验值有官方/PyTorch原始依据；旧完整稿引用仍待合稿审计，不宣称投稿前全部门禁通过。\n')
        with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{common.now()}] 外部结果与成本、案例、主次/负结果、MD/Word增补已生成；未按结果调整规则。\n')
    common.write_json(EVIDENCE/'status.json',dict(status='complete',external_included=args.include_external,studies=[p.name for p in studies],
        cost_response_rows=len(raw),state_rows=len(metrics),state_cases=len(chosen),image_examples=len(images),comparison_rows=len(outcomes),paper=str(target),monetary_cost_known=False))
    print(json.dumps(json.loads((EVIDENCE/'status.json').read_text()),ensure_ascii=False))

if __name__=='__main__':main()
