#!/usr/bin/env python3
"""Confirm frozen risk rules, compare quota-matched methods, and transfer faults."""
from __future__ import annotations
import argparse
import itertools
import json
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,spearmanr
from sklearn.metrics import precision_recall_fscore_support
from independent_confirmation_common import *
import analyze_dual_vlm_study as a
import analyze_sixclass_difficulty as supplement
from freeze_independent_sixclass_selection import load,constrained,risk_structure

def prediction(model,seed):
    path=BASE/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz'
    with np.load(path) as saved:
        rows=read_csv(BASE/'manifests/candidate_pool.csv')
        assert saved['sample_ids'].tolist()==[r['sample_id'] for r in rows] and saved['logits'].shape==(2400,6)
        return saved['features'],saved['logits'],saved['logits'].argmax(1)

def inference():
    import torch
    verify(BASE/'pre_prediction_freeze.sha256.json')
    _,contexts,_=context();ctx=contexts['local'];b=a.b
    b.CHECKPOINT_ROOT=WEIGHTS
    b.OLD_CACHE=BASE/'disabled_cache'
    torch.set_num_threads(8)
    all_paths=[]
    for index,(model,seed) in enumerate(itertools.product(MODELS,SEEDS),1):
        note('infer',run=index,total=24,model=model,seed=seed)
        folder=BASE/'benchmark/runs'/model/str(seed);folder.mkdir(parents=True,exist_ok=True)
        old=ROOT/'artifacts/sixclass_car_extension_20261004/benchmark/runs'/model/str(seed)/'train_features.npz'
        if not (folder/'train_features.npz').exists():shutil.copy2(old,folder/'train_features.npz')
        values=b.infer_and_extract(BASE/'benchmark',ctx,model,seed,4)
        assert values['candidate_logits'].shape==(2400,6)
        all_paths.extend([folder/'candidate_outputs.npz',folder/'train_features.npz'])
    freeze(BASE/'prediction_freeze.sha256.json',all_paths)
    write_json(BASE/'inference_audit.json',dict(status='pass',runs=24,classes=6,targets=2400,weights_retrained=False,
               prediction_inputs='same 5% margin crop and Resize256/CenterCrop224/ImageNet normalization as pool A'))

def confirm():
    verify(BASE/'pre_prediction_freeze.sha256.json');verify(BASE/'prediction_freeze.sha256.json')
    # Adapter changes destinations only; original supplement and original pool stay intact.
    supplement.SOURCE=BASE;supplement.OUTPUT=BASE/'confirmation'
    supplement.configure=lambda:context()[:2]
    shutil.copy2(BASE/'pre_prediction_freeze.sha256.json',BASE/'amendment_input_freeze.sha256.json')
    supplement.run()
    summary_path=BASE/'confirmation/summary.json'
    summary=json.loads(summary_path.read_text())
    summary.update(analysis_type='prospective_independent_sample_confirmation',
                   registration=str(BASE/'registration.json'),
                   independence='new samples; same COCO dataset source',
                   quota_selection_comparison='separate prespecified comparison/ stage')
    write_json(summary_path,summary)
    source_bootstrap()

def source_bootstrap():
    _,contexts,_=context();output=[]
    risk_scores=pd.read_csv(BASE/'confirmation/tables/difficulty_scores.csv')
    rng=np.random.default_rng(20261005)
    for source,ctx in contexts.items():
        indices={name:{tier:[] for tier in (1,5)} for name in CLASSES}
        lookup={r['sample_id']:i for i,r in enumerate(ctx['rows'])}
        for name in CLASSES:
            for tier in (1,5):indices[name][tier]=[lookup[k] for k in risk_scores[(risk_scores.source==source)&(risk_scores.class_name==name)&(risk_scores.tier==tier)].sample_id]
        draws={tier:np.concatenate([rng.choice(indices[name][tier],size=(2000,80),replace=True) for name in CLASSES],axis=1) for tier in (1,5)}
        vectors=[]
        for model,seed in itertools.product(MODELS,SEEDS):
            _,_,pred=prediction(model,seed)
            def vector_f1(selected):
                true=np.repeat(np.arange(6),80);encoded=np.arange(2000)[:,None]*36+true[None,:]*6+pred[selected]
                confusion=np.bincount(encoded.ravel(),minlength=2000*36).reshape(2000,6,6)
                tp=np.diagonal(confusion,axis1=1,axis2=2)
                denom=confusion.sum(1)+confusion.sum(2)
                return np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean(1)
            vectors.append(vector_f1(draws[5])-vector_f1(draws[1]))
        means=np.asarray(vectors).mean(0)
        output.append(dict(source=source,metric='Q5_minus_Q1_macro_f1',bootstrap_unit='source targets within class/tier; fixed all 24 runs',
                           ci95_low=float(np.percentile(means,2.5)),ci95_high=float(np.percentile(means,97.5)),replicates=2000))
    write_csv(BASE/'confirmation/tables/source_image_bootstrap.csv',output)

def model_orders(source,contexts,fixed,model,seed,logits,features):
    orders={m:fixed[source][m] for m in ('coverage','coverage_without_H','kcenter','cross_coverage')}
    folder=BASE/'benchmark/runs'/model/str(seed)
    with np.load(folder/'train_features.npz') as saved:
        surprise=a.b.dsa_scores(saved['features'],saved['labels'],features,logits.argmax(1))
    probabilities=a.b.softmax(logits);gini=1-(probabilities**2).sum(1)
    for method,score in [('deepgini',gini),('dsa',surprise)]:
        orders[method]={}
        for name in CLASSES:
            section=contexts[source]['by_class'][name];tiers,cells,quotas=risk_structure(source,name,contexts)
            orders[method][name]=constrained(section,tiers,cells,quotas,scores=score[section['global_indices']])
    return orders

def audit_order(order,source,contexts):
    for name in CLASSES:
        tiers,cells,quotas=risk_structure(source,name,contexts)
        assert len(order[name])==100 and len(set(order[name]))==100
        for t in range(5):
            chosen=order[name][t*20:(t+1)*20]
            assert set(chosen)<=set(tiers[t])
            assert all(sum(i in chosen for i in items)==quotas[t][k] for k,items in cells[t].items())

def compare():
    verify(BASE/'pre_prediction_freeze.sha256.json');verify(BASE/'prediction_freeze.sha256.json')
    _,contexts,fixed,random=load();fixed_rows=[];random_rows=[];controls=[];order_rows=[];groups=defaultdict(lambda:defaultdict(list))
    sensitivity=defaultdict(lambda:defaultdict(list));cross_groups=defaultdict(lambda:defaultdict(list))
    with np.load(BASE/'embeddings/vit_b16_candidates.npz') as saved:vit=saved['embeddings']
    for index,(model,seed) in enumerate(itertools.product(MODELS,SEEDS),1):
        note('compare',run=index,total=24,model=model,seed=seed)
        features,logits,pred=prediction(model,seed)
        structures=a.support.error_structures(contexts['local'],features,pred,model,seed)
        vit_structures=a.support.error_structures(contexts['local'],vit,pred,model,seed)
        for source,ctx in contexts.items():
            orders=model_orders(source,contexts,fixed,model,seed,logits,features)
            for method,order in orders.items():
                audit_order(order,source,contexts)
                if method in ('deepgini','dsa'):
                    order_rows.extend(dict(source=source,model=model,seed=seed,method=method,class_name=c,rank=i+1,sample_id=ctx['by_class'][c]['rows'][j]['sample_id']) for c in CLASSES for i,j in enumerate(order[c]))
            curves=defaultdict(list);vit_curves=defaultdict(list);cross_curve=[]
            for budget in (20,40,60,80,100):
                for method,order in orders.items():
                    fixed_rows.append(dict(source=source,model=model,seed=seed,method=method,budget=budget,**a.measured_one(ctx,order,budget,pred)))
                selection=a.prepared_selection(ctx,random[source],budget)
                values=a.measures(ctx,selection,pred,a.state_stats(ctx,selection,pred))
                random_rows.append(dict(source=source,model=model,seed=seed,budget=budget,**a.random_summary(values)))
                for baseline in ('random','deepgini','dsa','kcenter','coverage_without_H'):
                    for representation,structure,destination in [('classifier',structures,curves),('vit',vit_structures,vit_curves)]:
                        if baseline=='random':
                            pairs=[a.control.class_stratified_pair(ctx,orders['coverage'],rr,budget,pred,structure) for rr in random[source]]
                            aggregate,_=a.control.aggregate_random_pairs(pairs)
                        else:aggregate,_=a.control.class_stratified_pair(ctx,orders['coverage'],orders[baseline],budget,pred,structure)
                        value=a.control.add_delta(dict(source=source,model=model,seed=seed,baseline=baseline,budget=budget,representation=representation,**aggregate))
                        controls.append(value);destination[baseline].append(value)
                pairs=[a.control.class_stratified_pair(ctx,orders['cross_coverage'],rr,budget,pred,structures) for rr in random[source]]
                aggregate,_=a.control.aggregate_random_pairs(pairs)
                cross_curve.append(a.control.add_delta(dict(budget=budget,**aggregate)))
            for baseline in curves:
                for metric in ('confusion','clusters'):
                    groups[source,baseline,metric][model].append(a.stats.normalized_auc(curves[baseline],f'delta_{metric}'))
                    sensitivity[source,baseline,metric][model].append(a.stats.normalized_auc(vit_curves[baseline],f'delta_{metric}'))
            for metric in ('confusion','clusters'):cross_groups[source,metric][model].append(a.stats.normalized_auc(cross_curve,f'delta_{metric}'))
        # Persist at each completed run, enabling live checks and failure recovery.
        write_csv(BASE/'comparison/tables/fixed_metrics.csv',fixed_rows)
        write_csv(BASE/'comparison/tables/random_metrics.csv',random_rows)
        write_csv(BASE/'comparison/tables/equal_error_metrics.csv',controls)
    def stats_table(values,representation):
        records=[]
        for (source,baseline,metric),runs in values.items():records.append(dict(source=source,baseline=baseline,representation=representation,**a.stats.paired_stats(runs,f'{source}_coverage_vs_{baseline}_{representation}',metric)))
        for row,p in zip(records,a.control.holm([r['permutation_p'] for r in records])):row['holm_all20_p']=p
        for family,field in [(('random',),'holm_primary4_p'),(('deepgini','dsa','kcenter'),'holm_strong12_p'),(('coverage_without_H',),'holm_ablation4_p')]:
            chosen=[r for r in records if r['baseline'] in family]
            for row,p in zip(chosen,a.control.holm([r['permutation_p'] for r in chosen])):row[field]=p
        return records
    result=stats_table(groups,'classifier');vit_result=stats_table(sensitivity,'vit')
    cross=[dict(risk_source=source,state_source='online' if source=='local' else 'local',**a.stats.paired_stats(values,f'cross_{source}',metric)) for (source,metric),values in cross_groups.items()]
    for row,p in zip(cross,a.control.holm([r['permutation_p'] for r in cross])):row['holm_four_p']=p
    write_csv(BASE/'comparison/tables/comparisons.csv',result)
    write_csv(BASE/'comparison/tables/model_aware_orders.csv',order_rows)
    write_csv(BASE/'robustness/tables/vit_cluster_comparisons.csv',vit_result)
    write_csv(BASE/'robustness/tables/cross_source_risk_state.csv',cross)
    assert len(fixed_rows)==1440 and len(random_rows)==240 and len(controls)==2400
    write_json(BASE/'comparison/final_audit.json',dict(status='pass',runs=24,methods=6,cross_source_extra=True,
               fixed_rows=len(fixed_rows),random_rows=len(random_rows),equal_error_rows=len(controls),all_method_quotas_checked=True,comparisons=result))
    strong_image_bootstrap(contexts,fixed)

def expected_weighted(counts,matched,universe):
    """Vectorized exact rarefaction of integer bootstrap error multiplicities."""
    from scipy.special import gammaln
    if universe==0:return np.zeros(len(matched))
    total=counts.sum(1);remaining=total[:,None]-counts;m=matched[:,None]
    valid=(remaining>=m)&(m>0)&(counts>0)
    logp=(gammaln(remaining+1)-gammaln(np.maximum(remaining-m,0)+1)
          +gammaln(np.maximum(total[:,None]-m,0)+1)-gammaln(total[:,None]+1))
    absent=np.ones_like(counts,dtype=float)
    absent[valid]=np.exp(np.minimum(logp[valid],0))
    certain=(m>0)&(counts>0)&(remaining<m);absent[certain]=0
    return (1-absent).sum(1)/universe

def strong_image_bootstrap(contexts,fixed):
    rng=np.random.default_rng(20261006);draws=5000
    weights={c:rng.multinomial(400,np.full(400,1/400),size=draws).astype(np.int16) for c in CLASSES}
    grouped=defaultdict(list)
    def type_counts(section,order,budget,pred,structures,metric,class_weights):
        chosen=np.asarray(order[:budget]);glob=section['global_indices'][chosen]
        wrong=pred[glob]!=LABELS[section['rows'][0]['class_name']]
        chosen=chosen[wrong];glob=glob[wrong]
        universe=len(structures['possible_predictions']) if metric=='confusion' else structures['cluster_count']
        values=pred[glob] if metric=='confusion' else np.asarray([structures['cluster_by_global'][int(i)] for i in glob])
        labels=structures['possible_predictions'] if metric=='confusion' else list(range(universe))
        output=np.zeros((draws,universe),dtype=np.int32)
        for j,label in enumerate(labels):
            if len(chosen):output[:,j]=class_weights[:,chosen[values==label]].sum(1)
        return output,universe
    for index,(model,seed) in enumerate(itertools.product(MODELS,SEEDS),1):
        note('compare',operation='fixed_panel_source_image_bootstrap',run=index,total=24)
        features,logits,pred=prediction(model,seed)
        structures=a.support.error_structures(contexts['local'],features,pred,model,seed)
        for source,ctx in contexts.items():
            orders=model_orders(source,contexts,fixed,model,seed,logits,features)
            for baseline,metric in itertools.product(('deepgini','dsa','kcenter'),('confusion','clusters')):
                curve=[np.zeros(draws)]
                for budget in (20,40,60,80,100):
                    diffs=[]
                    for c in CLASSES:
                        section=ctx['by_class'][c]
                        left,u=type_counts(section,orders['coverage'][c],budget,pred,structures[c],metric,weights[c])
                        right,_=type_counts(section,orders[baseline][c],budget,pred,structures[c],metric,weights[c])
                        matched=np.minimum(left.sum(1),right.sum(1))
                        diffs.append(expected_weighted(left,matched,u)-expected_weighted(right,matched,u))
                    curve.append(np.asarray(diffs).mean(0))
                grouped[source,baseline,metric].append(np.trapezoid(np.asarray(curve),x=np.arange(0,101,20),axis=0)/100)
    output=[]
    original=pd.read_csv(BASE/'comparison/tables/comparisons.csv')
    for (source,baseline,metric),samples in grouped.items():
        values=np.asarray(samples).mean(0)
        point=original[(original.source==source)&(original.baseline==baseline)&(original.metric==metric)].iloc[0].mean_delta_auc
        output.append(dict(source=source,baseline=baseline,metric=metric,observed_delta_auc=float(point),
                           ci95_low=float(np.percentile(values,2.5)),ci95_high=float(np.percentile(values,97.5)),
                           bonferroni12_low=float(np.quantile(values,.05/(2*12))),bonferroni12_high=float(np.quantile(values,1-.05/(2*12))),
                           replicates=draws,scope='fixed 24-model panel, selected sets and full-pool diagnostic universe; class-wise common source-image bootstrap'))
    write_csv(BASE/'comparison/tables/strong_baseline_image_bootstrap.csv',output)

def transfer():
    _,contexts,_=context();weak=read_csv(BASE/'transfer/weak_states_A_frozen.csv');result=[];grouped=defaultdict(lambda:defaultdict(list))
    scores=pd.read_csv(BASE/'selection/risk_scores.csv')
    identities={(r.source,r.sample_id):(float(r.h_midpoint),int(r.unknown_dimensions)) for r in scores.itertuples()}
    for model,seed in itertools.product(MODELS,SEEDS):
        _,_,pred=prediction(model,seed)
        for source,ctx in contexts.items():
            gaps=[]
            for item in [r for r in weak if r['source']==source and r['model']==model]:
                name=item['class_name'];section=ctx['by_class'][name];correct=pred[section['global_indices']]==LABELS[name]
                states=section['states'];pair=(item['feature'],item['value'])
                mask=section['matrix'][:,states.index(pair)].astype(bool) if pair in states else np.zeros(400,dtype=bool)
                n=int(mask.sum());recall=float(correct[mask].mean()) if n else None;overall=float(correct.mean())
                row=dict(source=source,model=model,seed=seed,class_name=name,feature=pair[0],value=pair[1],A_gap=float(item['gap']),
                         B_support=n,B_evaluable=n>=20,B_recall=recall,B_class_recall=overall,B_gap=recall-overall if n else None)
                if n>=20:gaps.append(recall-overall)
                controls=defaultdict(list)
                for i,r in enumerate(section['rows']):
                    if not mask[i]:controls[identities[source,r['sample_id']]].append(i)
                differences=[]
                for i,r in enumerate(section['rows']):
                    if mask[i] and controls.get(identities[source,r['sample_id']]):
                        differences.append(float(correct[i])-float(correct[controls[identities[source,r['sample_id']]]].mean()))
                row.update(risk_matchable_n=len(differences),risk_matched_gap=float(np.mean(differences)) if len(differences)>=10 else None)
                result.append(row)
            if gaps:grouped[source][model].append(float(np.mean(gaps)))
    comparisons=[]
    for source,values in grouped.items():
        comparisons.append(dict(source=source,**a.stats.paired_stats(values,f'A_frozen_weak_states_to_B_{source}','B_state_minus_class_recall')))
    for row,p in zip(comparisons,a.control.holm([r['permutation_p'] for r in comparisons])):row['holm_two_p']=p
    write_csv(BASE/'transfer/tables/state_validation.csv',result);write_csv(BASE/'transfer/tables/comparisons.csv',comparisons)
    summary=[]
    frame=pd.DataFrame(result)
    for source in ('local','online'):
        part=frame[frame.source==source];eligible=part[part.B_evaluable]
        summary.append(dict(source=source,frozen_state_models=len(part)//3,evaluable_state_models=len(eligible)//3,
                            negative_direction_fraction=float((eligible.B_gap<0).mean()) if len(eligible) else None,
                            mean_gap=float(eligible.B_gap.mean()) if len(eligible) else None))
    write_json(BASE/'transfer/summary.json',dict(status='complete',summary=summary,comparisons=comparisons))
    previous=pd.read_csv(ROOT/'artifacts/sixclass_difficulty_supplement_20261004/tables/model_robustness.csv')
    current=pd.read_csv(BASE/'confirmation/tables/model_robustness.csv');ranking=[]
    for source in ('local','online'):
        old=previous[previous.source==source].set_index('model').loc[list(MODELS)]
        new=current[current.source==source].set_index('model').loc[list(MODELS)]
        for metric in ('full_strata_auc','high_risk_f1'):
            ranking.append(dict(source=source,metric=metric,A_B_kendall_tau=float(kendalltau(old[metric],new[metric]).statistic)))
    write_csv(BASE/'transfer/tables/ranking_validation.csv',ranking)

def robust_risks():
    labels,contexts,_=context();dimension_maps=supplement.MAPS
    variants=['drop_size',*['drop_'+field for field in dimension_maps],'equal_normalized']
    assignments={};profiles=[]
    for source,ctx in contexts.items():
        for variant in variants:
            assignments[source,variant]={}
            for name in CLASSES:
                section=ctx['by_class'][name];values=[]
                for row in section['rows']:
                    risk=supplement.risk(row,labels[source][row['sample_id']]['features'])
                    if variant=='equal_normalized':value=risk['size']/3+sum(risk[field]/max(mapping.values()) for field,mapping in dimension_maps.items())
                    else:value=risk['h_midpoint']-risk[variant.removeprefix('drop_')]
                    values.append(value)
                ordered=sorted(range(400),key=lambda i:(values[i],section['rows'][i]['sample_id']))
                assignments[source,variant][name]=[ordered[t*80:(t+1)*80] for t in range(5)]
                profiles.extend(dict(source=source,variant=variant,class_name=name,tier=t+1,risk_mean=float(np.mean([values[i] for i in ordered[t*80:(t+1)*80]]))) for t in range(5))
    metrics=[];groups=defaultdict(lambda:defaultdict(list));quality=[]
    for model,seed in itertools.product(MODELS,SEEDS):
        _,_,pred=prediction(model,seed)
        for source,variant in assignments:
            curve=[]
            ctx=contexts[source]
            for t in range(5):
                selected=[int(ctx['by_class'][name]['global_indices'][i]) for name in CLASSES for i in assignments[source,variant][name][t]]
                value=supplement.metrics(ctx['rows'],pred,selected);curve.append(value['macro_f1'])
                metrics.append(dict(source=source,variant=variant,model=model,seed=seed,tier=t+1,**value))
            groups[source,variant][model].append(curve[-1]-curve[0])
        for name in CLASSES:
            section=contexts['local']['by_class'][name]
            sizes=np.asarray([min(float(r['target_pixel_width']),float(r['target_pixel_height'])) for r in section['rows']])
            wrong=pred[section['global_indices']]!=LABELS[name]
            for category in ('tiny','small','medium','large'):
                mask=np.asarray([a.b.scale_state(float(r['bbox_area_ratio']))==category for r in section['rows']])
                if mask.sum():
                    coefficient=float(spearmanr(sizes[mask],wrong[mask].astype(int)).statistic) if len(set(wrong[mask]))>1 and len(set(sizes[mask]))>1 else None
                    quality.append(dict(model=model,seed=seed,class_name=name,scale=category,support=int(mask.sum()),
                                        median_original_short_side_pixels=float(np.median(sizes[mask])),error_rate=float(wrong[mask].mean()),pixel_error_spearman=coefficient))
    comparisons=[dict(source=source,variant=variant,**a.stats.paired_stats(values,f'{source}_{variant}','Q5_minus_Q1_macro_f1')) for (source,variant),values in groups.items()]
    for row,p in zip(comparisons,a.control.holm([r['permutation_p'] for r in comparisons])):row['holm_twelve_p']=p
    write_csv(BASE/'robustness/tables/risk_variant_metrics.csv',metrics)
    write_csv(BASE/'robustness/tables/risk_variant_comparisons.csv',comparisons)
    write_csv(BASE/'robustness/tables/risk_variant_profiles.csv',profiles)
    write_csv(BASE/'robustness/tables/pixel_scale_quality.csv',quality)
    write_json(BASE/'robustness/status.json',dict(status='complete',risk_variants=6,comparisons=12,manual_state_truth=False))

def report():
    import plot_sixclass_difficulty as plot
    plot.BASE=BASE/'confirmation';plot.OUT=plot.BASE/'figures';plot.run()
    contrasts=read_csv(BASE/'confirmation/tables/difficulty_contrasts.csv')
    comps=read_csv(BASE/'comparison/tables/comparisons.csv')
    text=['# 六类独立样本确认结果','',
          '固定原24份分类权重；全新COCO目标六类各400，属于独立样本确认，并非全新数据来源。双自动VLM标签不等于人工真值。','',
          '## 完整层Q5−Q1','', '| 来源 | unknown处理 | F1差(pp) | 95% CI(pp) | Holm p |','|---|---|---:|---|---:|']
    for r in contrasts:text.append(f"| {r['source']} | {r['unknown_mode']} | {100*float(r['mean_delta_auc']):+.2f} | [{100*float(r['ci95_low']):+.2f},{100*float(r['ci95_high']):+.2f}] | {float(r['holm_six_p']):.5f} |")
    text+=['','## 相同风险配额下的诊断比较','', '| 来源 | 对照 | 指标 | AUC差 | 95% CI | 全20项Holm p |','|---|---|---|---:|---|---:|']
    for r in comps:text.append(f"| {r['source']} | {r['baseline']} | {r['metric']} | {float(r['mean_delta_auc']):+.4f} | [{float(r['ci95_low']):+.4f},{float(r['ci95_high']):+.4f}] | {float(r['holm_all20_p']):.5f} |")
    text+=['','主4项、强基线12项及删字段4项检验族见comparison/tables/comparisons.csv；全20项敏感性同时报告。不能只根据F1下降判断选样有效。',
           '', '8架构双侧符号翻转最小p为0.0078125，12/20项Holm族不能启动拒绝，未通过不能当零效应证据。固定24模型面板的强基线源图重采样区间见comparison/tables/strong_baseline_image_bootstrap.csv；区间不覆盖新架构或重新选样的不确定性。',
           '', '## A到B的诊断迁移','',json.dumps(json.loads((BASE/'transfer/summary.json').read_text()),ensure_ascii=False,indent=2),
           '', '## 稳健性与证据边界','',
           '风险权重/来源交叉/统一ViT错误簇和原始像素分析见robustness/tables/。所有负结果保存；不声明普遍优越、自动标签准确或模型经测试后变强。',
           '', '完整图表见confirmation/figures/；输入、权重、标签、模型无关次序和旧薄弱状态均在新分类预测前冻结。']
    (BASE/'report_zh.md').write_text('\n'.join(text)+'\n')
    verify(BASE/'pre_prediction_freeze.sha256.json');verify(BASE/'prediction_freeze.sha256.json');verify(BASE/'preparation_freeze.sha256.json')
    write_json(BASE/'final_audit.json',dict(status='pass',targets=2400,runs=24,all_stages_complete=True,source_freezes_unchanged=True,
               independence='new samples; not new dataset source',annotation_truth='automatic only'))
    primary=[r for r in comps if r['baseline']=='random']
    supported=sum(float(r['mean_delta_auc'])>0 and float(r['holm_primary4_p'])<.05 for r in primary)
    (BASE/'claim_evidence.md').write_text('# 本轮结论与结果证据\n\n'
        '独立风险响应：confirmation/tables/difficulty_contrasts.csv，包含全部6项校正结果，不省略负或不显著结果。\n\n'
        f'同风险Coverage相对Random：comparison/tables/comparisons.csv，4项主结果中正向且校正通过{supported}/4；不由此声称普遍优势。\n\n'
        '强基线：同表12项与全20敏感性并报；8架构检验分辨率限制明确，固定面板图像区间为补充估计。\n\n'
        '薄弱状态迁移：transfer/tables/state_validation.csv、comparisons.csv；保留不可评价状态，不替换失败预测。\n\n'
        '规则与评价依赖：robustness/tables/完整结果；自动状态无人工真值，独立样本不等于新数据来源。\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['infer','confirm','compare','transfer','robustness','report'],required=True)
    stage=parser.parse_args().stage
    {'infer':inference,'confirm':confirm,'compare':compare,'transfer':transfer,'robustness':robust_risks,'report':report}[stage]()
