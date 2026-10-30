#!/usr/bin/env python3
"""Reuses original controls at N=400; all outputs are in the new study."""
import argparse
import itertools
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from automatic_strengthening_common import *


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans'],
        'font.size':8,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    return plt


def savefig(fig,name):
    folder=BASE/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{name}.svg',bbox_inches='tight');fig.savefig(folder/f'{name}.png',dpi=200,bbox_inches='tight')
    import xml.etree.ElementTree as ET
    root=ET.parse(folder/f'{name}.svg').getroot()
    assert any(element.tag.endswith('text') for element in root.iter()), 'SVG labels must remain editable'


def dense_curves(common):
    from analyze_sixclass_difficulty import metrics
    from run_multiclass_object_selection_benchmark import class_state_metrics
    _,contexts,_=common.context()
    orders=read_csv(D/'confirmation/tables/stratified_coverage_order.csv')
    print('dense order columns',list(orders[0]),flush=True)
    output=[]
    for source,ctx in contexts.items():
        lookup={r['sample_id']:i for i,r in enumerate(ctx['rows'])}
        chosen={name:[r['sample_id'] for r in sorted((r for r in orders if r['source']==source and r['class_name']==name),key=lambda r:int(r['rank']))] for name in CLASSES}
        for model,seed in itertools.product(MODELS,SEEDS):
            with np.load(D/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
                assert saved['sample_ids'].tolist()==[r['sample_id'] for r in ctx['rows']]
                pred=saved['logits'].argmax(1)
            for budget in range(5,101,5):
                indices=[lookup[iid] for name in CLASSES for iid in chosen[name][:budget]]
                coverage=[]
                for name in CLASSES:
                    section=ctx['by_class'][name];local_lookup={r['sample_id']:i for i,r in enumerate(section['rows'])}
                    selected=[local_lookup[iid] for iid in chosen[name][:budget]]
                    correct=pred[section['global_indices']]==LABELS[name]
                    coverage.append(class_state_metrics(section['matrix'],selected,correct)['frequency_coverage'])
                output.append(dict(source=source,model=model,seed=seed,budget=budget,total_targets=6*budget,
                    coverage=float(np.mean(coverage)),sequence='original non-common-quota stratified Coverage; descriptive only',**metrics(ctx['rows'],pred,indices)))
    write_csv(D/'confirmation/tables/dense_cumulative_metrics.csv',output)


def report_d(common):
    dense_curves(common)
    plt=plotting();full=pd.read_csv(D/'confirmation/tables/full_stratum_metrics.csv')
    dense=pd.read_csv(D/'confirmation/tables/dense_cumulative_metrics.csv');comps=pd.read_csv(D/'comparison/tables/comparisons.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.2),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for ax,source in zip(axes,('local','online')):
        part=full[(full.source==source)&(full.unknown_mode=='midpoint')]
        for i,model in enumerate(MODELS):
            curve=part[part.model==model].groupby('tier').macro_f1.mean()
            ax.plot(curve.index,100*curve.values,marker='o',markersize=3,lw=1,color=colors[i],label=model)
        ax.set(title=f'D: {source} risk labels',xlabel='Within-class risk tier',ylabel='Macro-F1 (%)',xticks=[1,2,3,4,5],ylim=(0,100))
    axes[-1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left');savefig(fig,'D_risk_model_profiles');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9.5,5.4),layout='constrained')
    for row,source in enumerate(('local','online')):
        mean=dense[dense.source==source].groupby('budget').mean(numeric_only=True)
        for col,metric in enumerate(('macro_f1','accuracy','macro_recall')):
            axes[row,col].plot(100*mean.coverage,100*mean[metric],'-o',lw=1,markersize=2,color='#3775BA')
            axes[row,col].set(xlabel='Candidate-relative state coverage (%)',ylabel=metric.replace('_',' ').title()+' (%)',title=source)
    savefig(fig,'D_coverage_performance_measured_20_points');plt.close(fig)
    contrasts=read_csv(D/'confirmation/tables/difficulty_contrasts.csv')
    lines=['# Open Images未用样本确认D','',
        '六类各400目标，24份冻结权重，双自动VLM；OI不同于COCO训练，但历史A已使用该来源。以下结果不能称为全新来源迁移或人工状态真值。','',
        '## 风险响应','', '| 来源 | unknown规则 | Q5−Q1 F1 (pp) | 95%架构/种子区间(pp) | Holm六项p |',
        '|---|---|---:|---|---:|']
    for r in contrasts:lines.append(f"| {r['source']} | {r['unknown_mode']} | {100*float(r['mean_delta_auc']):+.2f} | [{100*float(r['ci95_low']):+.2f}, {100*float(r['ci95_high']):+.2f}] | {float(r['holm_six_p']):.5f} |")
    lines+=['','## 全部方法对照','', '| 来源 | 对照 | 诊断 | AUC差 | 原全20项Holm p |','|---|---|---|---:|---:|']
    for r in comps.to_dict('records'):lines.append(f"| {r['source']} | {r['baseline']} | {r['metric']} | {r['mean_delta_auc']:+.5f} | {r['holm_all20_p']:.5f} |")
    lines+=['','原主4项、强基线12项及删字段4项分别见完整CSV。没有把不显著解释为零效应。',
        '不复用存在非线性重复权重偏差疑问的诊断图像bootstrap来加强结论；原架构/种子区间与主检验保留。',
        '20点覆盖—性能图来自实际累计集合，属于原非共同配额Coverage序列的展示，原五点主AUC没有改变。',
        '', '## 冻结弱状态确认','', '```json',json.dumps(json.loads((D/'transfer/summary.json').read_text()),ensure_ascii=False,indent=2),'```']
    (D/'report_zh.md').write_text('\n'.join(lines)+'\n')
    for name in ('preparation_freeze.sha256.json','pre_prediction_freeze.sha256.json','prediction_freeze.sha256.json'):common.verify(D/name)
    write_json(D/'final_audit.json',dict(status='pass',targets=2400,runs=24,source='Open Images unused train images',manual_review=False))


def final_report():
    if not (D/'final_audit.json').exists():
        existing_report();return
    plt=plotting();official=pd.read_csv(BASE/'automatic_audit/official_attribute_reference.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax,field in zip(axes,('occlusion_level','truncation_level')):
        part=official[official.feature==field]
        # Pool-level summaries retain denominator weighting and show support.
        for j,source in enumerate(('local','online')):
            means=[];names=[]
            for pool in sorted(part.pool.unique()):
                rows=part[(part.source==source)&(part.pool==pool)]
                total=rows.known.sum()
                if total:means.append(100*(rows.tp.sum()+rows.tn.sum())/total);names.append(pool)
            ax.bar(np.arange(len(names))+(j-.5)*.3,means,width=.3,label=source,color=['#3775BA','#9A4D8E'][j])
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylabel='Reference binary agreement (%)',title=field,ylim=(0,100))
        ax.legend()
    savefig(fig,'official_reference_agreement');plt.close(fig)
    source_script=BASE/'figures/analyze_automatic_strengthening.py'
    shutil.copy2(Path(__file__),source_script)
    shutil.copy2(ROOT/'scripts/automatic_strengthening_common.py',BASE/'figures/automatic_strengthening_common.py')
    # Source copies are provenance; normal reproduction uses repository script paths.
    cases=read_csv(E/'tables/condition_validation.csv') if (E/'tables/condition_validation.csv').exists() else []
    lines=['# 无人工复核补强结果','',f'生成：{now()}。','',
        '全部分类器保持冻结；状态来自本地/线上VLM，无本研究人工状态复核。官方属性核对仅为部分字段参考。','',
        '## 完整性','', '```json',json.dumps(json.loads((D/'final_audit.json').read_text()),ensure_ascii=False,indent=2),'```','',
        '## 数据与风险确认','', '[D完整结果](pool_D/report_zh.md)。B/C旧结果不删除；D为未用OI图像，不声称全新发现来源。','',
        '## 条件复现','', '[E状态确认](suite_E/report_zh.md)。同时保留未复现及支持不足条件。','',
        '## 自动状态参考','', '[属性对照](automatic_audit/official_attribute_reference.csv)、[双VLM一致性](automatic_audit/dual_vlm_agreement.csv)、[目标输入审计](automatic_audit/audit.json)。',
        '双VLM一致不能替代标签准确率；known与unknown分别报告。','',
        '## 统计补充','', '[统计执行状态](statistics/status.json)。变分贝叶斯GLMM使用近似后验区间，不替代原精确检验；拟合失败也保留。','',
        '## 测试交付','', '[状态回归测试清单](suite_E/regression_suite.csv)、[复评脚本](../../scripts/replay_visual_regression_suite.py)。',
        '固定套件用于同输入同模型复评；它不证明模型已经得到修复。','',
        '## 成本','', '本地/线上请求原始记录在pool_D/annotations与suite_E/annotations；阶段墙钟和失败记录在stages/、logs/。',
        '货币费用无账单依据时记为未知，不由token估计冒充实际账单。','',
        '## 结论边界','', '风险差异、额外诊断广度和状态迁移分别陈述；没有全面方法优越性、因果根因或人工标签准确性声明。']
    (BASE/'report_zh.md').write_text('\n'.join(lines)+'\n')
    audits=[json.loads((D/'final_audit.json').read_text()),json.loads((E/'final_audit.json').read_text()),json.loads((BASE/'automatic_audit/audit.json').read_text())]
    write_json(BASE/'final_audit.json',dict(status='pass' if all(r['status']=='pass' for r in audits) else 'partial',audits=audits,
                                          statistics=json.loads((BASE/'statistics/status.json').read_text()) if (BASE/'statistics/status.json').exists() else {'status':'unavailable'}))
    (BASE/'claim_evidence.md').write_text('# 结论—证据\n\n风险差异→pool_D/confirmation/tables/difficulty_contrasts.csv；诊断增益→pool_D/comparison/tables/comparisons.csv；弱状态→pool_D/transfer/与suite_E/tables/。全部正负结果保留。\n\n自动字段参考→automatic_audit/；双VLM一致率不是全字段准确率；D不是全新发现来源；统计补充不替代原检验。\n')
    (BASE/'review.md').write_text('# 执行门禁\n\n数值追溯到各CSV和原始预测；数据、权重、协议冻结核对见final_audit.json。结论按各项实际支持等级陈述；自动状态与来源范围保留。混合模型方法依据statsmodels作者文档，近似后验区间不称频率学置信区间。\n')
    with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{now()}] 主流水线结束，最终状态和统计限制见final_audit.json与report_zh.md。\n')


def regression_kit():
    examples=read_csv(ROOT/'artifacts/empirical_evidence_strengthening_20261005/case_images.csv');records=[];seen=set()
    metadata={name:{r['sample_id']:r for r in read_csv(path/'manifests/candidate_pool.csv')} for name,path in [('B',B),('C',C)]}
    for example in examples:
        pool='B' if example['study']==B.name else 'C'
        path=B if pool=='B' else C;model=example['model']
        if model not in MODELS:
            continue  # Annotation-only examples do not define a classifier regression assertion.
        sample=metadata[pool][example['sample_id']]
        for seed in SEEDS:
            key=(pool,model,seed,sample['sample_id'])
            if key in seen:continue
            seen.add(key)
            with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as saved:
                ids=saved['sample_ids'].tolist();i=ids.index(sample['sample_id']);logits=saved['logits'][i]
            weight=ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/model/'oi_gap_add0'/str(seed)/'best.pt'
            records.append(dict(**sample,pool=pool,model=model,seed=seed,condition_feature=example['feature'],condition_value=example['value'],
                case_type=example['case_type'],annotation_source=example['source'],expected_predicted_label=int(logits.argmax()),
                expected_logits=json.dumps(logits.tolist()),checkpoint_path=str(weight),checkpoint_sha256=sha256(weight)))
    write_csv(BASE/'regression_suite.csv',records)
    write_json(BASE/'regression_suite_spec.json',dict(status='exported',origin='existing exploratory diagnostic cases; not a new confirmation pool',
        comparisons=len(records),unique_images=len({(r['pool'],r['sample_id']) for r in records}),new_training=False,
        command=f'{PYTHON} {ROOT}/scripts/replay_visual_regression_suite.py --manifest {BASE}/regression_suite.csv'))
    return examples


def existing_report():
    plt=plotting();examples=regression_kit()
    official=pd.read_csv(BASE/'automatic_audit/official_attribute_reference.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),layout='constrained')
    for ax,field in zip(axes,('occlusion_level','truncation_level')):
        for j,source in enumerate(('local','online')):
            values=[]
            for name in CLASSES:
                rows=official[(official.feature==field)&(official.source==source)&(official.class_name==name)]
                n=rows.known.sum();values.append(100*(rows.tp.sum()+rows.tn.sum())/n if n else np.nan)
            ax.bar(np.arange(6)+(j-.5)*.32,values,width=.32,color=['#3775BA','#9A4D8E'][j],label=source)
        ax.set(xticks=np.arange(6),xticklabels=CLASSES,ylabel='Reference binary agreement (%)',title=field,ylim=(0,100));ax.legend(fontsize=7)
    savefig(fig,'official_reference_agreement');plt.close(fig)
    agreement=pd.read_csv(BASE/'automatic_audit/dual_vlm_agreement.csv')
    fields=sorted(set(agreement.feature)-{'derived_risk'})
    fig,axes=plt.subplots(1,3,figsize=(10.3,4.1),layout='constrained')
    for ax,pool in zip(axes,('A','B','C')):
        table=agreement[(agreement.pool==pool)&agreement.feature.isin(fields)].pivot(index='feature',columns='class_name',values='exact_agreement').reindex(index=fields,columns=CLASSES)
        image=ax.imshow(table.values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xticks=np.arange(6),xticklabels=CLASSES,yticks=np.arange(len(fields)),yticklabels=fields if pool=='A' else [],title=f'{pool}: dual-VLM agreement')
        ax.tick_params(axis='x',labelrotation=40)
    fig.colorbar(image,ax=axes,label='Exact agreement',shrink=.8);savefig(fig,'dual_vlm_field_agreement');plt.close(fig)
    # Two predetermined diagnostic conditions, plus explicitly different unreplicated conditions.
    specifications=[('dog','body_orientation','back','state_failure'),('dog','body_orientation','back','state_success_counterexample'),
        ('dog','target_scale','medium','state_success_counterexample'),('sheep','blur_level','severe','state_failure'),
        ('sheep','blur_level','severe','state_success_counterexample'),('sheep','interaction_context','with_person','state_success_counterexample')]
    selected=[]
    for name,feature,value,kind in specifications:
        options=[r for r in examples if r['class_name']==name and r['feature']==feature and r['value']==value and r['image_kind']==kind]
        if not options:raise ValueError(f'Missing prespecified panel case: {name}/{feature}/{value}/{kind}')
        selected.append(min(options,key=lambda r:r['sample_id']))
    fig,axes=plt.subplots(2,3,figsize=(9.2,6.0),layout='constrained')
    from PIL import Image
    for ax,row in zip(axes.flat,selected):
        with Image.open(row['marked_image_path']) as image:ax.imshow(image.convert('RGB'))
        predictions=json.loads(row['predicted_classes'])
        ax.set_title(f"{row['class_name']}: {row['feature']}={row['value']}\n{row['model']} / {row['source']}; seeds: {', '.join(predictions)}",fontsize=7)
        ax.set_xlabel(row['sample_id'],fontsize=5);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    savefig(fig,'diagnostic_success_failure_counterexamples');plt.close(fig)
    write_csv(BASE/'figures/diagnostic_panel_cases.csv',selected)
    stats_status=json.loads((BASE/'statistics/status.json').read_text()) if (BASE/'statistics/status.json').exists() else {'status':'pending'}
    effects=[]
    for source in ('local','online'):
        path=BASE/'statistics'/source/'risk_effects.csv'
        if path.exists():effects.extend(read_csv(path))
    if effects:
        fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
        for i,row in enumerate(effects):
            x=100*float(row['marginal_error_difference_median']);left=100*float(row['approximate_cr95_delta_low']);right=100*float(row['approximate_cr95_delta_high'])
            ax.errorbar(x,i,xerr=[[x-left],[right-x]],fmt='o',color='#3775BA',capsize=2)
        ax.axvline(0,color='gray',lw=.8);ax.set(yticks=np.arange(len(effects)),yticklabels=[f"{r['dataset']} / {r['source']} / converged={r['fit_converged']}" for r in effects],
            xlabel='Q5 - Q1 marginal error probability (pp); approximate VB CrI',title='Supplementary model, not the original primary test')
        savefig(fig,'supplementary_risk_effects');plt.close(fig)
    shutil.copy2(Path(__file__),BASE/'figures/analyze_automatic_strengthening.py')
    write_csv(BASE/'figures/risk_effect_input.csv',effects)
    audit=json.loads((BASE/'automatic_audit/audit.json').read_text());availability=json.loads((BASE/'audit/availability.json').read_text())
    summaries=[]
    for (source,feature),part in official.groupby(['source','feature']):
        n=int(part.known.sum());tp=int(part.tp.sum());tn=int(part.tn.sum());fp=int(part.fp.sum());fn=int(part.fn.sum())
        summaries.append(dict(source=source,feature=feature,known=n,exact_agreement=(tp+tn)/n,
            positive_agreement=tp/(tp+fn) if tp+fn else None,negative_agreement=tn/(tn+fp) if tn+fp else None,unknown=int(part.unknown.sum())))
    write_csv(BASE/'automatic_audit/reference_summary.csv',summaries)
    lines=['# 无人工复核补强：已完成工作与新池限制','',f'生成：{now()}。','',
        '## 执行状态','', '已有A/B/C自动核对、统计补充和测试案例交付完成执行；新D/E未执行。',
        '按全历史排重，Open Images训练集939个合格sheep目标全部已使用，无法满足新池六类各400。没有替换来源、复用旧图或调用新标注填充。',
        '初始car元数据流完整性尚未通过，因此不报告其全量可用数；不影响sheep数量限制。','',
        '## 官方属性参考','', '| VLM | 字段 | 已知样本 | 二值参考一致率 | 正类一致率 | 负类一致率 | unknown |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['source']} | {r['feature']} | {r['known']} | {r['exact_agreement']:.2%} | {r['positive_agreement']:.2%} | {r['negative_agreement']:.2%} | {r['unknown']} |")
    lines+=['','官方参考只核对遮挡/截断二值归并，不证明全部严重程度或其他字段准确；双VLM一致不能替代真值。',
        '', '## 统计执行','', '```json',json.dumps(stats_status,ensure_ascii=False,indent=2),'```',
        '变分贝叶斯区间是近似后验区间；原主检验与所有负结果保持不变。拟合未收敛或边界项不升级为确认结论。','',
        '## 测试资产','', '[回归测试清单](regression_suite.csv)、[案例面板](figures/diagnostic_success_failure_counterexamples.svg)。',
        '套件由已有探索性案例导出，不能称为新样本状态确认；同状态正确实例和未复现条件均保留。','',
        '## 后续需要的决定','', 'D/E要继续，需要在新分类预测前另立来源修订，先做六类官方框数量及历史重叠审计，再冻结新来源清单。当前没有擅自改用其他来源。']
    (BASE/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write_json(BASE/'final_audit.json',dict(status='partial_complete',automatic_audit=audit,availability=availability,statistics=stats_status,
        D_E_executed=False,new_VLM_requests=0,manual_review=False))
    (BASE/'claim_evidence.md').write_text('# 本轮证据与边界\n\n官方属性参考→automatic_audit/reference_summary.csv及逐类CSV；风险稳定性→automatic_audit/existing_risk_sensitivity.csv；统计→statistics/（近似变分后验，原检验保留）；案例→figures/diagnostic_panel_cases.csv；回归套件→regression_suite.csv。\n\nD/E因合格新sheep不足未运行，不声称完成外部补强。二值参考一致率不等于全状态准确率。\n')
    (BASE/'review.md').write_text('# 本轮审阅\n\n已有自动核对输入门禁见automatic_audit/audit.json。统计拟合门禁逐来源/指标报告，不替代主检验。新数据门禁未通过：无法满足未用OI sheep数量。论文不得填入D/E结果或宣称补强全部完成。\n')
    with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{now()}] 完成可独立执行的证据补强；D/E数量门禁未通过，输出部分完成报告和真实案例资产。\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,
        choices=('freeze_selection','infer','confirm','compare','transfer','robustness','report_D','final_report'))
    stage=parser.parse_args().stage
    if stage=='final_report':final_report();return
    common,analysis,selection,confirmation=activate_d()
    if stage=='freeze_selection':
        (D/'embeddings').mkdir(parents=True,exist_ok=True);selection.build()
        mapping=json.loads((D/'pre_prediction_freeze.sha256.json').read_text())
        mapping[str(BASE/'execution_freeze.sha256.json')]=sha256(BASE/'execution_freeze.sha256.json')
        for path,digest in json.loads((BASE/'execution_freeze.sha256.json').read_text()).items():mapping[path]=digest
        assert not list((D/'benchmark').glob('**/candidate_outputs.npz'))
        write_json(D/'pre_prediction_freeze.sha256.json',mapping);return
    if stage=='report_D':report_d(common);return
    if stage=='compare':
        # This optional legacy analysis is scientifically questionable and not a main result.
        confirmation.strong_image_bootstrap=lambda contexts,fixed:write_json(D/'comparison/nonlinear_bootstrap_not_used.json',
            dict(status='omitted',reason='unresolved duplicate-weight rarefaction bias; original paired architecture inference unchanged'))
    {'infer':confirmation.inference,'confirm':confirmation.confirm,'compare':confirmation.compare,'transfer':confirmation.transfer,
     'robustness':confirmation.robust_risks}[stage]()
    if stage=='confirm':
        path=D/'confirmation/summary.json';value=json.loads(path.read_text())
        value.update(analysis_type='unused_openimages_sample_confirmation',independence='OI distinct from COCO training, already used in discovery A')
        write_json(path,value)


if __name__=='__main__':main()
