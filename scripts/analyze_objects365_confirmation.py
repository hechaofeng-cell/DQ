#!/usr/bin/env python3
"""Frozen analyses on the new source; complete negative as well as positive outcomes."""
import argparse
import itertools
import json
import numpy as np
from objects365_confirmation_common import *


def support_audit(common):
    labels,contexts,_=common.context();records=[]
    weak=read_csv(BASE/'transfer/weak_states_A_frozen.csv')
    unique=sorted({(r['source'],r['class_name'],r['feature'],r['value']) for r in weak})
    for source,name,field,value in unique:
        section=contexts[source]['by_class'][name]
        pair=field,value
        n=int(section['matrix'][:,section['states'].index(pair)].sum()) if pair in section['states'] else 0
        records.append(dict(source=source,class_name=name,feature=field,value=value,support=n,evaluable=n>=20))
    write_csv(BASE/'transfer/tables/pre_prediction_support.csv',records)
    write_json(BASE/'transfer/support_audit.json',dict(status='complete_before_prediction',conditions=len(records),
        supported=sum(r['evaluable'] for r in records),support_minimum=20,prediction_read=False))
    assert not list((BASE/'benchmark').glob('**/candidate_outputs.npz'))
    common.freeze(BASE/'support_freeze.sha256.json',[BASE/'transfer/tables/pre_prediction_support.csv',BASE/'transfer/support_audit.json'])


def report(common):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.8,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    figures=BASE/'figures';figures.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(figures/f'{name}.svg',bbox_inches='tight');fig.savefig(figures/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
    full=pd.read_csv(BASE/'confirmation/tables/full_stratum_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for ax,source in zip(axes,('local','online')):
        part=full[(full.source==source)&(full.unknown_mode=='midpoint')]
        for i,model in enumerate(MODELS):
            subset=part[part.model==model].groupby('tier').macro_f1.agg(['mean','std'])
            ax.plot(subset.index,100*subset['mean'],marker='o',markersize=3,lw=1,color=colors[i],label=model)
            ax.fill_between(subset.index,100*(subset['mean']-subset['std'].fillna(0)),
                100*(subset['mean']+subset['std'].fillna(0)),color=colors[i],alpha=.08)
        ax.set(title=f'Objects365 / {source} labels',xlabel='Within-class risk tier',ylabel='Macro-F1 (%)',xticks=range(1,6),ylim=(0,100))
    axes[-1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left')
    save(fig,'risk_model_profiles')
    validation=pd.read_csv(BASE/'transfer/tables/state_validation.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for ax,source in zip(axes,('local','online')):
        part=validation[(validation.source==source)&validation.B_evaluable]
        means=part.groupby(['class_name','feature','value']).B_gap.mean().sort_values()
        ax.axvline(0,color='.5',lw=.7)
        if len(means):
            ax.scatter(100*means.values,np.arange(len(means)),s=12,color='#3775BA')
            ax.set(yticks=np.arange(len(means)),yticklabels=[' / '.join(k) for k in means.index])
        else:ax.text(.5,.5,'No condition reaches support 20',ha='center',va='center',transform=ax.transAxes)
        ax.set(title=source,xlabel='State minus class Recall (pp)')
    save(fig,'frozen_state_reproduction')
    contrasts=read_csv(BASE/'confirmation/tables/difficulty_contrasts.csv')
    comparisons=read_csv(BASE/'comparison/tables/comparisons.csv')
    source=json.loads((BASE/'metadata/source_audit.json').read_text())
    transfer=json.loads((BASE/'transfer/summary.json').read_text())
    prep=json.loads((BASE/'preparation_audit.json').read_text())
    support=json.loads((BASE/'transfer/support_audit.json').read_text())
    lines=['# Objects365新来源确认结果','',f'完成时间：{now()}。','',
        '原8架构×3种子24份分类权重固定，六类各400目标。本地/线上VLM分别标注，无本研究新增人工状态复核。',
        '来源为Objects365 v2 train发布分包中的未用图像，仅用于模型测试，不用于训练。验证集sheep不足的原审计保留。',
        '该池条件于预定分包前缀、资格过滤和排重；不能代表所有Objects365场景，不能排除预训练数据重叠。','',
        '## 风险层确认','', '| 来源 | unknown规则 | Q5−Q1 F1 (pp) | 95%架构/种子区间(pp) | 原六项Holm p |',
        '|---|---|---:|---|---:|']
    for r in contrasts:
        lines.append(f"| {r['source']} | {r['unknown_mode']} | {100*float(r['mean_delta_auc']):+.2f} | [{100*float(r['ci95_low']):+.2f}, {100*float(r['ci95_high']):+.2f}] | {float(r['holm_six_p']):.5f} |")
    lines+=['','区间条件于当前架构面板和候选池。原图重采样区间另见confirmation/tables/source_image_bootstrap.csv。',
        '高低层差异不代表中间层单调，也不说明覆盖率提高具有因果效应。','',
        '![风险层模型曲线](figures/risk_model_profiles.png)','', '阴影为三个种子的标准差，不是置信区间。',
        '', '## 原A薄弱条件的确认','',
        f"新预测前检查{support['conditions']}项来源/类别/状态条件，其中{support['supported']}项支持至少20。",'',
        '沿用全部175条旧状态/模型组合，不依据本次错误重选；不可评价、非负差值和未复现均保留。','',
        '```json',json.dumps(transfer,ensure_ascii=False,indent=2),'```','',
        '![冻结状态差值](figures/frozen_state_reproduction.png)','',
        '状态图是可评价组合与种子的描述性均值，不等于独立逐条件显著检验。条件关联不证明模型内部错误根因。','',
        '## 同预算次要方法结果','', '| 来源 | 对照 | 指标 | AUC差 | 原家族Holm p | 全20项Holm p |',
        '|---|---|---|---:|---:|---:|']
    for r in comparisons:
        family='holm_primary4_p' if r['baseline']=='random' else 'holm_ablation4_p' if r['baseline']=='coverage_without_H' else 'holm_strong12_p'
        lines.append(f"| {r['source']} | {r['baseline']} | {r['metric']} | {float(r['mean_delta_auc']):+.5f} | {float(r[family]):.5f} | {float(r['holm_all20_p']):.5f} |")
    lines+=['','这些是逐类等错误数后的诊断广度AUC差，不是F1变化。原8架构检验的分辨率限制仍适用。',
        '不引入未收敛GLMM或有重复权重偏差疑问的非线性图像bootstrap强化推断。','',
        '## 独立性、成本与复现','', '```json',json.dumps(prep,ensure_ascii=False,indent=2),'```','',
        '请求账本见cost/request_attempts.jsonl；实际货币费用未取得账单时为未知。阶段日志见logs/，完整样本清单见manifests/。',
        '新样本支持不足和技术替换限制单独报告；自动状态分歧并未消失。旧B/C与本次不支持的结果均保留。']
    (BASE/'report_zh.md').write_text('\n'.join(lines)+'\n')
    for filename in ('execution_code_freeze.sha256.json','preparation_freeze.sha256.json','pre_prediction_freeze.sha256.json',
                     'support_freeze.sha256.json','prediction_freeze.sha256.json'):
        common.verify(BASE/filename)
    paths=list((BASE/'benchmark/runs').glob('*/*/candidate_outputs.npz'))
    assert len(paths)==24
    expected=[r['sample_id'] for r in read_csv(BASE/'manifests/candidate_pool.csv')]
    for p in paths:
        with np.load(p) as cache:
            assert cache['sample_ids'].tolist()==expected and cache['logits'].shape==(2400,6) and np.isfinite(cache['logits']).all()
    write_json(BASE/'final_audit.json',dict(status='pass',targets=2400,runs=24,freeze_checks='pass',source=source['source'],
        risk_rows=len(contrasts),method_rows=len(comparisons),transfer_support=support,manual_review=False,weights_retrained=False))
    (BASE/'claim_evidence.md').write_text('# 新来源确认：结果入口\n\n风险差异→confirmation/tables/difficulty_contrasts.csv；源图不确定性→confirmation/tables/source_image_bootstrap.csv；冻结状态→transfer/tables/state_validation.csv和transfer/summary.json；方法比较→comparison/tables/comparisons.csv。\n\n支持等级按实际方向、区间、预定检验族和支持量决定，不因流程完成就判定普遍有效。\n')
    (BASE/'review.md').write_text('# 结果完整性\n\n所有冻结哈希与24份2400×6预测通过。原主次检验族与失败结果保留。状态自动标注，无人工真值；仅代表本次新来源预定分包候选。报告列出全部20项方法比较，不声明普遍优势。\n')
    with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{now()}] 新来源确认完成，数值与负结果见report_zh.md及final_audit.json。\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,
        choices=('freeze_selection','support','infer','confirm','compare','transfer','report'))
    stage=parser.parse_args().stage
    common,a,selection,analysis=activate()
    if stage=='freeze_selection':
        (BASE/'embeddings').mkdir(parents=True,exist_ok=True)
        selection.build()
        path=BASE/'pre_prediction_freeze.sha256.json'
        mapping=json.loads(path.read_text())
        for p in code_paths()+[BASE/'registration.json',BASE/'execution_code_freeze.sha256.json']:
            mapping[str(p)]=sha256(p)
        assert not list((BASE/'benchmark').glob('**/candidate_outputs.npz'))
        write_json(path,mapping)
    elif stage=='support':support_audit(common)
    elif stage=='report':report(common)
    else:{'infer':analysis.inference,'confirm':analysis.confirm,'compare':analysis.compare,'transfer':analysis.transfer}[stage]()


if __name__=='__main__':main()
