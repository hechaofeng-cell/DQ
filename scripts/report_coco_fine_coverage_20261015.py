#!/usr/bin/env python3
"""Reproducible SVG/PNG plots and Chinese report from the COCO study outputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qe_coco_fine_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('resnet18','resnet50','mobilenet_v3_small','mobilenet_v3_large','densenet121','efficientnet_b0','convnext_tiny','swin_t')
NAMES = ('ResNet-18','ResNet-50','MobileNetV3-S','MobileNetV3-L','DenseNet-121','EfficientNet-B0','ConvNeXt-T','Swin-T')
CLASSES = ('dog','cat','horse','sheep','person','car')
COLORS = ['#A6ACB5','#B3CDE3','#83A6C8','#E7CBA9','#D49569']


def rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def mean(data, key, **filters):
    values = [float(r[key]) for r in data if all(str(r.get(k)) == str(v) for k,v in filters.items()) and r.get(key) not in ('',None)]
    return float(np.mean(values)) if values else np.nan


def table(headers, records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in records])


def plot_style():
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'sans-serif',
        'font.sans-serif':['DejaVu Sans','Liberation Sans'],'font.size':9,'axes.titlesize':11,
        'axes.labelsize':9,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,
        'xtick.direction':'out','ytick.direction':'out','axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','savefig.facecolor':'white'})


def save(fig, folder, name):
    fig.savefig(folder/(name+'.svg'), bbox_inches='tight')
    fig.savefig(folder/(name+'.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_plots(root, model, paired, scoring, tiers, conditions, scores, fields, dataset):
    folder = root/'figures'; folder.mkdir(exist_ok=True)
    plot_style()
    # 1. The actual five-dimensional decomposition, and uncertainty by class.
    dims = ('scale','visibility','occlusion','truncation','orientation')
    fig, axes = plt.subplots(1, 2, figsize=(11.5,3.7), layout='constrained')
    bottom = np.zeros(6)
    for d, color in zip(dims, ['#B3CDE3','#B9D9C5','#E8CEA9','#D8C5E5','#D5D8DB']):
        v = [20*mean(scores, d+'_midpoint', scheme='equal_parts', class_name=c) for c in CLASSES]
        axes[0].bar(CLASSES, v, bottom=bottom, color=color, label=d.capitalize(), width=.7)
        bottom += v
    axes[0].set(ylabel='Mean score contribution (0-20 per dimension)', title='A  Local five-dimensional score composition')
    axes[0].legend(ncol=3, fontsize=7, loc='upper left', bbox_to_anchor=(0,1.0), frameon=False)
    axes[0].set_ylim(0, max(bottom)*1.45)
    for offset, scheme, color, label in [(-.13,'equal_global',COLORS[2],'Global'),(.13,'equal_parts',COLORS[4],'Local')]:
        v = [mean(scores,'score_upper',scheme=scheme,class_name=c)-mean(scores,'score_lower',scheme=scheme,class_name=c) for c in CLASSES]
        axes[1].bar(np.arange(6)+offset, v, width=.26, color=color, label=label)
    axes[1].set(xticks=np.arange(6), xticklabels=CLASSES, ylabel='Mean upper-lower score width',title='B  Uncertainty retained in the scores')
    axes[1].legend(frameon=False)
    save(fig, folder, '01_five_dimension_scores')
    # 2. AUC comparisons with visible training-seed variation, all/common-known.
    fig, axes = plt.subplots(1, 2, figsize=(11.5,4.5), sharey=True, layout='constrained')
    for ax, cohort, title in zip(axes, ('all','common_complete'), ('A  All 2,400 targets','B  Common fully-known targets')):
        for i, m in enumerate(MODELS):
            vals = [mean(scoring,'error_auroc',model=m,cohort=cohort,scheme=s) for s in ('equal_global','equal_parts')]
            ax.plot(vals,[i-.13,i+.13],color='#ADB5BD',lw=1,zorder=1)
            for y, s, color in [(i-.13,'equal_global',COLORS[2]),(i+.13,'equal_parts',COLORS[4])]:
                seeds = [float(r['error_auroc']) for r in scoring if r['model']==m and r['cohort']==cohort and r['scheme']==s]
                ax.scatter(seeds,[y]*len(seeds),color=color,s=10,alpha=.5)
                ax.scatter(np.mean(seeds),y,color=color,s=35,zorder=3)
        ax.axvline(.5,color='#777777',ls=':',lw=.8)
        ax.set(yticks=np.arange(8),yticklabels=NAMES,xlabel='Class-macro error AUROC',title=title)
    axes[0].invert_yaxis()
    axes[0].scatter([],[],color=COLORS[2],label='Global'); axes[0].scatter([],[],color=COLORS[4],label='Local')
    axes[0].legend(frameon=False,loc='lower left')
    save(fig,folder,'02_scoring_error_auc')
    # 3. Same pool and same selected model panel, new tiers for each scorer.
    fig, axes = plt.subplots(1, 2, figsize=(11.3,4.6), sharey=True, layout='constrained')
    matrices = []
    for scheme in ('equal_global','equal_parts'):
        matrices.append(np.array([[100*mean(tiers,'recall',group='POOL',scheme=scheme,model=m,tier=t) for t in range(1,6)] for m in MODELS]))
    lo,hi = min(v.min() for v in matrices),max(v.max() for v in matrices)
    for ax,z,title in zip(axes,matrices,('A  Global score tiers','B  Local score tiers')):
        im=ax.imshow(z,cmap='Blues',vmin=lo,vmax=hi,aspect='auto',interpolation='nearest')
        ax.set(xticks=range(5),xticklabels=['Q1','Q2','Q3','Q4','Q5'],yticks=range(8),yticklabels=NAMES,title=title,xlabel='Class-specific risk tier')
        for (i,j),v in np.ndenumerate(z):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>(lo+hi)/2 else '#203050')
    fig.colorbar(im,ax=axes,label='Class-macro Recall (%)',shrink=.8)
    save(fig,folder,'03_model_risk_tier_recall')
    # 4. Fixed dictionary, fixed cap; the local field coverage is separate.
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.5),layout='constrained')
    for ax,key,title in zip(axes,('presence_coverage','support_completion','local_presence_coverage'),
                           ('A  Shared-state presence','B  Shared-state support, cap=20','C  Auxiliary local-state presence')):
        vals=[100*float(r[key]) for r in dataset]
        bars=ax.bar([r['group'] for r in dataset],vals,color=COLORS,width=.65)
        ax.set(ylabel='Class-macro coverage (%)',title=title,ylim=(0,105))
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    save(fig,folder,'04_coverage_comparison')
    # 5. Both matched comparisons; never interpret lower F1 alone as a gain.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7),sharey=True,layout='constrained')
    for ax,contrast,color,title in zip(axes,('E2-E1','E4-E3'),(COLORS[2],COLORS[4]),
                                     ('A  Global: coverage - same-quota random','B  Local: coverage - same-quota random')):
        for i,m in enumerate(MODELS):
            vals=[100*float(r['equal_error_breadth_delta']) for r in paired if r['model']==m and r['contrast']==contrast]
            ax.plot([min(vals),max(vals)],[i,i],color=color,lw=2,alpha=.6)
            ax.scatter(vals,[i-.07,i,i+.07],s=16,color=color,alpha=.6)
            ax.scatter(np.mean(vals),i,s=42,color=color,edgecolors='white',zorder=4)
        ax.axvline(0,color='#888888',ls=':',lw=.8)
        ax.set(yticks=range(8),yticklabels=NAMES,xlabel='Equal-error confusion breadth difference (pp)',title=title)
    axes[0].invert_yaxis()
    save(fig,folder,'05_equal_error_breadth')
    # 6. A common E4 test set per model; class-macro within predeclared conditions.
    specs=[('body_visibility','partial','Body: partial'),('head_visibility','partial','Head: partial'),
           ('leg_visibility','one','One leg visible'),('occlusion_level','moderate','Moderate occlusion'),
           ('occlusion_level','heavy','Heavy occlusion'),('truncation_level','moderate','Moderate truncation'),
           ('body_orientation','back','Body: back'),('head_orientation','back','Head: back'),
           ('target_scale','tiny','Tiny target')]
    mat=np.full((8,len(specs)),np.nan); support=[]
    for j,(field,state,label) in enumerate(specs):
        support.append(sum(mean(conditions,'mean_support',group='E4',model=MODELS[0],class_name=c,field=field,state=state) for c in CLASSES
                           if np.isfinite(mean(conditions,'mean_support',group='E4',model=MODELS[0],class_name=c,field=field,state=state))))
        for i,m in enumerate(MODELS):
            per_class=[]
            for c in CLASSES:
                matching=[r for r in conditions if r['group']=='E4' and r['model']==m and r['class_name']==c and r['field']==field and r['state']==state and float(r['mean_support'])>=20-1e-8]
                if matching:per_class.append(np.mean([float(r['conditional_recall']) for r in matching]))
            if per_class:mat[i,j]=100*np.mean(per_class)
    fig,ax=plt.subplots(figsize=(12.2,4.8),layout='constrained')
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E9ECEF')
    im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,vmin=0,vmax=100,aspect='auto',interpolation='nearest')
    ax.set(xticks=range(len(specs)),xticklabels=[f'{s[2]}\nn={n:.0f}' for s,n in zip(specs,support)],
           yticks=range(8),yticklabels=NAMES,title='E4: common visual conditions, class-macro Recall; class-state support >=20')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right',fontsize=8)
    for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:.0f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=8,color='white' if v>60 else '#203050')
    fig.colorbar(im,ax=ax,label='Recall (%)',shrink=.8)
    save(fig,folder,'06_model_visual_condition_recall')
    # Additional compact model-performance view requested in the plan.
    fig,ax=plt.subplots(figsize=(10.7,4.6),layout='constrained')
    for i,(group,color) in enumerate(zip(('E1','E2','E3','E4'),COLORS[1:])):
        values=[100*mean(model,'macro_f1',model=m,group=group) for m in MODELS]
        ax.plot(range(8),values,marker='o',ms=4,color=color,label=group)
    ax.set(xticks=range(8),xticklabels=NAMES,ylabel='Macro-F1 (%)',title='Same model checkpoints on four selected test sets')
    plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    ax.legend(frameon=False,ncol=4);ax.grid(axis='y',alpha=.15)
    save(fig,folder,'07_model_macro_f1')
    return folder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    root=parser.parse_args().input.resolve()
    model,paired,scoring,tiers,conditions,scores,fields,dataset=[rows(root/(name+'.csv')) for name in
        ('model_run_metrics','paired_model_run_metrics','scoring_metrics','tier_model_metrics','condition_metrics','scores','field_scores','dataset_metrics')]
    tests=rows(root/'tests.csv')
    summaries=[]
    for group in ('E0','E1','E2','E3','E4'):
        summaries.append(dict(group=group,macro_f1=mean(model,'macro_f1',group=group),accuracy=mean(model,'accuracy',group=group),errors=mean(model,'errors',group=group)))
    dataset_by={r['group']:r for r in dataset}
    test_by={r['contrast']:r for r in tests}
    coverage_delta=100*(float(dataset_by['E4']['support_completion'])-float(dataset_by['E3']['support_completion']))
    breadth_delta=100*mean(paired,'equal_error_breadth_delta',contrast='E4-E3')
    auc_delta=mean(scoring,'error_auroc',scheme='equal_parts',cohort='all')-mean(scoring,'error_auroc',scheme='equal_global',cohort='all')
    positive=int(test_by['E4-E3']['positive_architectures']); p=float(test_by['E4-E3']['holm_p'])
    with (root/'model_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['model','E1_f1','E2_f1','E3_f1','E4_f1','global_auc','local_auc','global_breadth_delta','local_breadth_delta','local_positive_seeds']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for m in MODELS:
            writer.writerow(dict(model=m,**{g+'_f1':mean(model,'macro_f1',group=g,model=m) for g in ('E1','E2','E3','E4')},
                global_auc=mean(scoring,'error_auroc',model=m,cohort='all',scheme='equal_global'),
                local_auc=mean(scoring,'error_auroc',model=m,cohort='all',scheme='equal_parts'),
                global_breadth_delta=mean(paired,'equal_error_breadth_delta',model=m,contrast='E2-E1'),
                local_breadth_delta=mean(paired,'equal_error_breadth_delta',model=m,contrast='E4-E3'),
                local_positive_seeds=sum(float(r['equal_error_breadth_delta'])>0 for r in paired if r['model']==m and r['contrast']=='E4-E3')))
    folder=make_plots(root,model,paired,scoring,tiers,conditions,scores,fields,dataset)
    svg_audit=[]
    for path in sorted(folder.glob('*.svg')):
        tree=ET.parse(path); texts=sum(e.tag.rsplit('}',1)[-1]=='text' for e in tree.iter())
        assert texts>0 and tree.getroot().get('viewBox')
        svg_audit.append(dict(file=path.name,text_elements=texts,editable_text=True))
    # Companion figure source; all data references are relative to this input directory.
    shutil.copyfile(__file__,folder/'report_coco_fine_coverage_20261015.py')
    conclusions=[
        f'局部评分下，覆盖选样E4相对同风险格配额随机E3的支持覆盖变化为{coverage_delta:+.2f}个百分点。',
        f'等错误数混淆广度变化为{breadth_delta:+.2f}个百分点，{positive}/8架构的种子平均差为正；本轮四项检验统一Holm校正p={p:.6f}。',
        f'局部相对整体评分的完整池类别宏平均错误AUROC差为{auc_delta:+.4f}；方向和分模型结果如下，不将局部细化预设为改进。',
        f"整体评分对照的等错误数广度增量为{100*mean(paired,'equal_error_breadth_delta',contrast='E2-E1'):+.2f}个百分点，8/8架构为正，Holm校正p={float(test_by['E2-E1']['holm_p']):.6f}。当前未观察到局部细化相对整体评分的明确额外收益。"]
    if p<.05 and breadth_delta>0:
        conclusions.append('在当前八架构面板与本轮COCO设置下，混淆广度正向差通过本轮校正检验；不推广为跨数据集结论。')
    else:
        conclusions.append('本轮不能宣称局部评分覆盖选样具有统计显著的混淆广度优势，应以效应大小、模型差异与覆盖结果解释。')
    lines=['# COCO旧特征五维细粒度评分与覆盖选样：实验结果','',
        '本报告对应2026-10-15新方案：COCO2017单源2400个目标、六类各400；每组选择1200个目标、每类200。使用既有本地原生标注和八模型×三种子预测。没有重新训练、没有新增VLM识别、没有合并其他数据源。','',
        '## 一、可直接向老师说明的结论','',*['- '+s for s in conclusions],'',
        '## 二、本轮怎样评分与选样','',
        '五维各20分。局部方案的非车可见性为头部/腿部各50%，朝向为身体/头部各50%；尺度、遮挡、截断沿用旧分级。车辆沿用整体可见性和车身朝向。unknown保留上下界，以中点排序；没有把缺失当确定最高风险。','',
        'E0为无风险约束覆盖；E1/E2分别为整体评分下风险随机/风险覆盖；E3/E4分别为局部评分下风险随机/风险覆盖。两套评分分别在COCO类别内重分Q1—Q5，每层80候选、入选40；随机与覆盖组匹配风险中点与未知区间宽度格配额。随机1000次是选样波动，不是1000个训练模型。','',
        '## 三、选样结果','',
        table(['集合','出现覆盖','支持覆盖τ=20','局部字段出现覆盖','平均Macro-F1','平均错误数'],[
            [d['group'],f"{100*float(d['presence_coverage']):.2f}%",f"{100*float(d['support_completion']):.2f}%",f"{100*float(d['local_presence_coverage']):.2f}%",f"{100*mean(model,'macro_f1',group=d['group']):.2f}%",f"{mean(model,'errors',group=d['group']):.2f}"] for d in dataset]),'',
        '出现覆盖与支持覆盖是不同指标；局部字段覆盖另用预设适用字典，不混入主71状态分母。Macro-F1变化来自测试目标构成变化，不能解释为模型经过训练后性能提升。','',
        '## 四、模型之间的差异','',
        table(['模型','整体评分AUROC','局部评分AUROC','E3 F1','E4 F1','E4−E3广度/百分点','广度正向种子'],[
            [name,f"{mean(scoring,'error_auroc',model=m,cohort='all',scheme='equal_global'):.4f}",f"{mean(scoring,'error_auroc',model=m,cohort='all',scheme='equal_parts'):.4f}",
             f"{100*mean(model,'macro_f1',model=m,group='E3'):.2f}%",f"{100*mean(model,'macro_f1',model=m,group='E4'):.2f}%",f"{100*mean(paired,'equal_error_breadth_delta',model=m,contrast='E4-E3'):+.2f}",
             f"{sum(float(r['equal_error_breadth_delta'])>0 for r in paired if r['model']==m and r['contrast']=='E4-E3')}/3"] for m,name in zip(MODELS,NAMES)]),'',
        'AUROC衡量风险分对错误的区分，先按真实类别计算再平均。广度为控制同模型、种子及类别的错误数后，期望发现的错误预测类别比例；不等于独立语义根因数量。','',
        '## 五、结果图与解读','']
    captions=[('01_five_dimension_scores','五维组成与未知区间','左图为局部评分的五维平均贡献，右图比较两套评分的未知区间宽度；部件细化不等于信息全部已知。'),
        ('02_scoring_error_auc','评分是否更能区分错误','左右分别为全部目标与两套方案共同完整字段目标；小点为三个训练种子，大点为种子均值。共同完整子集结果不是替代主候选池。'),
        ('03_model_risk_tier_recall','不同模型在风险层中的表现','完整COCO候选池按两套分数分别分层，格内为类别宏平均Recall，再对三个种子平均；Q1—Q5意义由相应评分定义。'),
        ('04_coverage_comparison','覆盖选样是否增加状态支持','三图分别展示主状态出现覆盖、主状态支持覆盖、辅助局部字段出现覆盖。三者分开解释，不比较不同分母的百分比大小来声称改进。'),
        ('05_equal_error_breadth','相同错误数下的模型混淆信息','两图分别展示整体/局部评分下的覆盖−随机差。细线和小点为三个种子，大点为种子均值；不是置信区间。'),
        ('06_model_visual_condition_recall','共同视觉条件下的模型差异','使用共同E4集合。只纳入类别—状态支持至少20的条件，再类宏平均；NA表示支持不足或不可评价。列标签n为该状态跨类总支持，不能代替逐类别支持门槛。'),
        ('07_model_macro_f1','同一模型在四种所选测试集上的F1','用于观察测试构成响应；F1越低本身不证明测试方法更好，也不代表模型被修改。')]
    for stem,title,caption in captions:
        lines += [f'### {title}','',f'![{title}](figures/{stem}.png)','',caption,'']
    lines += ['## 六、统计与边界','',
        table(['本轮预先登记比较','平均差','正向架构','双侧精确p','Holm校正p'],[
            [r['contrast'],f"{float(r['mean_delta']):+.6f}",r['positive_architectures']+'/8',f"{float(r['signflip_two_sided_p']):.6f}",f"{float(r['holm_p']):.6f}"] for r in tests]),'',
        '本轮单独登记四项新COCO评分—选样探索比较，未重算或缩小旧7800合并池18项检验家族；既有池并非从未分析的新盲测数据。架构内先平均三个种子，再以八架构做双侧精确符号翻转；只描述这一模型面板。随机选样分位数是选样分布，不能当作模型总体置信区间。','',
        '条件报告中随机组采用全部随机重复的正确支持总数/支持总数，支持量展示每套测试集平均值；不是逐重复条件Recall的简单平均，也不是把同一目标重复出现算作独立新样本。无错误配对的稀释结果按0计入，并另报可评价比例。自动状态未经本轮新增人工语义核验，结果依赖已有状态标签。','',
        '本轮结果不自动触发调权重或新增数据源。下一步是否在VOC2007、Objects365复核，应基于覆盖收益、评分有效性及分模型诊断结果共同决定。','',
        '## 七、原始证据与复现','',
        '主要输入：protocol.json、manifest.csv、native_features.jsonl、input_hashes.json；评分：field_scores.csv、scores.csv；选样：selection.npz、risk_cell_quotas.csv、greedy_trace.csv；评价：model_run_metrics.csv、paired_model_run_metrics.csv、scoring_metrics.csv、condition_metrics.csv、tier_model_metrics.csv；检查：preparation_audit.json、selection_audit.json、prediction_cache_audit.json、evaluation_audit.json。','',
        f'复现入口：`{ROOT}/scripts/run_coco_fine_coverage_experiment_20261015.py`。全量运行需使用新输出目录，防止覆盖已完成结果。图表由`report_coco_fine_coverage_20261015.py --input {root}`重新生成。','']
    (root/'report_zh.md').write_text('\n'.join(lines))
    claim=['# 本轮结论—证据映射','',table(['主张','证据','状态与边界'],[
        ['旧COCO2400目标完成双评分及五组选择','preparation_audit.json; selection_audit.json','实现核对通过；不是新VLM语义准确率证明'],
        [f'局部覆盖选样支持覆盖变化{coverage_delta:+.2f}个百分点','dataset_metrics.csv: E4-E3 support_completion','本池本预算；不推导模型训练提升'],
        [f'等错误数混淆广度变化{breadth_delta:+.2f}个百分点','paired_model_run_metrics.csv: E4-E3; tests.csv','限当前模型组；是否显著以Holm4为准'],
        [f'局部−整体AUROC差{auc_delta:+.4f}','scoring_metrics.csv: all cohort, class macro then seeds/architectures','观察到的评分差，不能宣称局部必然更优'],
        ['跨模型条件差异','condition_metrics.csv; tier_model_metrics.csv','描述性比较；支持门槛20，重叠状态不独立']]),'']
    (root/'claim_evidence.md').write_text('\n'.join(claim))
    audit=dict(status='pass',figures=svg_audit,result_source='current COCO run CSVs, no legacy merged results substituted',
               local_support_delta_pp=coverage_delta,local_breadth_delta_pp=breadth_delta,auc_delta=auc_delta)
    (root/'report_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
