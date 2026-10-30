#!/usr/bin/env python3
"""Readable reports and SVG figures for fixed-budget risk-mixture responses."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_controlled_risk_distribution import BASE,ROOT,MODELS,CLASSES,METRICS

NAMES=dict(zip(MODELS,('ResNet-18','ResNet-50','MobileNetV3-S','MobileNetV3-L',
                     'DenseNet-121','EfficientNet-B0','ConvNeXt-Tiny','Swin-T')))
COLORS=['#767676','#4D4D4D','#9E9AC8','#756BB1','#42949E','#D49336','#3775BA','#9A4D8E']
POOL_NAMES={'B':'COCO','C':'VOC2007','D':'Objects365'}


def layout(ylabel):
    fig,axes=plt.subplots(3,2,figsize=(9.3,7.5),layout='constrained')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            ax=axes[i,j]
            ax.set(title=f'{pool}: {POOL_NAMES[pool]} / {source}',xlabel='Q5 targets (%)',ylabel=ylabel,
                   xlim=(0,100),xticks=[0,20,40,60,80,100])
            ax.text(-.1,1.04,chr(ord('a')+i*2+j),transform=ax.transAxes,fontweight='bold')
    return fig,axes


def save(fig,name):
    fig.savefig(BASE/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(BASE/'figures'/f'{name}.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


def series(ax,frame,key,label,color,band=True,style='-'):
    ordered=frame.sort_values('high_risk_percent');x=ordered.high_risk_percent.to_numpy()
    ax.plot(x,100*ordered[key+'_mean'],style,color=color,lw=1.4,marker='.',ms=2.1,label=label)
    if band:
        ax.fill_between(x,100*ordered[key+'_sampling_low'],100*ordered[key+'_sampling_high'],color=color,alpha=.09,lw=0)


def main():
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,
        'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    panel=pd.read_csv(BASE/'tables/panel_summary.csv');model=pd.read_csv(BASE/'tables/model_summary.csv')
    contrast=pd.read_csv(BASE/'tables/distribution_contrasts.csv')
    common=panel[panel.total==120];common_models=model[model.total==120]
    # Class-panel ranges must average models inside each shared draw before taking percentiles.
    class_panel=[];class_contrasts=[]
    for file in sorted((BASE/'raw').glob('*.npz')):
        pool,source,total=file.stem.split('_');total=int(total[1:])
        with np.load(file) as z:
            values=z['class_recall'].astype(float).mean(2);proportions=z['proportions']
        for p,ratio in enumerate(proportions):
            for c,name in enumerate(CLASSES):
                v=values[p,:,c]
                class_panel.append(dict(pool=pool,source=source,total=total,high_risk_percent=int(ratio),class_name=name,
                    recall_mean=float(v.mean()),recall_sampling_low=float(np.percentile(v,2.5)),recall_sampling_high=float(np.percentile(v,97.5))))
        for c,name in enumerate(CLASSES):
            delta=values[-1,:,c]-values[0,:,c]
            class_contrasts.append(dict(pool=pool,source=source,total=total,class_name=name,
                recall_at0=float(values[0,:,c].mean()),recall_at100=float(values[-1,:,c].mean()),
                recall_delta=float(delta.mean()),recall_sampling_low=float(np.percentile(delta,2.5)),
                recall_sampling_high=float(np.percentile(delta,97.5))))
    classes=pd.DataFrame(class_panel);classes.to_csv(BASE/'tables/class_panel.csv',index=False)
    pd.DataFrame(class_contrasts).to_csv(BASE/'tables/class_panel_contrasts.csv',index=False)

    fig,axes=layout('Classification score (%)')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            f=common[(common.pool==pool)&(common.source==source)];ax=axes[i,j]
            series(ax,f,'macro_f1','Macro-F1','#3775BA')
            series(ax,f,'macro_precision','Macro-Precision','#9A4D8E',False,'--')
            series(ax,f,'macro_recall','Accuracy / Macro-Recall','#42949E',False,':')
            ax.set_ylim(35,100)
            if i==0 and j==0:ax.legend(fontsize=7,loc='lower left')
    save(fig,'performance_response')

    fig,axes=layout('Macro-F1 (%)')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            f=common_models[(common_models.pool==pool)&(common_models.source==source)]
            for k,m in enumerate(MODELS):series(axes[i,j],f[f.model==m],'macro_f1',NAMES[m],COLORS[k],True)
            axes[i,j].set_ylim(30,100)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7)
    save(fig,'model_f1_response')

    fig,axes=layout('Class Recall (%)')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            f=classes[(classes.total==120)&(classes.pool==pool)&(classes.source==source)]
            for k,c in enumerate(CLASSES):series(axes[i,j],f[f.class_name==c],'recall',c,COLORS[k+2],True)
            axes[i,j].set_ylim(0,100)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=7)
    save(fig,'class_recall_response')

    for name,keys,ylabel in [
        ('failure_breadth_response',[('failed_state_breadth','Failed states','#3775BA'),
            ('confusion_breadth','Confusion types','#9A4D8E'),('cluster_breadth','Error clusters','#42949E')],'Raw breadth (%)'),
        ('equal_error_breadth_response',[('equal_error_confusion_delta','Confusion types','#9A4D8E'),
            ('equal_error_cluster_delta','Error clusters','#42949E')],'Equal-error breadth delta (pp)')]:
        fig,axes=layout(ylabel)
        for i,pool in enumerate(('B','C','D')):
            for j,source in enumerate(('local','online')):
                f=common[(common.pool==pool)&(common.source==source)];ax=axes[i,j]
                for key,label,color in keys:series(ax,f,key,label,color,True)
                if name.startswith('equal'):
                    ax.axhline(0,color='.7',lw=.7);ax.axvline(20,color='.8',lw=.7,ls=':')
                else:ax.set_ylim(0,100)
                if i==0 and j==0:ax.legend(fontsize=7,loc='upper left')
        save(fig,name)

    fig,axes=layout('Model')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            f=common_models[(common_models.pool==pool)&(common_models.source==source)]
            matrix=f.pivot(index='model',columns='high_risk_percent',values='rank').loc[list(MODELS)].to_numpy()
            ax=axes[i,j]
            im=ax.pcolormesh(np.arange(22)*5-2.5,np.arange(9)-.5,matrix,cmap='Blues_r',vmin=1,vmax=8,shading='flat',rasterized=False)
            ax.set(ylim=(7.5,-.5),xlim=(-2.5,102.5),yticks=range(8),yticklabels=[NAMES[m] for m in MODELS] if j==0 else [],ylabel='')
    cb=fig.colorbar(im,ax=axes,location='bottom',shrink=.65,ticks=range(1,9),pad=.03)
    cb.solids.set_rasterized(False)
    cb.set_label('Mean-score rank (1 = highest F1)')
    save(fig,'rank_response')

    fig,axes=plt.subplots(2,2,figsize=(8,5.2),layout='constrained')
    for i,pool in enumerate(('B','D')):
        for j,source in enumerate(('local','online')):
            ax=axes[i,j]
            for total,color in [(120,'#3775BA'),(480,'#9A4D8E')]:
                f=panel[(panel.pool==pool)&(panel.source==source)&(panel.total==total)]
                series(ax,f,'macro_f1',f'N={total}',color)
            ax.set(title=f'{pool}: {POOL_NAMES[pool]} / {source}',xlabel='Q5 targets (%)',ylabel='Panel Macro-F1 (%)',xlim=(0,100),ylim=(35,100))
            if i==0 and j==0:ax.legend(fontsize=7)
            ax.text(-.1,1.04,chr(ord('a')+i*2+j),transform=ax.transAxes,fontweight='bold')
    save(fig,'budget_sensitivity')

    rank_summary=[]
    for (pool,source,total),g in model.groupby(['pool','source','total']):
        ordered=g.pivot(index='model',columns='high_risk_percent',values='rank')
        first=g[g.high_risk_percent==0].sort_values('rank');last=g[g.high_risk_percent==100].sort_values('rank')
        row=dict(pool=pool,source=source,total=int(total),top1_at0=first.model.iloc[0],top1_at100=last.model.iloc[0],
            top2_at0=first.model.iloc[1],top2_at100=last.model.iloc[1],
            min_tau_vs20=float(g.point_kendall_vs20.min()),max_rank_range=float((ordered.max(axis=1)-ordered.min(axis=1)).max()),
            models_with_rank_change=int(((ordered.max(axis=1)-ordered.min(axis=1))>0).sum()))
        rank_summary.append(row)
    pd.DataFrame(rank_summary).to_csv(BASE/'tables/rank_response_summary.csv',index=False)
    lines=['# 固定预算风险分布响应实验：完整结果','',
        '实验完成：三候选池分别分析，8架构×3种子、双VLM中点标签；主实验120图（每类20），高风险占比0%–100%，每5个百分点一个真实点，1000次重复。B/D另有480图规模敏感性。',
        '高风险指原类别内Q5，其他目标从原Q1–Q4无放回抽取。总量、六类比例、模型及标签固定；相邻比例通过替换目标改变构成，不是累计增加测试数量。','',
        '六类为dog、cat、horse、sheep、person、car；person是人体目标而非独立人脸类别。模型为ResNet-18、ResNet-50、MobileNetV3-Small/Large、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny、Swin-T，每个架构使用既有三个种子的预测。local/online指风险分层和状态标注来源，不是两组分类模型。',
        '风险分数H沿用目标尺度、身体可见程度、遮挡、截断、朝向及未知状态的中点规则，按池内各类别等分五层；Q5是该池该类的相对高风险层，不代表跨池相同绝对难度。本轮复用冻结预测、状态标签与协议，未重新训练、推理或调用VLM。1000次抽样来自同一有限候选池，不是1000个独立数据集。','',
        '## 1 固定预算下的性能响应','',
        '以下先平均三个种子，再等权平均八架构。区间为配对0%→100%差值在1000次重复抽样中的2.5/97.5百分位，单位百分点；它不是均值置信区间或部署总体区间。','',
        '| 池 | 来源 | 0% Q5 F1（%） | 100% Q5 F1（%） | 差值（pp） | 95%重复抽样区间（pp） |',
        '|---|---|---:|---:|---:|---|']
    for pool,source in [(p,s) for p in ('B','C','D') for s in ('local','online')]:
        f=common[(common.pool==pool)&(common.source==source)].set_index('high_risk_percent')
        r=contrast[(contrast.pool==pool)&(contrast.source==source)&(contrast.total==120)&(contrast.model=='panel_mean')].iloc[0]
        lines.append(f'| {pool} | {source} | {100*f.loc[0].macro_f1_mean:.2f} | {100*f.loc[100].macro_f1_mean:.2f} | {100*r.macro_f1_delta:+.2f} | [{100*r.macro_f1_sampling_low:+.2f}, {100*r.macro_f1_sampling_high:+.2f}] |')
    lines+=['','六组主面板均出现终点下降。0%不是只含Q1，而是从Q1–Q4抽样，所以差值不同于上一轮Q5−Q1。改变的是评价分布，模型参数及每图预测没有改变。原始曲线全部21点保留，不强制逐点单调。',
        '', '![总体分类指标；阴影仅为Macro-F1的95%重复抽样区间，Accuracy与Macro-Recall在类别平衡下相等。](figures/performance_response.png)',
        '', '![八架构Macro-F1；每线先平均三个种子，阴影为固定池内重复抽样范围。](figures/model_f1_response.png)',
        '', '## 2 类别Recall与排名','',
        '类别Recall不能由总体F1代替。下图每类别先在固定24运行内平均，再计算抽样区间；不同类别对风险占比的响应幅度各异。完整模型/类别结果保留tables/class_recall.csv，逐类配对终点差及区间保留tables/class_panel_contrasts.csv。',
        '并非每一类Recall都下降。例如B/local的car从95.18%变为97.20%，终点差+2.02个百分点，95%重复抽样区间为[-4.38,+10.42]个百分点。因此自动规则定义的高风险与某个模型或类别的实际识别难度不能等同，也不能从总体下降推断所有类别同步下降。',
        '', '![六类别Recall响应；阴影为同一重复抽样中的模型面板均值区间。](figures/class_recall_response.png)',
        '', '| 池 | 来源 | 0%平均第一 | 100%平均第一 | 最小tau（相对20%） | 最大名次跨度 |',
        '|---|---|---|---|---:|---:|']
    for row in rank_summary:
        if row['total']!=120:continue
        lines.append(f"| {row['pool']} | {row['source']} | {NAMES[row['top1_at0']]} | {NAMES[row['top1_at100']]} | {row['min_tau_vs20']:.3f} | {row['max_rank_range']:.0f} |")
    lines+=['','当前六组主实验的平均前两名在所有比例点均为ConvNeXt-Tiny和Swin-T，但中间排名存在变化；不能将平均第二名等同于抽样中始终第二。各模型95%名次范围与进入前两名频率在model_summary.csv，保留并列席位分配。',
        '', '![各比例的平均分数排名；21列对应21个真实比例点，深色代表较靠前。](figures/rank_response.png)',
        '', '## 3 原始失败广度与等错误数控制','',
        '失败状态广度要求所选状态支持至少3且含错误；分母为每类固定状态宇宙。混淆类型与ViT错误簇提供不依赖状态枚举的代理广度，但错误簇不等于根因。Raw指标随错误数量改变，另以同一重复的20%Q5样本按真类别匹配错误数计算稀释期望差。',
        '', '| 池 | 来源 | 0%失败状态广度（%） | 100%失败状态广度（%） | 100%混淆等错误数差（pp） | 100%簇等错误数差（pp） |',
        '|---|---|---:|---:|---:|---:|']
    for pool,source in [(p,s) for p in ('B','C','D') for s in ('local','online')]:
        f=common[(common.pool==pool)&(common.source==source)].set_index('high_risk_percent');r=f.loc[100]
        lines.append(f'| {pool} | {source} | {100*f.loc[0].failed_state_breadth_mean:.2f} | {100*r.failed_state_breadth_mean:.2f} | {100*r.equal_error_confusion_delta_mean:+.2f} | {100*r.equal_error_cluster_delta_mean:+.2f} |')
    lines+=['','原始失败状态广度终点增加与等错误数广度收益不一致。更高风险比例能暴露更多错误，却可能使错误更集中于已有混淆类型或表征簇；此解释与结果相容，不证明错误根因。控制后的收益不普遍为正，不把Raw广度增加称为独立诊断优势。',
        '', '![原始失败广度；三种分母与定义不同，应分别解释。](figures/failure_breadth_response.png)',
        '', '![相对20%高风险基准的逐类等错误数广度差；20%处按定义为0，阴影为重复抽样范围。](figures/equal_error_breadth_response.png)',
        '', '## 4 规模敏感性与覆盖','',
        '| 池 | 来源 | N=120 F1差（pp） | N=480 F1差（pp） | N=480重复区间（pp） |',
        '|---|---|---:|---:|---|']
    for pool,source in [(p,s) for p in ('B','D') for s in ('local','online')]:
        f=contrast[(contrast.pool==pool)&(contrast.source==source)&(contrast.model=='panel_mean')].set_index('total')
        a,b=f.loc[120],f.loc[480]
        lines.append(f'| {pool} | {source} | {100*a.macro_f1_delta:+.2f} | {100*b.macro_f1_delta:+.2f} | [{100*b.macro_f1_sampling_low:+.2f}, {100*b.macro_f1_sampling_high:+.2f}] |')
    lines+=['','扩大每比例预算后，面板终点差的方向保持，选样波动范围通常收窄。C在100%Q5、B/D在480图100%Q5时使用完整Q5，抽样区间退化仅表示当前有限集合无选样变动。',
        '', '![120与480固定预算响应；仅B/D拥有共同可行的480图预算。](figures/budget_sensitivity.png)',
        '', '频次覆盖cap20、状态存在覆盖、平均H与可评价状态比例同时保留在panel_summary.csv。本实验替换样本，覆盖不必单调增长。主预算每类20图时，单状态支持不超过cap20，逐类频次覆盖与已知状态激活次数总量成正比，不能用它代替状态比例或均匀程度的变化。',
        '', '## 5 可支持的结论与范围','',
        '在固定预算、类别比例、预测与自动状态协议条件下，提高池内Q5占比对应更低的测得性能，下降幅度具有来源和类别差异。前列平均模型排序相对稳定，部分中间名次随测试构成改变。原始失败广度增加不等于等错误数后的广度增益。',
        '这些是对有限已标注候选池的受控混合测试结果，不证明所有视觉状态准确、单字段导致失败或自然部署分布的普遍规律。采样规则在本轮新增结果前固定，但本研究已有历史结果，因此属于事后补充实验。',
        '', '## 6 复算与产物','',
        '[完整协议]('+str(ROOT/'docs/controlled_risk_distribution_protocol_20261005.md')+')。运行scripts/run_controlled_risk_distribution.py，绘图报告scripts/report_controlled_risk_distribution.py；首次smoke独立保存在smoke/。',
        '原始张量raw/*.npz；每次排序前缀sampling/*.npz；固定ViT簇clusters/*.npz；逐运行run_summary.csv；模型指标/排序区间model_summary.csv；类别class_recall.csv/class_panel.csv；面板panel_summary.csv；配对终点distribution_contrasts.csv。',
        '21个点均来自真实统计，SVG保留文字与矢量曲线并附绘图源代码。日志run.log与状态progress.json，冻结清单input_freeze.sha256.json，统计核验analysis_audit.json。']
    lines+=['','可编辑矢量图：'+ '；'.join(f'[{name}](figures/{name}.svg)' for name in (
        'performance_response','model_f1_response','class_recall_response','rank_response',
        'failure_breadth_response','equal_error_breadth_response','budget_sensitivity'))+'。',
        '', '[逐模型指标与排名区间](tables/model_summary.csv)；[逐模型各类Recall](tables/class_recall.csv)；[逐类配对终点差](tables/class_panel_contrasts.csv)；[面板汇总](tables/panel_summary.csv)；[模型配对终点差](tables/distribution_contrasts.csv)。']
    (BASE/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ('performance_response','model_f1_response','class_recall_response','rank_response',
                 'failure_breadth_response','equal_error_breadth_response','budget_sensitivity'):
        shutil.copy2(Path(__file__),BASE/'figures'/f'{name}.py')


if __name__=='__main__':main()
