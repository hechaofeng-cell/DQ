#!/usr/bin/env python3
"""Report and editable figures from cached rank/reference validation tables."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_ranking_attribute_validation import OUT, ROOT, MODELS, CONFIGS

NAMES = dict(zip(MODELS, ('ResNet-18', 'ResNet-50', 'MobileNetV3-S', 'MobileNetV3-L',
                         'DenseNet-121', 'EfficientNet-B0', 'ConvNeXt-Tiny', 'Swin-T')))


def main():
    plt.rcParams.update({'svg.fonttype':'none', 'font.family':'DejaVu Sans', 'font.size':8,
        'axes.linewidth':.8, 'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    folder = OUT / 'figures'
    r = pd.read_csv(OUT / 'tables/model_ranks.csv')
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), layout='constrained')
    for j, (pool, title) in enumerate([('B','COCO'), ('C','VOC2007'), ('D','Objects365')]):
        subset = r[r.pool == pool]
        matrix = np.array([[subset[(subset.model == model) & (subset.source == s) & (subset.unknown_mode == m)].iloc[0]['rank']
                            for s, m in CONFIGS] for model in MODELS])
        frequency = np.array([[subset[(subset.model == model) & (subset.source == s) & (subset.unknown_mode == m)].iloc[0].top2_frequency
                              for s, m in CONFIGS] for model in MODELS])
        for row, (values, cmap, vmin, vmax) in enumerate([(matrix,'Blues_r',1,8), (frequency,'Blues',0,1)]):
            ax = axes[row, j]
            ax.pcolormesh(np.arange(7)-.5, np.arange(9)-.5, values,
                          cmap=cmap, vmin=vmin, vmax=vmax, shading='flat', rasterized=False)
            ax.set(xlim=(-.5,5.5), ylim=(7.5,-.5))
            for a in range(8):
                for b in range(6):
                    label = f'{values[a,b]:.0f}' if row == 0 else f'{100*values[a,b]:.0f}'
                    color = 'white' if (row == 0 and values[a,b] <= 3) or (row == 1 and values[a,b] >= .65) else '#272727'
                    ax.text(b, a, label, ha='center', va='center', fontsize=7, color=color)
            ax.set(xticks=range(6), xticklabels=['L-low','L-mid','L-high','O-low','O-mid','O-high'],
                   yticks=range(8), yticklabels=[NAMES[m] for m in MODELS] if j == 0 else [],
                   title=f'{pool}: {title}' + (' / mean rank' if row == 0 else ' / top-two frequency (%)'))
            ax.tick_params(axis='x', rotation=35, length=0)
            ax.tick_params(axis='y', length=0)
            ax.text(-.09,1.03,chr(ord('a')+row*3+j),transform=ax.transAxes,fontweight='bold')
    fig.savefig(folder / 'rank_sensitivity.svg', bbox_inches='tight')
    fig.savefig(folder / 'rank_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    a = pd.read_csv(OUT / 'tables/attribute_metrics.csv')
    keys = [(c,'class') for c in ('dog','cat','horse','sheep','person','car')] + [('all','target_weighted'),('all','class_equal')]
    labels = ['dog','cat','horse','sheep','person','car','Pooled targets','Class mean']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), layout='constrained')
    for j, (field, title) in enumerate([('occlusion_level','Occlusion'),('truncation_level','Truncation')]):
        ax = axes[j]
        for source, shift, marker, color in [('local',-.12,'o','#3775BA'),('online',.12,'s','#9A4D8E')]:
            values = [a[(a.source==source)&(a.field==field)&(a.class_name==c)&(a.aggregation==aggregation)].iloc[0] for c, aggregation in keys]
            center = np.array([v.f1 for v in values])
            low = np.array([v.f1_ci95_low for v in values]); high = np.array([v.f1_ci95_high for v in values])
            ax.errorbar(center, np.arange(8)+shift, xerr=[center-low,high-center], fmt=marker,
                        color=color, ms=4, capsize=2, label=source)
        ax.set(title=title + ' / binary reference', xlabel='Positive-state F1',
               yticks=range(8), yticklabels=labels, xlim=(.3,.9), ylim=(7.6,-.6))
        ax.legend(fontsize=7,loc='upper left')
        ax.axhline(5.5,color='.85',lw=.7)
        ax.text(-.08,1.03,chr(ord('a')+j),transform=ax.transAxes,fontweight='bold')
    fig.savefig(folder / 'attribute_reference.svg', bbox_inches='tight')
    fig.savefig(folder / 'attribute_reference.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(Path(__file__), folder / Path(__file__).name)

    summary = pd.read_csv(OUT / 'tables/ranking_summary.csv')
    pair = pd.read_csv(OUT / 'tables/configuration_pairs.csv')
    model_pairs = pd.read_csv(OUT / 'tables/model_pair_differences.csv')
    lines = ['# 排名敏感性与官方属性参考核验','',
        '本轮是事后缓存分析：无新增训练、分类推理、VLM请求或人工复核。固定8架构×3种子，三个确认池5400目标。',
        '排名依据为各配置完整Q5上的架构三种子Macro-F1均值；本地/线上×unknown下界/中点/上界构成六配置。','',
        '## 1 排名对配置的敏感性','',
        '| 池 | 每类Q5 | 15对配置tau范围 | 前两名集合相同 | 最大名次变化 | 有效重采样 |',
        '|---|---:|---|---|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.pool} | {row.q5_per_class} | {row.tau_min:.3f}–{row.tau_max:.3f} | {row.same_top2_pairs}/15 | {row.max_rank_change:.0f} | {row.bootstrap_valid}/2000 |')
    lines += ['', '所有18种池/配置的点估计第一名为ConvNeXt-Tiny，第二名为Swin-T。完整排序不是完全一致，D最大变化为两名。', '',
        '| 池 | 模型 | 六配置进入前两名频率范围 | 六配置名次95%区间外包络 |', '|---|---|---|---|']
    for pool in ('B','C','D'):
        for model in ('convnext_tiny','swin_t'):
            frame = r[(r.pool==pool)&(r.model==model)]
            lines.append(f'| {pool} | {NAMES[model]} | {100*frame.top2_frequency.min():.2f}%–{100*frame.top2_frequency.max():.2f}% | {frame.rank_ci95_low.min():.0f}–{frame.rank_ci95_high.max():.0f} |')
    lines += ['', '范围是六配置各自结果的最小/最大，不是合并后的置信区间。进入前两名频率是固定模型与风险标签下的源图重采样频率，不是真实世界最佳模型概率。',
        '', '| 池 | 两种中点排序tau | tau源图区间 | Q5 Jaccard |', '|---|---:|---|---:|']
    for pool in ('B','C','D'):
        row = pair[(pair.pool==pool)&(pair.source_left=='local')&(pair.mode_left=='midpoint')&(pair.source_right=='online')&(pair.mode_right=='midpoint')].iloc[0]
        lines.append(f'| {pool} | {row.kendall_tau:.3f} | [{row.tau_ci95_low:.3f}, {row.tau_ci95_high:.3f}] | {row.q5_jaccard:.3f} |')
    lines += ['', '两种来源的高风险目标集合有差异，但平均模型排序较接近。上一轮严格全字段一致子集仍不可评价；本分析检验完整池的来源与unknown规则敏感性，二者问题不同。',
        'B/C中ConvNeXt-Tiny与Swin-T六配置的配对F1差区间全部包含零。D六配置区间均为正，但仅为描述性区间，未作新的多重比较确认。D中Swin-T前两名频率明显较低，不宜将其点估计第二名写成可靠独占第二。',
        '', '![排名与前两名频率](figures/rank_sensitivity.svg)', '',
        '## 2 官方属性参考','',
        'A的2289个Open Images目标具有框级参考：dog/cat/horse/person/car各400，sheep289。none归并0，其他已知严重度归并1，unknown单列。以下为非unknown目标加权指标，字段正类是有遮挡/有截断，非对象分类F1。', '',
        '| 字段 | 来源 | 有效/计划 | 一致率 | Precision | Recall | F1 | F1源图区间 | Specificity | Kappa |',
        '|---|---|---|---:|---:|---:|---:|---|---:|---:|']
    for field in ('occlusion_level','truncation_level'):
        for source in ('local','online'):
            row = a[(a.field==field)&(a.source==source)&(a.aggregation=='target_weighted')].iloc[0]
            lines.append(f'| {field} | {source} | {row.known:.0f}/{row.planned:.0f} | {100*row.agreement:.2f}% | {100*row.precision:.2f}% | {100*row.recall:.2f}% | {100*row.f1:.2f}% | [{100*row.f1_ci95_low:.2f}, {100*row.f1_ci95_high:.2f}]% | {100*row.specificity:.2f}% | {row.kappa:.3f} |')
    lines += ['', '遮挡二值参考的Recall高于Precision，且Specificity偏低，表明相对官方参考存在较多阳性误报。线上截断参考指标较好，但不能外推到其他字段或完整严重度真值。逐类别混淆矩阵、Wilson一致率区间、Balanced Accuracy与类别等权汇总均保留。',
        '', '![官方属性参考](figures/attribute_reference.svg)', '',
        '## 3 尺度与输入核验','',
        'A/B/C/D共7800目标的清单面积与官方坐标复算尺度标签全部一致，裁到图像边界后也无尺度状态变化。D四个边界舍入框超界最多0.0006714像素，符合原来源资格的0.001像素容差。首次严格边界断言失败及独立恢复脚本/哈希保留，不修改原数据或冻结分析代码。',
        '', '## 4 解释边界与复算','',
        '本轮支持配置敏感性下的模型能力比较，并提供部分二值属性参考核验；不证明所有自动视觉状态准确、严格一致子集排名稳健或状态是失败根因。Coverage全面优于强基线不是本研究成立条件。',
        '2000次类别内完整源图重采样，所有模型及六种配置共享抽样，保留重叠依赖；原Q5身份固定，区间条件于24权重、标签和候选池。训练变动另见seed_ranks.csv。',
        '分析协议：[方案]('+str(ROOT/'docs/ranking_attribute_validation_protocol_20261005.md')+')；复算scripts/resume_ranking_attribute_validation.py，绘图/报告scripts/plot_ranking_attribute_validation.py。',
        '全部排名：tables/model_ranks.csv；45对配置：tables/configuration_pairs.csv；504对模型差：tables/model_pair_differences.csv；官方属性：tables/attribute_metrics.csv与attribute_samples.csv；尺度：tables/scale_details.csv。',
        '完整性：analysis_audit.json；冻结输入：input_freeze.sha256.json；技术修订：technical_amendment.sha256.json；首次失败：initial_failure.json。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
