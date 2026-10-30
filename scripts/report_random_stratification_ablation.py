#!/usr/bin/env python3
"""Create editable controls figures and integrate verified evidence into a new paper."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_random_stratification_ablation import BASE,ROOT,VARIANTS

SOURCE=ROOT/'docs/visual_state_testing_paper.md'
PAPER=ROOT/'docs/visual_state_testing_paper_controls_20261006.md'
POOLS={'B':'COCO','C':'VOC2007','D':'Objects365'}
SOURCES={'local':'本地','online':'线上'}
LABELS={'full':'完整H','drop_size':'删除尺度','drop_body_visibility':'删除可见性',
        'drop_occlusion_level':'删除遮挡','drop_truncation_level':'删除截断',
        'drop_body_orientation':'删除朝向','equal_normalized':'等权归一化'}
PLOT_LABELS=['Full H','Without scale','Without visibility','Without occlusion','Without truncation','Without orientation','Equal-normalized']


def table(caption,headers,rows):
    return '\n'.join([caption,'','| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
        *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def main():
    r=pd.read_csv(BASE/'tables/random_summary.csv')
    a=pd.read_csv(BASE/'tables/ablation_summary.csv')
    m=pd.read_csv(BASE/'tables/membership_summary.csv')
    risk_supported=bool((r.holm_six_p<.05).all())
    negatives=bool((a[a.variant!='full'].negative_runs==24).all())
    assert risk_supported and negatives,'Observed outcome changed; write a bounded interpretation before integrating'
    random_rows=[]
    for v in r.itertuples():
        random_rows.append([v.pool,SOURCES[v.source],f'{100*v.observed_delta:+.2f}',f'{100*v.random_mean:+.2f}',
            f'[{100*v.random_sampling_low:+.2f}, {100*v.random_sampling_high:+.2f}]',str(v.extreme_count),f'{v.holm_six_p:.5f}'])
    rt=table('**表26 类别内随机五等分对照。差值为八架构/三种子面板的Q5−Q1 Macro-F1，单位百分点；随机范围为10000次划层的百分位。**',
        ['池','来源','风险分层差','随机均值','随机95%范围','极端次数','六项Holm p'],random_rows)
    at=table('**表27 五维删维与等权变体重新划层后的Q5−Q1 Macro-F1差，单位百分点。每格平均24个运行，所有变体保留。**',
        ['评分','B本地','B线上','C本地','C线上','D本地','D线上'],
        [[LABELS[v],*[f"{100*a[(a.pool==p)&(a.source==s)&(a.variant==v)].mean_delta_auc.iloc[0]:+.2f}"
            for p in ('B','C','D') for s in ('local','online')]] for v in VARIANTS])
    membership_rows=[]
    for pool in ('B','C','D'):
        for source in ('local','online'):
            sub=a[(a.pool==pool)&(a.source==source)].set_index('variant')
            member=m[(m.pool==pool)&(m.source==source)&(m.variant=='drop_occlusion_level')&(m.tier==5)].iloc[0]
            membership_rows.append([pool,SOURCES[source],f'{100*sub.loc["full"].mean_delta_auc:+.2f}',
                f'{100*sub.loc["drop_occlusion_level"].mean_delta_auc:+.2f}',
                f'{100*sub.loc["drop_occlusion_level"].change_vs_full:+.2f}',f'{member.jaccard:.3f}',f'{100*member.retention:.2f}'])
    mt=table('**表28 删除遮挡后的差值变化及原Q5成员重合。变化为删维差减完整H差；成员指标先逐类别计算再平均。**',
        ['池','来源','完整H差','删遮挡差','差值变化（pp）','Q5 Jaccard','Q5保留率（%）'],membership_rows)
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,
        'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(3,2,figsize=(8.5,7.2),layout='constrained')
    for i,pool in enumerate(('B','C','D')):
        with np.load(BASE/'random_strata'/f'{pool}_metrics.npz') as z:curves=z['panel_curve']
        for j,source in enumerate(('local','online')):
            with np.load(BASE/'ablation'/f'{pool}_{source}.npz') as z:observed=z['f1'][0].mean(0)
            ax=axes[i,j];x=np.arange(1,6)
            ax.fill_between(x,100*np.percentile(curves,2.5,axis=0),100*np.percentile(curves,97.5,axis=0),color='#767676',alpha=.18,lw=0,label='Random 95% range')
            ax.plot(x,100*curves.mean(0),color='#767676',lw=1.1,ls='--',label='Random mean')
            ax.plot(x,100*observed,color='#3775BA',lw=1.6,marker='o',ms=3,label='Risk strata')
            ax.set(title=f'{pool}: {POOLS[pool]} / {source}',xlabel='Within-class quintile',ylabel='Panel Macro-F1 (%)',xticks=x,ylim=(35,100))
            ax.text(-.1,1.04,chr(97+i*2+j),transform=ax.transAxes,fontweight='bold')
            if i==0 and j==0:ax.legend(fontsize=7,loc='lower left')
    save(fig,'risk_vs_random_strata')
    fig,axes=plt.subplots(3,2,figsize=(9,7.2),layout='constrained')
    for i,pool in enumerate(('B','C','D')):
        for j,source in enumerate(('local','online')):
            data=a[(a.pool==pool)&(a.source==source)].set_index('variant').loc[VARIANTS]
            ax=axes[i,j];mean=100*data.mean_delta_auc.to_numpy()
            lo=100*data.ci95_low.to_numpy();hi=100*data.ci95_high.to_numpy()
            ax.errorbar(mean,np.arange(7),xerr=np.array([mean-lo,hi-mean]),fmt='o',ms=3,color='#3775BA',elinewidth=.8,capsize=2)
            ax.axvline(mean[0],color='#767676',ls='--',lw=.8)
            ax.set(title=f'{pool}: {POOLS[pool]} / {source}',xlabel='Q5 minus Q1 Macro-F1 (pp)',
                yticks=range(7),yticklabels=PLOT_LABELS,ylim=(6.6,-.6),xlim=(-49,-5))
            ax.text(-.1,1.04,chr(97+i*2+j),transform=ax.transAxes,fontweight='bold')
    save(fig,'risk_dimension_ablation')
    random_prose=(
        '随机划层的面板差均值接近零；六个原风险差均超出池内随机划层的95%范围，且10000次中没有出现绝对值至少同样大的随机差。'
        '每项两侧Monte Carlo p为1/10001，六项Holm p为0.00060，不报告p=0。'
        '这支持排除同类别配额下随机分组波动这一解释。它不证明单字段导致失败，也不证明评分优于所有其他条件分组。'
        '随机对照不依赖VLM，因此同池两种来源共享同一随机分布；两来源观测值不是两个独立随机实验。')
    ablation_prose=(
        '三个池、两种来源的全部删维和等权变体中，24个运行的Q5−Q1均为负。各池12项补充家族的Holm p均为0.09375，'
        '所以方向保留不能写成所有消融获得校正显著支持。区间沿用5000次架构/种子层级bootstrap，条件于有限池和已测面板，'
        '不是随机划层区间。完整逐运行值、差值相对完整H的配对变化区间和各类成员重合保留于复现材料。')
    d=a[a.pool=='D'].set_index(['source','variant'])
    scale_local=100*d.loc[('local','drop_size')].change_vs_full
    scale_online=100*d.loc[('online','drop_size')].change_vs_full
    original_gap=[100*d.loc[(s,'full')].mean_delta_auc for s in ('local','online')]
    occlusion_gap=[100*d.loc[(s,'drop_occlusion_level')].mean_delta_auc for s in ('local','online')]
    interpretation=(
        f'Objects365中，删除尺度后的落差缩小最多，本地/线上分别缩小{scale_local:.2f}/{scale_online:.2f}个百分点，'
        '但仍保留负向差。删除遮挡后，完整H的'
        f'{original_gap[0]:.2f}/{original_gap[1]:.2f}个百分点变为{occlusion_gap[0]:.2f}/{occlusion_gap[1]:.2f}个百分点，'
        '说明本轮落差不以遮挡维度存在为必要条件。删维后的样本成员发生改变，且其他维度可能承载相关信号，'
        '因此不将差值变化称为单维独立因果贡献，也不据此宣称遮挡标签准确。')
    result_sections='\n\n'.join([
        '### 5.14 随机分位分层对照：排除同配额随机分组波动',
        '这里检验分层依据，而非既有风险层内部的选样策略。类别内随机五等分在预算与类别比例上与完整风险层一致；统计以Q5−Q1面板差为依据。',
        rt,random_prose,
        f'![图16 风险分层与类别内随机五等分。蓝线为固定风险层的24运行平均Macro-F1，灰虚线为10000次随机划层均值，灰带为随机划层百分位范围。灰带描述分组波动，不是均值置信区间；同池两个标注来源共用随机对照。]({BASE}/figures/risk_vs_random_strata.png)',
        '### 5.15 五维风险消融：方向保留与成员变化',
        '所有删维都重新评分、排序和划层；它与Coverage_without_H的选样字段删除不同。B/C复算核对历史完整五层指标，D补齐同一协议。',
        at,ablation_prose,mt,interpretation,
        f'![图17 完整H、逐维删除与等权变体的风险落差。点为24运行均值，横线为5000次架构/种子层级bootstrap区间，虚线为完整H均值。删维改变了分层成员，区间不作单维因果贡献或全家族显著性保证。]({BASE}/figures/risk_dimension_ablation.png)'])
    text=SOURCE.read_text()
    insertion='### 3.10 随机分位分层与风险维度消融\n\n'
    insertion+=('为检查风险层差是否超出任意随机分组的波动，在每池各类别内均匀随机排列全部目标后等分五层，保持层大小及六类比例。'
        '同次排列由全部24模型和两种标注来源共享。随机划层不读取风险分数、预测或错误，与既有层内Random选样分别评价。\n\n'
        '逐维消融删除尺度、可见性、遮挡、截断或朝向对应分值，保留其他维映射与unknown中点；每个变体按类别重新排序并等分五层，同分沿用样本ID。'
        '另保留各维按最大分值归一化的等权变体。记录Q5−Q1差、相对完整H的配对变化及Q1/Q5成员Jaccard和保留率。'
        '风险维度相关，删维也改变测试成员，因而该分析评价评分敏感性而非字段因果作用。\n\n')
    text=text.replace('## 4 实验设置',insertion+'## 4 实验设置',1)
    settings=('### 4.8 随机划层统计与消融复算\n\n'
        '随机划层重复10000次，B/C/D每类每层分别80/20/80个目标，总层大小480/120/480；各目标每次恰好进入一层。'
        '主统计量为先平均每架构三个种子、再平均八架构的Q5−Q1 Macro-F1差。两侧Monte Carlo p采用(1+极端次数)/(10000+1)，'
        '极端判据为随机差绝对值至少达到观察差绝对值；三池×两来源六项组成新增补充家族，作Holm校正，不替换原风险检验。'
        '随机2.5/97.5百分位只描述有限池内随机分组波动，10000次不充当独立数据集。\n\n'
        '消融沿用历史八架构精确符号翻转和5000次架构/种子层级bootstrap，每池两来源×六变体作12项Holm校正。'
        'B/C复算与既有五层指标、差值、区间及p核对；D新增同口径统计。变体减完整H的变化区间仅用于描述，未扩大为新的显著性检索。'
        '两项补充均在历史结果已见后设计，本轮统计前固定规则；冻结清单、标签、预测与实现，未重新训练、推理、VLM标注或人工复核。\n\n')
    text=text.replace('## 5 实验结果',settings+'## 5 实验结果',1)
    text=text.replace('## 6 讨论',result_sections+'\n\n## 6 讨论',1)
    text=text.replace('三个确认池、两种标注下，全部24次运行的最高风险层Macro-F1均低于最低层。',
        '三个确认池、两种标注下，全部24次运行的最高风险层Macro-F1均低于最低层；类别内随机五等分重复10000次的对照显示，六个风险落差均超出随机分组波动。逐维删除后各运行仍保留负向落差，但消融家族校正未通过。',1)
    text=text.replace('Under both annotators in all three pools, every classifier run has lower Macro-F1 in the highest-risk stratum than in the lowest.',
        'Under both annotators in all three pools, every classifier run has lower Macro-F1 in the highest-risk stratum than in the lowest. A class-balanced random-quintile control with 10,000 repetitions shows that all six risk gaps exceed random-grouping fluctuations. Leave-one-dimension-out variants retain negative run-level gaps, although their corrected ablation tests do not pass.',1)
    target='本文围绕四个研究问题展开。'
    text=text.replace(target,'我们进一步以类别内随机五等分对照检验风险差是否超出随机分组波动，并通过逐维删除与成员重合分析检查评分对单维的依赖。两者分别补充风险层依据与评分敏感性证据，不以此建立视觉字段的因果解释。\n\n'+target,1)
    discussion=('随机分层对照进一步排除了同类别配额下随机分组波动这一解释，逐维删除则说明观察方向不以某一维存在为必要条件。'
        '其中尺度删除改变了较多落差，但相关维度与新层成员共变仍妨碍独立归因；随机零对照通过不表示对其他有意义条件分层普遍优越。')
    text=text.replace('### 6.2 覆盖、难度与性能为何不应合并为一个指标',discussion+'\n\n### 6.2 覆盖、难度与性能为何不应合并为一个指标',1)
    limitation=('随机划层检验条件于固定预测和类别内交换的零假设，只排除随机分组波动，不覆盖所有可能的分组依据。'
        '两标注来源共享随机对照，检验存在依赖，仍按六项作保守Holm校正。消融12项家族受八架构精确p值分辨率限制，'
        '各变体负方向与区间不能替代校正显著性。风险维相关及成员重新分配意味着消融不证明单维因果贡献，也不消除自动状态语义局限。')
    text=text.replace('**流程与选择偏差。**',limitation+'\n\n**流程与选择偏差。**',1)
    text=text.replace('冻结薄弱状态的新池确认补充了具体条件诊断证据，Coverage相对Random的额外诊断广度则具有来源、控制和表征依赖性。',
        '类别内随机五等分对照表明，六个风险落差均超出随机分组波动；逐维删除后，三池两来源所有运行仍保留负向落差，但12项消融校正未通过。'
        '该结果补充了风险层依据及评分敏感性证据，不证明标签真值或字段因果贡献。冻结薄弱状态的新池确认补充了具体条件诊断证据，Coverage相对Random的额外诊断广度则具有来源、控制和表征依赖性。',1)
    anchor='| 风险差和排名在全字段一致筛选下稳健 |'
    text=text.replace(anchor,
        '| 风险分层落差超出随机划层波动 | 三池双来源六项随机对照通过 | 固定池及预测面板的类别内交换零假设，不是因果或所有分组依据优越性 |\n'
        '| 删除单维后风险落差方向保留 | 三池两来源全部运行负向 | 12项消融校正未通过，相关维及成员变化不支持单维归因 |\n'+anchor,1)
    PAPER.write_text(text)
    report='\n\n'.join(['# 随机分层对照与五维消融补强结果',
        '正式统计完成：三池、24份冻结预测、10000次类别内随机五等分；补齐D并复算B/C的五维删除和等权变体。没有新增训练、分类推理、VLM调用或人工复核。',
        '## 1 随机分层对照',rt,random_prose,
        '![随机与风险层性能](figures/risk_vs_random_strata.png)',
        '## 2 五维消融与成员变化',at,ablation_prose,mt,interpretation,
        '![五维消融](figures/risk_dimension_ablation.png)',
        '## 3 可写入论文的结论',
        '固定风险分层的性能落差超出类别内随机分组波动，删除任一风险维后仍保留负向方向。这两项补充支持风险层的经验区分度及方向敏感性，但未验证自动状态真值、评分最优或单字段因果。尺度在本轮删维中影响较大，也不等于尺度是唯一失败原因。',
        '旧强基线与标签分歧限制不变；最新固定预算混合实验已在整合稿保留。主论文另存新版本，原稿保留。',
        '## 4 复算材料',
        f'[实验协议]({ROOT}/docs/random_stratification_ablation_protocol_20261006.md)；[整合论文]({PAPER})。',
        '[随机面板统计](tables/random_summary.csv)；[逐模型随机对照](tables/random_model_summary.csv)；[消融区间与差值变化](tables/ablation_summary.csv)；[成员变化](tables/membership_summary.csv)。',
        '[随机分层SVG](figures/risk_vs_random_strata.svg)；[消融SVG](figures/risk_dimension_ablation.svg)。原始random_strata/*.npz保存完整排列与预测评分，ablation/*.npz保存各评分、层身份与指标。',
        '输入冻结input_freeze.sha256.json，统计审计analysis_audit.json，日志run.log，进度progress.json。首次smoke失败与修订保留，修正后smoke独立通过。'])
    (BASE/'report_zh.md').write_text(report+'\n')
    for name in ('risk_vs_random_strata','risk_dimension_ablation'):
        shutil.copy2(Path(__file__),BASE/'figures'/f'{name}.py')
    print(json.dumps(dict(report=str(BASE/'report_zh.md'),manuscript=str(PAPER),figures=2),ensure_ascii=False),flush=True)


def save(fig,name):
    fig.savefig(BASE/'figures'/f'{name}.svg',bbox_inches='tight')
    fig.savefig(BASE/'figures'/f'{name}.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
