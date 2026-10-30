#!/usr/bin/env python3
"""Render bilingual review copies, English SI and official-template LaTeX.

No inference/training is performed. Tables are checked against retained CSV/NPZ
evidence, and the source manuscript and existing reports remain unchanged.
"""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from docx.oxml.ns import qn

from export_empirical_paper_docx import add_inline, add_page_number
from export_standard_visual_state_paper import three_line_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/journal_revision_20261006'
DOCS = ROOT / 'docs'
EN = DOCS / 'visual_state_testing_journal_en_20261006.md'
ZH = DOCS / 'visual_state_testing_journal_zh_20261006.md'
SI = DOCS / 'visual_state_testing_journal_supplement_20261006.md'
OLD = DOCS / 'visual_state_testing_paper_controls_20261006.md'
CONTROL = ROOT / 'artifacts/random_stratification_ablation_20261006'
MIX = ROOT / 'artifacts/controlled_risk_distribution_20261005'
BASES = {
    'B': ROOT / 'artifacts/sixclass_independent_confirmation_20261004',
    'C': ROOT / 'artifacts/sixclass_voc_external_confirmation_20261005',
    'D': ROOT / 'artifacts/objects365_independent_confirmation_20261005',
}
POOL = {'B': 'COCO', 'C': 'VOC2007', 'D': 'Objects365'}
MODELS = ['resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large',
          'densenet121', 'efficientnet_b0', 'convnext_tiny', 'swin_t']
INPUTS = set()
BLUE, ORANGE = '#2166AC', '#C96D2D'


def read(path):
    INPUTS.add(path)
    return pd.read_csv(path)


def table_blocks(text):
    return re.findall(r'(?:^\|[^\n]*\n){2,}', text, re.M)


def cells(block):
    return [[x.strip() for x in line.strip().strip('|').split('|')]
            for i, line in enumerate(block.strip().splitlines()) if i != 1]


def number(value):
    return float(value.replace('−', '-').replace('%', '').replace(',', ''))


def close(value, expected, decimals=2):
    assert abs(number(value) - float(expected)) < .5 * 10**(-decimals) + 1e-9, (value, expected)


def evidence_and_checks(text):
    blocks = table_blocks(text)
    assert len(blocks) == 11
    random = read(CONTROL / 'tables/random_summary.csv')
    ablation = read(CONTROL / 'tables/ablation_summary.csv')
    assert len(ablation)==42 and (ablation.negative_runs==24).all()
    panel = read(MIX / 'tables/panel_summary.csv')
    contrast = read(MIX / 'tables/distribution_contrasts.csv')
    endpoints=contrast[(contrast.total==120)&(contrast.model!='panel_mean')]
    assert len(endpoints)==48 and (endpoints.macro_f1_delta<0).all()
    ranking=read(MIX / 'tables/model_summary.csv')
    rank_groups=ranking[ranking.total==120].groupby(['pool','source','high_risk_percent'])
    assert len(rank_groups)==126
    for _,group in rank_groups:
        assert group.sort_values('macro_f1_mean',ascending=False).model.iloc[:2].tolist()==['convnext_tiny','swin_t']
    checked = 0
    scores = {}
    for row in cells(blocks[2])[1:]:
        p, s = row[0], row[1].lower()
        r = random[(random.pool == p) & (random.source == s)].iloc[0]
        close(row[2], r.observed_delta * 100); close(row[3], r.random_mean * 100)
        for value, key in zip(re.findall(r'[−+\-]?\d+\.\d+', row[4]), ['random_sampling_low', 'random_sampling_high']):
            close(value, r[key] * 100)
        close(row[5], r.holm_six_p, 5); checked += 5
    for row in cells(blocks[3])[1:]:
        for cell, variant in zip(row[2:], ['full', 'drop_size', 'drop_occlusion_level']):
            r = ablation[(ablation.pool == row[0]) & (ablation.source == row[1].lower()) & (ablation.variant == variant)].iloc[0]
            close(cell, 100 * r.mean_delta_auc); checked += 1
    for row in cells(blocks[4])[1:]:
        p, s = row[0], row[1].lower()
        data = panel[(panel.pool == p) & (panel.source == s) & (panel.total == 120)]
        for cell, share in zip(row[2:4], [0, 100]):
            close(cell, 100 * data[data.high_risk_percent == share].iloc[0].macro_f1_mean); checked += 1
        r = contrast[(contrast.pool == p) & (contrast.source == s) & (contrast.total == 120) & (contrast.model == 'panel_mean')].iloc[0]
        close(row[4], 100 * r.macro_f1_delta); checked += 1
        for value, key in zip(re.findall(r'[−+\-]?\d+\.\d+', row[5]), ['macro_f1_sampling_low', 'macro_f1_sampling_high']):
            close(value, 100 * r[key]); checked += 1
    for j, (p, s) in enumerate((p, s) for p in POOL for s in ['local', 'online']):
        path = CONTROL / 'ablation' / f'{p}_{s}.npz'; INPUTS.add(path)
        with np.load(path) as z:
            score = z['f1'][0].copy()
            assert z['models'].tolist() == [m for m in MODELS for _ in range(3)]
            assert ((score[:, 4] - score[:, 0]) < 0).all()
        scores[(p, s)] = score
        for i, row in enumerate(cells(blocks[5])[1:]):
            close(row[j + 1], 100 * score.reshape(8, 3, 5)[i, :, 4].mean()); checked += 1
    comparisons = {}
    for p, base in BASES.items():
        comparisons[p] = read(base / 'comparison/tables/comparisons.csv')
    for row in cells(blocks[6])[1:]:
        data = comparisons[row[0]]
        data = data[(data.source == row[1].lower()) & (data.baseline == 'random')]
        for j, metric in enumerate(['confusion', 'clusters']):
            r = data[data.metric == metric].iloc[0]
            close(row[2 + 2*j], r.mean_delta_auc, 4)
            close(row[3 + 2*j], r.holm_primary4_p, 5); checked += 2
    for row in cells(blocks[7])[1:]:
        data = read(BASES[row[0]] / 'transfer/tables/comparisons.csv')
        r = data[data.source == row[1].lower()].iloc[0]
        close(row[4], 100 * r.mean_delta_auc); close(row[5], r.holm_two_p, 5); checked += 2
    # Independently recompute support and case summaries from retained outputs.
    transfer = {p: read(base / 'transfer/tables/state_validation.csv') for p, base in BASES.items()}
    for row in cells(blocks[7])[1:]:
        data = transfer[row[0]]
        data = data[data.source == row[1].lower()]
        keys = ['model', 'class_name', 'feature', 'value']
        close(row[2], len(data[keys].drop_duplicates()))
        close(row[3], len(data[data.B_evaluable.astype(str).str.lower() == 'true'][keys].drop_duplicates()))
        checked += 2
    conditions = [('local','resnet50','dog','body_orientation','back'),
                  ('local','resnet50','cat','target_scale','small'),
                  ('online','resnet50','sheep','blur_level','severe'),
                  ('local','densenet121','dog','target_scale','medium')]
    for row, condition in zip(cells(blocks[8])[1:], conditions):
        source,model,category,field,state=condition
        data=transfer['B']
        data=data[(data.source==source)&(data.model==model)&(data.class_name==category)&(data.feature==field)&(data.value==state)]
        assert len(data)==3,condition
        close(row[2],data.B_support.iloc[0]);close(row[3],100*data.B_class_recall.mean())
        close(row[4],100*data.B_recall.mean());close(row[5],100*data.B_gap.mean());checked+=4
    reference=read(ROOT/'artifacts/ranking_attribute_validation_20261005/tables/attribute_metrics.csv')
    for row in cells(blocks[9])[1:]:
        field={'Occlusion':'occlusion_level','Truncation':'truncation_level'}[row[0]]
        data=reference[(reference.field==field)&(reference.source==row[1].lower())&(reference.aggregation=='target_weighted')].iloc[0]
        close(row[2],data.known);close(row[3],100*data.agreement)
        close(row[4],100*data.f1);close(row[5],data.kappa,3);checked+=4
    retention=read(ROOT/'artifacts/label_disagreement_sensitivity_20261005/tables/retention_distribution.csv')
    for row in cells(blocks[10])[1:]:
        data=retention[(retention.pool==row[0])&(retention.source=='local')]
        risk=data[data.group=='risk_agree'];shared=data[data.group=='shared_agree']
        close(row[2],risk.n.sum());close(row[3],100*risk.n.sum()/float(row[1].replace(',','')))
        close(row[4],shared.n.sum());close(row[5],risk[risk.tier==5].n.sum());checked+=4
    # Also retain compatibility checks with the independently audited full paper.
    old = OLD.read_text(); INPUTS.add(OLD)
    for idx in [7, 8, 9, 10]:
        for row in cells(blocks[idx])[1:]:
            for value in row:
                if re.fullmatch(r'[−+\-]?\d[\d,]*(?:\.\d+)?', value):
                    assert value in old or value.replace(',', '') in old, value
    for base in [CONTROL, MIX]:
        audit = base / 'final_audit.json'; INPUTS.add(audit)
        assert json.loads(audit.read_text())['status'] == 'pass'
    return random, ablation, panel, contrast, scores, comparisons, checked


TRANSLATIONS = {
    'Dimension': '维度', 'Mapping to score': '分值映射', 'Area ratio a': '面积比a',
    'Body visibility': '身体可见性', 'Occlusion': '遮挡', 'Truncation': '截断', 'Orientation': '朝向',
    'Set': '集合', 'Source': '来源', 'Targets': '目标数', 'Per class': '每类', 'Role': '用途',
    'Training': '训练', 'Validation': '验证', 'Base test': '原始测试',
    'Classifier fitting': '模型训练', 'Checkpoint selection': '权重选择', 'Historical reference': '历史参考',
    'Weak-state discovery': '薄弱状态发现', 'New-target confirmation': '新目标确认',
    'External-source confirmation': '外部来源确认', 'Larger external confirmation': '更大外部确认',
    'Pool': '池', 'Annotator': '标注来源', 'Local': '本地', 'Online': '线上', 'Program/local': '程序/本地',
    'Observed Q5−Q1': '实际Q5−Q1', 'Random mean': '随机均值', 'Random 95% range': '随机95%范围',
    'Six-test Holm p': '六项Holm p', 'Full score': '完整评分', 'Without scale': '去尺度', 'Without occlusion': '去遮挡',
    '0% Q5': '0% Q5', '100% Q5': '100% Q5', 'Difference': '差值', 'Paired 95% sampling range': '成对95%抽样范围',
    'Architecture': '架构', 'B local': 'B本地', 'B online': 'B线上', 'C local': 'C本地',
    'C online': 'C线上', 'D local': 'D本地', 'D online': 'D线上',
    'Confusion AUC difference': '混淆AUC差', 'Cluster AUC difference': '错误簇AUC差',
    'Frozen combinations': '冻结组合', 'Evaluable': '可评价', 'Mean recall difference (pp)': '平均Recall差(pp)',
    'Two-test Holm p': '两项Holm p', 'Architecture and condition': '架构与条件', 'Support': '支持量',
    'Class recall (%)': '类别Recall(%)', 'State recall (%)': '状态Recall(%)', 'Difference (pp)': '差(pp)',
    'ResNet-50, back-facing dog': 'ResNet-50，背向dog', 'ResNet-50, small-area cat': 'ResNet-50，小尺度cat',
    'ResNet-50, heavily blurred sheep': 'ResNet-50，严重模糊sheep', 'DenseNet-121, medium-area dog': 'DenseNet-121，中等尺度dog',
    'Attribute': '属性', 'Agreement (%)': '一致率(%)', 'Attribute F1 (%)': '属性F1(%)',
    'Four-risk-field agreement': '四风险字段一致', 'Retained (%)': '保留(%)',
    'All-14-field agreement': '全14字段一致', 'Retained Q5 targets': '保留Q5目标',
}


def resolve_chinese(en):
    text = ZH.read_text()
    for i, block in enumerate(table_blocks(en), 1):
        translated = block
        for source in sorted(TRANSLATIONS, key=len, reverse=True):
            translated = re.sub(r'(?<=\| )' + re.escape(source) + r'(?= \|)', TRANSLATIONS[source], translated)
        text = text.replace(f'<!-- JOURNAL_TABLE_{i} -->', translated.strip())
    text = text.split('## 参考文献\n', 1)[0] + '## 参考文献\n\n' + en.split('## References\n', 1)[1].strip() + '\n'
    assert '<!-- JOURNAL_' not in text
    ZH.write_text(text)


def save(fig, name):
    for ext in ['svg', 'pdf', 'eps', 'png']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig)


def figures(random, panel):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'axes.linewidth': .7})
    fig, ax = plt.subplots(figsize=(6.2, 3.7)); ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis('off')
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1', facecolor=color, edgecolor='#666666', linewidth=.7))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, color='#666666', linewidth=1))
    box(.1, 2.3, 2.1, 1.3, 'Official labels\nTarget boxes\n+ categories', '#E7EEF7')
    box(3.15, 3.4, 2.8, 1.15, 'Dual VLM states\nNo classifier\noutcomes', '#E7EEF7')
    box(3.15, 1.5, 2.8, 1.15, 'Frozen classifiers\n24 prediction sets', '#F6E8DA')
    box(7.0, 3.4, 2.7, 1.15, 'Coverage + risk\nTest-suite\norganization', '#E7EEF7')
    box(7.0, 1.5, 2.7, 1.15, 'Conditional results\nPerformance\n+ diagnosis', '#F6E8DA')
    for a,b in [((2.3,3.1),(3.05,3.95)),((2.3,2.65),(3.05,2.05)),((6.05,3.95),(6.9,3.95)),((6.05,2.05),(6.9,2.05)),((8.35,3.3),(8.35,2.75))]: arrow(a,b)
    ax.text(5,.45,'Discovery A: freeze conditions\nConfirm B / C / D: reuse frozen models',ha='center',fontsize=10)
    save(fig,'01_workflow')
    fig, ax = plt.subplots(figsize=(6.2, 3.9), layout='constrained')
    for i, r in random.iterrows():
        lo, hi, gap = 100*r.random_sampling_low, 100*r.random_sampling_high, 100*r.observed_delta
        ax.plot([lo,hi],[i,i],color='#B9B9B9',linewidth=8,solid_capstyle='butt',label='Random central 95%' if i==0 else None)
        ax.scatter(100*r.random_mean,i,color='#333333',marker='|',s=100)
        ax.scatter(gap,i,color=BLUE,marker='D',s=30,label='Risk-defined gap' if i==0 else None)
        ax.annotate(f'{gap:.2f}',(gap,i),xytext=(-7,0),textcoords='offset points',ha='right',va='center',color=BLUE,fontsize=10)
        ax.annotate(f'[{lo:+.2f}, {hi:+.2f}]',(hi,i),xytext=(7,0),textcoords='offset points',ha='left',va='center',fontsize=10)
    ax.axvline(0,color='#777777',ls='--',lw=.6)
    ax.set(yticks=range(6),yticklabels=[f'{POOL[r.pool]} · {r.source}' for _,r in random.iterrows()],ylim=(5.5,-.7),xlim=(-53,22),xlabel='Macro-F1: Q5 − Q1 (percentage points)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=2,fontsize=10)
    save(fig,'02_risk_random')
    fig, axes = plt.subplots(1,3,figsize=(6.2,2.65),sharey=True,layout='constrained')
    for ax,(p,name) in zip(axes,POOL.items()):
        for source,color,style in [('local',BLUE,'-'),('online',ORANGE,'--')]:
            d=panel[(panel.pool==p)&(panel.source==source)&(panel.total==120)].sort_values('high_risk_percent')
            assert len(d)==21
            delta=100*(d.macro_f1_mean.to_numpy()-d.macro_f1_mean.iloc[0])
            ax.plot(d.high_risk_percent,delta,color=color,ls=style,lw=1.5,marker='o' if source=='local' else 's',ms=3,markevery=4,label=source.capitalize())
            offset = -12 if source=='local' else 12
            ax.annotate(f'{delta[-1]:.2f}',(100,delta[-1]),xytext=(-6,offset),textcoords='offset points',ha='right',va='top' if offset<0 else 'bottom',fontsize=10,color=color,bbox={'facecolor':'white','edgecolor':'none','pad':.7},arrowprops={'arrowstyle':'-','color':color,'lw':.6})
        ax.axhline(0,color='#888888',lw=.6); ax.set(title=name,xlim=(-2,104),ylim=(-44,3),xticks=[0,25,50,75,100],xlabel='Q5 share (%)')
        ax.grid(axis='y',color='#E6E6E6',lw=.5)
    axes[0].set_ylabel('Δ Macro-F1 (percentage points)'); axes[0].legend(fontsize=10,loc='lower left')
    save(fig,'03_mixture_response')
    fig,ax=plt.subplots(figsize=(6.2,3.7),layout='constrained')
    names=['ResNet-50\nBack-facing dog','ResNet-50\nSmall-area cat','ResNet-50\nBlurred sheep','DenseNet-121\nMedium-area dog']
    category=[81.92,82.67,88.42,73.42]; state=[57.25,39.13,57.58,81.01]; x=np.arange(4)
    b1=ax.bar(x-.18,category,.36,color=BLUE,label='Class recall'); b2=ax.bar(x+.18,state,.36,color=ORANGE,hatch='//',label='Conditional recall')
    ax.bar_label(b1,fmt='%.2f',padding=3,fontsize=10); ax.bar_label(b2,fmt='%.2f',padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Recall (%)');ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.12),fontsize=10)
    save(fig,'04_condition_cases')


def md_table(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        vals=[f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row]
        rows.append('| '+' | '.join(vals)+' |')
    return '\n'.join(rows)


def supplement(ablation, comparisons):
    lines=['# Supplementary Information: Visual-State Risk Stratification',
           '## S1 Protocol, schema and reproducibility scope',
           'The supplement accompanies the revised empirical manuscript. Numbers below are retained results, not new model runs. The evidence bundle includes CSV tables, protocol files and an index; image assets and full prediction caches remain in the local experiment directories and are not falsely described as publicly deposited.',
           'The generic native object schema and the car prompt are included. The shared mapping is implemented in `scripts/dual_vlm_common.py`; source-selection and confirmation logic is preserved by the accompanying protocol documents. Unknown states remain unknown. Original-image scale is computed, not inferred by a VLM.',
           '**Table S1. Generic native fields and allowed states. Shared-field normalization is given in the included source code.**']
    schema_path=ROOT/'configs/multiclass_object_state_schema_v1_0.json';INPUTS.add(schema_path)
    schema=json.loads(schema_path.read_text())
    frame=pd.DataFrame([{'Field':r['feature_id'],'Type':r['value_type'],'Allowed states':', '.join(r['possible_values'])} for r in schema['universal_features']])
    lines.append(md_table(frame))
    lines += ['## S2 Analysis families and interval interpretation',
              'Architecture-level paired inference first averages the three training seeds. The eight-architecture exact two-sided sign-flip test has minimum p = 2/256 = 0.0078125. Pool-wise Holm families remain separate, and the all-20 sensitivity correction remains available. The 12-test strong-baseline and ablation families cannot reject their first null at 0.05 with this test resolution. A positive effect is not a corrected rejection, and a failure to reject is not evidence of equivalence.',
              'Random-stratification central ranges describe a finite-pool grouping null. Fixed-budget mixture ranges describe paired repeated sampling, rather than the precision of the mean. Hierarchical model/seed confidence intervals condition on the available targets; source-image bootstraps condition on the fixed classifiers. Failed mixed-model fits supply no inference. No additional successful statistical model is invented.',
              'The historical discovery and later cached analyses are not all prospective. Conditions and source-selection rules for confirmation were frozen before new predictions where documented. The random-stratification and dimension-ablation analyses are explicitly supplementary checks added after earlier results.',
              '## S3 All score variants',
              'Each row is the panel-average Q5−Q1 difference under a redefined score, not a same-target causal contrast. Negative/positive run counts are descriptive. Confidence intervals condition on the retained targets. Membership rank correlations and Q5 overlaps are available in `membership_summary.csv`.',
              '**Table S2. Full and six alternative score definitions across all pool–annotator settings. Differences and confidence intervals are in F1 units (multiply by 100 for percentage points).**']
    lines.append(md_table(ablation[['pool','source','variant','mean_delta_auc','ci95_low','ci95_high','negative_runs','holm_twelve_p']].fillna('—')))
    lines += ['## S4 All selection contrasts, including negative results',
              'The main comparison imposes the same category/stratum risk-cell quotas for all strategies. Error quantities are matched within each true category by exact rarefaction. AUC is normalized trapezoidal area over budget and includes the zero-budget origin. It is not ROC-AUC. A classifier-representation cluster is a diversity proxy, not a confirmed fault root cause. DeepGD is discussed as related work but is not evaluated directly.']
    for j,(p,frame) in enumerate(comparisons.items(),3):
        lines += [f'**Table S{j}. Complete selection contrasts in {POOL[p]}; primary, secondary and joint corrections are retained.**',
                  md_table(frame[['source','baseline','metric','mean_delta_auc','ci95_low','ci95_high','holm_primary4_p','holm_strong12_p','holm_ablation4_p','holm_all20_p']].fillna('—'))]
    lines += ['## S5 Risk matching and common-representation sensitivity',
              "The exact-histogram control first changes Coverage's feasible set, then matches Random to that Coverage sequence. It is a different protocol from the common quotas. Separate tables for all pools are provided in the evidence bundle. For D, only the local confusion-type contrast passes the original four-test correction under exact matching. The common ViT cluster sensitivity supports D but not B or C. These limits prevent a universal risk-independent or representation-independent selection claim.",
              '## S6 Frozen-state support and transfer',
              'A supplies 87 local and 88 online state–architecture hypotheses. A state requires at least 20 unique targets and a recall deficit of at least ten percentage points at discovery. Confirmation never replaces unsupported hypotheses with favorable new ones. B supports 78/75 combinations; C supports 1/3; D supports 79/81. D retains 69/73 negative directions after averaging seeds, but the primary mean uses the prescribed architecture/run aggregation. Overlapping state descriptions do not create independent defects. Risk-matched residual recall deficits in D (−8.33/−7.80 points) remain descriptive associations.',
              '## S7 Annotation disagreement and reference checks',
              'Strict agreement requires all four annotated risk dimensions to be known and equal. B/C/D retain 166/56/187 targets and only five Q5 targets in each pool. This is insufficient for the planned six-class agreement-only endpoint comparison. Agreement on all 14 shared fields retains 4/0/8 targets. Fieldwise agreement-only confirmation retains many B/D deficit directions, but uses a weaker selected subset and does not validate the entire protocol.',
              'The reference-attribute CSV contains the full precision, recall, specificity, F1 and kappa outputs for official binary Open Images occlusion and truncation. Different official and VLM definitions limit interpretation. Programmatic scale is recalculated on all 7,800 pool targets; agreement of computed scale is not validation of semantic VLM labels.',
              '## S8 Mixtures, rankings and raw failure breadth',
              'The primary mixture budget is 120 (20 per category), with 21 actual risk-share points and 1,000 paired draws per point. No observations are inserted by smoothing. The zero-share suite samples Q1–Q4; it is not a pure Q1 suite. B/D additionally permit 480 targets. Complete panel, class and architecture tables are included. ConvNeXt-Tiny and Swin-T lead mean rankings at all 126 primary settings; image-bootstrap uncertainty remains material, especially on D. Intermediate ranks can move four places.',
              'Raw failed-state breadth increases, whereas class-specific equal-error confusion and cluster changes against the 20%-Q5 reference do not show consistent increases. These controls prevent interpreting more mistakes as automatically broader distinct failure evidence.',
              '## S9 Cost and implementation provenance',
              'B successful-request median/P95 seconds: local 4.00/4.30, online 11.41/14.62. C: local 3.94/4.30, online 9.92/12.31. D all-attempt median/P95 seconds: local 4.00/4.34, online 10.13/13.84; attempt counts are 2,409/2,826. The success-only and all-attempt latency populations are not comparable. No invoice-backed monetary cost is available.',
              'The D source amendment precedes new classification predictions: initial validation sheep supply was insufficient, so fixed train-package prefixes were added and the documented download budget increased from 40 to 80 GiB. The realized fixed-prefix download was 67.76 GiB. Annotation-service and technical replacement records are preserved; selection was not optimized on classifier outcomes.',
              '## S10 Evidence inventory and local reconstruction',
              'The bundle provides source-relative paths and SHA-256 checksums in `evidence_index.json`. Main Table 3 maps to random_summary.csv; Table 4 to ablation_summary.csv; Tables 5–6 to distribution/panel tables and full-stratum NPZs; Table 7 to each pool comparison table; Table 8 to frozen transfer tables; Tables 9–11 to the audited historical manuscript and attribute/disagreement outputs. These indices identify retained local evidence rather than a currently public repository.',
              'The full Chinese evidence manuscript is separately preserved as a working archive; its 28 numbered tables, appendix and 17 figures have not been deleted. It is not advertised as an English journal supplement. Figure sources are editable SVG with English labels, plus PDF/EPS/PNG exports. The LaTeX source uses the official Springer class and contains actual author/declaration placeholders; authors must complete those facts and actual data-access conditions before submitting.']
    SI.write_text('\n\n'.join(lines)+'\n')


def word(source, math=True):
    doc=Document();sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Cm(2.5)
    add_page_number(sec.footer.paragraphs[0])
    for name,size in [('Normal',11),('Title',17),('Heading 1',13),('Heading 2',11.5),('Heading 3',11),('Caption',9)]:
        st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size)
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'黑体' if 'Heading' in name else '宋体')
        st.paragraph_format.space_after=Pt(6)
        if 'Heading' in name: st.paragraph_format.keep_with_next=True
    equations=[]
    if math:
        old=Document(DOCS/'visual_state_testing_empirical_paper_updated_20261006.docx')
        equations=[deepcopy(n) for n in old._element.findall('.//'+qn('m:oMath'))]
        assert len(equations)==3
    lines=source.read_text().splitlines();i=0;eq=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line=='$$':
            while i<len(lines) and lines[i].strip()!='$$':i+=1
            i+=1;p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p._p.append(equations[eq]);eq+=1;continue
        if line.startswith('|'):
            rows=[line]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            three_line_table(doc,rows);continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            path=(source.parent/match[2]).resolve();assert path.is_file()
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path),width=Cm(16))
            p=doc.add_paragraph(style='Caption');add_inline(p,match[1],9);continue
        match=re.match(r'^(#{1,4})\s+(.*)',line)
        if match:
            depth=len(match[1]);title=match[2]
            if depth==1:doc.add_paragraph(title,style='Title')
            else:doc.add_heading(title,level=depth-1)
            continue
        if line.startswith('**Table ') or line.startswith('**表'):
            p=doc.add_paragraph(style='Caption');p.paragraph_format.keep_with_next=True;add_inline(p,line,9);continue
        p=doc.add_paragraph(style='List Bullet' if line.startswith('- ') else None)
        add_inline(p,line[2:] if line.startswith('- ') else line,11)
        p.paragraph_format.line_spacing=1.15
    if math:assert eq==3
    dest=source.with_suffix('.docx');doc.save(dest)
    return {'file':str(dest.relative_to(ROOT)),'tables':len(doc.tables),'pictures':len(doc.inline_shapes),'equations':eq}


def tex_escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                  '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
                  '−':r'\ensuremath{-}','≥':r'\ensuremath{\geq}','≤':r'\ensuremath{\leq}',
                  '–':'--','—':'---','×':r'\ensuremath{\times}','→':r'\ensuremath{\rightarrow}',
                  'Δ':r'\ensuremath{\Delta}'}
    return ''.join(replacements.get(c,c) for c in text)


def tex_inline(text):
    pieces=re.split(r'(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|https?://[^\s]+)',text)
    result=[]
    for item in pieces:
        if item.startswith('`'):result.append(r'\texttt{'+tex_escape(item[1:-1])+'}')
        elif item.startswith('**'):result.append(r'\textbf{'+tex_escape(item[2:-2])+'}')
        elif item.startswith('*'):result.append(r'\textit{'+tex_escape(item[1:-1])+'}')
        elif item.startswith('http'):result.append(r'\url{'+item+'}')
        else:result.append(tex_escape(item))
    return ''.join(result)


def tex_prose(text):
    names={
        'Aghababaeyan et al., 2023':['ref1'], 'Aghababaeyan et al., 2024':['ref2'],
        'Aghababaeyan et al., 2023, 2024':['ref1','ref2'], 'Dola et al., 2023':['ref3'],
        'Feng et al., 2020':['ref4'], 'Harel-Canada et al., 2020':['ref5'],
        'Kim et al., 2019':['ref6'], 'Lin et al., 2014':['ref7'], 'Ma et al., 2018':['ref8'],
        'Objects365 Consortium, n.d.':['ref9'], 'Open Images, n.d.':['ref10'],
        'PASCAL VOC, 2007':['ref11'], 'Pei et al., 2017':['ref12'], 'Sener and Savarese, 2018':['ref13'],
    }
    macros=[]
    def placeholder(macro):
        key=f'@@CITE{len(macros)}@@';macros.append((key,macro));return key
    def citation(match):
        parts=[p.strip() for p in match[1].split(';')]
        if not all(p in names for p in parts):return match[0]
        keys=[key for p in parts for key in names[p]]
        return placeholder(r'\citep{'+','.join(keys)+'}')
    text=re.sub(r'\(([^()]+)\)',citation,text)
    for label,keys in names.items():
        if len(keys)!=1:continue
        author,year=label.rsplit(', ',1)
        text=text.replace(author+' ('+year+')',placeholder(r'\citet{'+keys[0]+'}'))
    result=tex_inline(text)
    for key,macro in macros:result=result.replace(key,macro)
    return result


def latex_main(text, reference_labels=None):
    dest=OUT/'latex';dest.mkdir(exist_ok=True)
    with ZipFile(OUT/'vendor/springer_template.zip') as z:
        for member in ['sn-article-template/sn-jnl.cls','sn-article-template/bst/sn-basic.bst']:
            target=dest/Path(member).name
            target.write_bytes(z.read(member))
    title=text.splitlines()[0][2:]
    abstract=text.split('## Abstract\n',1)[1].split('**Keywords:',1)[0].strip()
    keywords=text.split('**Keywords:** ',1)[1].splitlines()[0]
    preamble=r'''\documentclass[pdflatex,sn-basic]{sn-jnl}
\usepackage{graphicx,booktabs,tabularx,array,amsmath,amssymb}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\raggedbottom
\begin{document}
'''
    output=[preamble,r'\title{'+tex_escape(title)+'}',
            r'\author*[1]{\fnm{AUTHOR} \sur{TO COMPLETE}}',r'\email{AUTHOR EMAIL TO COMPLETE}',
            r'\affil*[1]{\orgname{AUTHOR AFFILIATION TO COMPLETE},\orgaddress{\country{AUTHOR COUNTRY TO COMPLETE}}}',
            r'\abstract{'+tex_inline(abstract)+'}',r'\keywords{'+tex_escape(keywords)+'}',r'\maketitle']
    lines=text.split('## 1 Introduction\n',1)[1].splitlines()
    lines.insert(0,'## 1 Introduction');i=0;tablecount=0;figcount=0;refs=False;refcount=0
    if reference_labels is None:
        reference_labels=['Aghababaeyan et al.(2023)','Aghababaeyan et al.(2024)','Dola et al.(2023)','Feng et al.(2020)',
                      'Harel-Canada et al.(2020)','Kim et al.(2019)','Lin et al.(2014)','Ma et al.(2018)',
                      'Objects365 Consortium(n.d.)','Open Images(n.d.)','PASCAL VOC(2007)','Pei et al.(2017)','Sener and Savarese(2018)']
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line=='$$':
            formula=[]
            while lines[i].strip()!='$$':formula.append(lines[i]);i+=1
            i+=1;output.append('\\begin{equation}\n'+'\n'.join(formula)+'\n\\end{equation}');continue
        if line.startswith('**Table '):
            caption=re.sub(r'^\*\*Table \d+\.\s*','',line).removesuffix('**')
            while i<len(lines) and not lines[i].startswith('|'):i+=1
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
            values=cells('\n'.join(rows));n=len(values[0]);tablecount+=1
            output += [r'\begin{table}[!htbp]',r'\caption{'+tex_inline(caption)+r'}\label{tab:'+str(tablecount)+'}',
                       r'\small\setlength{\tabcolsep}{3pt}',r'\begin{tabularx}{\textwidth}{'+'Y'*n+'}',r'\toprule',
                       ' & '.join(r'\textbf{'+tex_inline(c)+'}' for c in values[0])+r' \\',r'\midrule']
            output += [' & '.join(tex_inline(c) for c in row)+r' \\' for row in values[1:]]
            output += [r'\botrule',r'\end{tabularx}',r'\end{table}'];continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            figcount+=1;original=(EN.parent/match[2]).resolve();local=f'figure_{figcount}.pdf'
            shutil.copy2(original.with_suffix('.pdf'),dest/local)
            caption=re.sub(r'^Figure \d+\.\s*','',match[1])
            output += [r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=\textwidth]{'+local+'}',
                       r'\caption{'+tex_inline(caption)+r'}\label{fig:'+str(figcount)+'}',r'\end{figure}'];continue
        match=re.match(r'^(#{2,3})\s+(.*)',line)
        if match:
            title=re.sub(r'^\d+(?:\.\d+)*\s+','',match[2]);depth=len(match[1])
            if title=='References':
                output.append(r'\begin{thebibliography}{'+str(len(reference_labels))+'}');refs=True;continue
            command='section' if depth==2 else 'subsection'
            if title=='Statements and Declarations':command+='*'
            output.append('\\'+command+'{'+tex_escape(title)+'}');continue
        if refs:
            output.append(r'\bibitem['+reference_labels[refcount]+']{ref'+str(refcount+1)+'} '+tex_inline(line));refcount+=1
        else:output.append(tex_prose(line)+'\n')
    output += [r'\end{thebibliography}',r'\end{document}']
    assert tablecount==len(re.findall(r'\*\*Table \d+\.',text))
    assert figcount==len(re.findall(r'!\[Figure \d+\.',text))
    assert refcount==len(reference_labels)
    (dest/'main.tex').write_text('\n\n'.join(output)+'\n')
    generated=(dest/'main.tex').read_text()
    referenced=set(key for group in re.findall(r'\\cite[pt]\{([^}]+)\}',generated) for key in group.split(','))
    assert referenced==set('ref'+str(i) for i in range(1,len(reference_labels)+1))
    for env in ['document','table','figure','equation','tabularx','thebibliography']:
        assert generated.count(r'\begin{'+env+'}')==generated.count(r'\end{'+env+'}')
    (dest/'README.txt').write_text('Review draft: fill actual authors, funding, conflicts, contributions and data access before submission.\nCompile main.tex with pdflatex (twice). Bibliography is embedded; author-year citations are explicit in the text.\nThe official sn-jnl.cls and sn-basic.bst are unchanged template files. Figures use English labels and embedded fonts.\nThe supplement is a separate document. This source bundle does not include raw images or full prediction caches.\n')


def inventory_and_audit(text, word_records, checked):
    abstract=text.split('## Abstract\n',1)[1].split('**Keywords:',1)[0].strip()
    words=len(re.findall(r"\b[\w]+(?:[’'\-][\w]+)*\b",abstract))
    assert 150<=words<=250,words
    keywords=text.split('**Keywords:** ',1)[1].splitlines()[0].split(';');assert len(keywords)==6
    assert re.findall(r'\*\*Table (\d+)\.',text)==list(map(str,range(1,12)))
    assert re.findall(r'!\[Figure (\d+)\.',text)==['1','2','3','4']
    assert len(re.findall(r'^#{4,}',text,re.M))==0
    assert len(re.findall(r'\$\$\n',text))==6
    svgs=[]
    for path in sorted((OUT/'figures').glob('*.svg')):
        root=ET.parse(path).getroot();tags=[n.tag.split('}')[-1] for n in root.iter()]
        assert tags.count('text')>0 and tags.count('image')==0
        svgs.append({'file':str(path.relative_to(ROOT)),'text':tags.count('text'),'embedded_rasters':tags.count('image')})
    for record in word_records[:2]:
        assert record['tables']==11 and record['pictures']==4 and record['equations']==3
    entries=[]
    for p in sorted(INPUTS):
        entries.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'evidence_index.json').write_text(json.dumps(entries,indent=2)+'\n')
    report={'status':'pass','abstract_words':words,'keywords':len(keywords),'main_tables':11,'main_figures':4,
            'numeric_cells_checked_against_csv_npz':checked,'additional_case_reference_support_checks':'CSV recomputation and retained audited manuscript',
            'source_files':len(entries),'word_documents':word_records,'svg_checks':svgs,
            'author_facts_pending':True,'public_repository_pending':True,
            'compilation':{'status':'blocked_by_resource_download','attempts':3,'compiled_pdf_available':False},
            'word_full_page_rendering':'not performed; structure and embedded figures checked',
            'new_classifier_or_vlm_runs':0,'full_prediction_caches_bundled':False}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def evidence_bundle():
    folder=OUT/'evidence_tables';folder.mkdir(exist_ok=True)
    for p in sorted(INPUTS):
        if p.suffix=='.csv':
            name='__'.join(p.parts[-4:]);shutil.copy2(p,folder/name)
    for parent in [CONTROL/'tables',MIX/'tables',ROOT/'artifacts/label_disagreement_sensitivity_20261005/tables',ROOT/'artifacts/ranking_attribute_validation_20261005/tables']:
        for p in sorted(parent.glob('*.csv')):
            if p.stat().st_size<5_000_000:
                shutil.copy2(p,folder/('__'.join(p.parts[-3:])))
    with ZipFile(DOCS/'visual_state_testing_journal_materials_20261006.zip','w',ZIP_DEFLATED) as z:
        for p in [EN,ZH,SI,EN.with_suffix('.docx'),ZH.with_suffix('.docx'),SI.with_suffix('.docx'),DOCS/'期刊优化说明_20261006.md',DOCS/'期刊优化说明_20261006.docx',OUT/'evidence_index.json',OUT/'audit.json']:
            z.write(p,'manuscripts/'+p.name)
        for p in OUT.glob('*.md'):z.write(p,'revision_audit/'+p.name)
        for p in folder.iterdir():z.write(p,'evidence_tables/'+p.name)
        for p in (OUT/'figures').iterdir():z.write(p,'figures/'+p.name)
        for p in (OUT/'latex').iterdir():
            if p.suffix not in ['.log','.aux','.out']:z.write(p,'latex/'+p.name)
        for p in DOCS.glob('*protocol*2026100*.md'):z.write(p,'protocols/'+p.name)
        for p in [ROOT/'configs/multiclass_object_state_schema_v1_0.json',ROOT/'scripts/dual_vlm_common.py',ROOT/'prompts/feature_annotation_car_shared_v1_0.txt',ROOT/'scripts/build_journal_revision.py']:
            z.write(p,'protocol_sources/'+p.name)
    with ZipFile(DOCS/'visual_state_testing_sqj_latex_20261006.zip','w',ZIP_DEFLATED) as z:
        for p in (OUT/'latex').iterdir():
            if p.suffix not in ['.log','.aux','.out']:z.write(p,p.name)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    text=EN.read_text()
    random,ablation,panel,contrast,scores,comparisons,checked=evidence_and_checks(text)
    resolve_chinese(text);figures(random,panel);supplement(ablation,comparisons)
    records=[word(EN),word(ZH),word(SI,False),word(DOCS/'期刊优化说明_20261006.md',False)]
    latex_main(text);audit=inventory_and_audit(text,records,checked);evidence_bundle()
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
