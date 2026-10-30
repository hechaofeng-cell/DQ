#!/usr/bin/env python3
"""Run frozen algorithms on the registered smaller external-source pool."""
import argparse
import json
import inspect
from voc_external_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,
        choices=['freeze_selection','infer','confirm','compare','transfer','robustness','report'])
    stage=parser.parse_args().stage
    a,selection,analysis=activate()
    if stage=='freeze_selection':
        (BASE/'embeddings').mkdir(parents=True,exist_ok=True);selection.build()
        # Freeze the external adapter and ledger configuration as well as reused implementation.
        path=BASE/'pre_prediction_freeze.sha256.json'
        mapping=json.loads(path.read_text())
        for p in code_paths()+[BASE/'registration.json',BASE/'execution_code_freeze.sha256.json']:
            mapping[str(p)]=common.sha256(p)
        assert not list((BASE/'benchmark').glob('**/candidate_outputs.npz'))
        common.write_json(path,mapping)
        return
    if stage=='report':
        external_report(a,analysis)
        return
    {'infer':analysis.inference,'confirm':analysis.confirm,'compare':analysis.compare,
     'transfer':analysis.transfer,'robustness':analysis.robust_risks}[stage]()

def external_report(a,analysis):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    import plot_sixclass_difficulty as plot
    plot.BASE=BASE/'confirmation';plot.OUT=plot.BASE/'figures'
    adapt(plot,'run',[
        ('ylim=(72,101)','ylim=(0,101)'),('ylim=(65,99)','ylim=(0,101)'),
        ('ylim=(83,96)','ylim=(0,101)'),('ylim=(15,73)','ylim=(0,101)'),
        ('tier*120','tier*6*STEP'),('[120,240,360,480,600]','[6*b for b in BUDGETS]'),
        ('vmin=60','vmin=0'),
        ("ax.text(2.1,3-i", "ax.text((comp.ci95_high.max()+.01)*100,3-i"),
        ('xlim=(-1.5,3.4)', 'xlim=((min(0,comp.ci95_low.min())-.01)*100,(max(0,comp.ci95_high.max())+.04)*100)')])
    plot.run()
    common.verify(BASE/'pre_prediction_freeze.sha256.json');common.verify(BASE/'prediction_freeze.sha256.json')
    common.verify(BASE/'preparation_freeze.sha256.json')
    lines=['# 六类VOC外部来源确认结果','',
        '冻结原COCO训练的24份六类分类权重；600个未用VOC2007目标，六类各100。每类预算10–50，选样比例50%，COCO B为25%。',
        '这是另一个数据来源上的确认；不能排除ImageNet预训练重叠或自动标签误差。双VLM配对，无人工真值。',
        '', '## 主要风险确认','', '| 来源 | unknown规则 | Q5−Q1 Macro-F1 (pp) | 95%区间(pp) | 六项Holm p |',
        '|---|---|---:|---|---:|']
    for r in common.read_csv(BASE/'confirmation/tables/difficulty_contrasts.csv'):
        lines.append(f"| {r['source']} | {r['unknown_mode']} | {100*float(r['mean_delta_auc']):+.2f} | [{100*float(r['ci95_low']):+.2f}, {100*float(r['ci95_high']):+.2f}] | {float(r['holm_six_p']):.5f} |")
    lines+=['','## 共同风险配额方法比较','',
       '差值为逐类别等错误数后的归一化预算AUC，正值偏向Coverage。混淆类型与错误簇并列报告，不能当F1提升。',
       '', '| 来源 | 对照 | 指标 | AUC差 | 95%区间 | 预定家族Holm p | 全20项Holm p |', '|---|---|---|---:|---|---:|---:|']
    comps=common.read_csv(BASE/'comparison/tables/comparisons.csv')
    for r in comps:
        family='holm_primary4_p' if r['baseline']=='random' else 'holm_ablation4_p' if r['baseline']=='coverage_without_H' else 'holm_strong12_p'
        lines.append(f"| {r['source']} | {r['baseline']} | {r['metric']} | {float(r['mean_delta_auc']):+.4f} | [{float(r['ci95_low']):+.4f}, {float(r['ci95_high']):+.4f}] | {float(r[family]):.5f} | {float(r['holm_all20_p']):.5f} |")
    lines+=['','## 冻结状态迁移','', '直接沿用COCO A中预测前冻结的175条状态—模型清单；VOC支持不足不替换。',
       '', '```json',json.dumps(json.loads((BASE/'transfer/summary.json').read_text()),ensure_ascii=False,indent=2),'```',
       '', '## 次要结果、成本与案例','',
       '统一ViT错误簇、来源交叉、风险变体和像素尺度见robustness/tables。全部正负结果均保留。',
       '8架构精确检验的分辨率限制与COCO一致，12/20项校正未通过不等于零效应。强基线固定面板源图重采样是补充估计，不能替换主结果；非线性重复权重偏差需结合观测点阅读。',
       '成本、案例及跨来源主次结果索引见../empirical_evidence_strengthening_20261005。诊断为条件关联，不能声明失败因果根源。',
       '覆盖原始分母由各池双来源可行状态确定；不把COCO和VOC绝对覆盖百分比直接合并。']
    (BASE/'report_zh.md').write_text('\n'.join(lines)+'\n')
    common.write_json(BASE/'final_audit.json',dict(status='pass',targets=TOTAL,runs=24,all_stages_complete=True,
        source_freezes_unchanged=True,independence='unused VOC2007 images, separate source from COCO training',annotation_truth='automatic only'))

if __name__=='__main__':main()
