#!/usr/bin/env python3
"""Summarize all frozen-prediction budgets without borrowing earlier findings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/QE_ThreeStrategy_Full_20261013'
CLASSES = ('dog', 'cat', 'horse', 'sheep', 'person', 'car')
METHODS = ('coverage', 'risk', 'risk_coverage')
LABELS = {'coverage': 'Coverage', 'risk': 'Risk quota random', 'risk_coverage': 'Risk + coverage'}
COLORS = {'coverage': '#0072B2', 'risk': '#777777', 'risk_coverage': '#D55E00'}
NAMES = {'coverage': '覆盖选样', 'risk': '风险配额随机', 'risk_coverage': '风险＋覆盖'}

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def number(row, key):
    return float(row[key])

def md_table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]

def docx_from_md(text, path, figures):
    doc = Document()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            cells = [[x.strip() for x in row.strip('|').split('|')] for row in block]
            table = doc.add_table(rows=1, cols=len(cells[0])); table.style = 'Table Grid'
            for cell, value in zip(table.rows[0].cells, cells[0]): cell.text = value
            for row in cells[2:]:
                for cell, value in zip(table.add_row().cells, row): cell.text = value
            continue
        if line.startswith('#'):
            doc.add_heading(line.lstrip('#').strip(), level=min(len(line)-len(line.lstrip('#')), 3)-1)
        elif line:
            doc.add_paragraph(line)
        i += 1
    for figure in figures:
        doc.add_paragraph(figure.stem)
        doc.add_picture(str(figure), width=__import__('docx').shared.Inches(6.2))
    doc.save(path)

def plot_curves(out, summary, metrics, name):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 3.0*len(metrics)), squeeze=False)
    for col, source in enumerate(('local', 'online')):
        for row, (key, label, scale) in enumerate(metrics):
            ax = axes[row, col]
            for method in METHODS:
                vals = sorted([r for r in summary if r['source'] == source and r['method'] == method], key=lambda r: int(r['budget_per_class']))
                x = [int(v['total_targets']) for v in vals]
                y = [scale*number(v, key) for v in vals]
                ax.plot(x, y, marker='o', markersize=3, color=COLORS[method], label=LABELS[method])
                if method == 'risk' and key+'_selection_p025' in vals[0]:
                    ax.fill_between(x, [scale*number(v, key+'_selection_p025') for v in vals], [scale*number(v, key+'_selection_p975') for v in vals], color=COLORS[method], alpha=.13)
            ax.set_title(source if row == 0 else '')
            ax.set_ylabel(label); ax.set_xlabel('Total selected targets'); ax.grid(alpha=.2)
            ax.set_xlim(0, max(x)*1.02)
            if row == 0: ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png', 'svg'): fig.savefig(out/'figures'/f'{name}.{suffix}', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); out = args.output
    summary = read_csv(out/'summary.csv')
    runs = read_csv(out/'model_run_metrics.csv')
    tests = read_csv(out/'primary_tests.csv')
    budgets = sorted({int(r['budget_per_class']) for r in summary})
    assert budgets == [20, 40, 100, 200, 400, 600, 800, 1000, 1200, 1300]
    assert len(summary) == 2*3*len(budgets)
    assert len(runs) == len(summary)*24
    indexed = {(r['source'], int(r['budget_per_class']), r['method']): r for r in summary}
    assert len(indexed) == len(summary)
    input_paths = [out/'summary.csv', out/'model_run_metrics.csv', out/'primary_tests.csv', out/'paired_model_run_metrics.csv']
    frozen_inputs = {str(p): sha(p) for p in input_paths}
    reference_path = out/'independent_full_pool_reference.json'
    assert reference_path.is_file(), 'Require independently computed full-pool reference.'
    reference = json.loads(reference_path.read_text())
    frozen_inputs[str(reference_path)] = sha(reference_path)
    full_metrics = []
    full_equality = []
    full_keys = ('presence_coverage', 'support_completion', 'other_label_support_completion', 'risk_lower', 'risk_midpoint', 'risk_upper', 'risk_unknown_fraction', 'accuracy', 'macro_f1', 'errors', 'confusion_type_count', 'confusion_breadth') + tuple(c+'_recall' for c in CLASSES)
    for source in ('local', 'online'):
        ref = indexed[source, 1300, 'coverage']
        for key in ('accuracy','macro_f1','errors','confusion_type_count'):
            assert np.isclose(number(ref,key),reference[key],atol=1e-9,rtol=0), (source,key)
        for key in ('presence_coverage','support_completion','risk_midpoint'):
            assert np.isclose(number(ref,key),reference['annotation_sources'][source][key],atol=1e-9,rtol=0), (source,key)
        for method in METHODS:
            row = indexed[source, 1300, method]
            for key in full_keys:
                assert np.isclose(number(row, key), number(ref, key), atol=1e-10, rtol=0), (source, method, key)
            full_metrics.append(dict(source=source, method=method, total_targets=7800, **{key:number(row,key) for key in full_keys}))
            full_equality.append(dict(source=source, method=method, all_full_metrics_identical=True))
    write_csv(out/'full_pool_metrics.csv', full_metrics)
    efficiencies = []
    for source in ('local', 'online'):
        full = indexed[source, 1300, 'coverage']
        for method in METHODS:
            vals = [indexed[source, b, method] for b in budgets]
            for key in ('support_completion', 'presence_coverage'):
                endpoint = number(full,key)
                for fraction in (.95, 1.0):
                    passed = [r for r in vals if number(r,key) >= fraction*endpoint-1e-10]
                    first = passed[0]
                    efficiencies.append(dict(source=source, method=method, metric=key, fraction_of_full_pool_endpoint=fraction,
                                             full_pool_endpoint=endpoint, threshold_value=fraction*endpoint,
                                             first_observed_budget_per_class=int(first['budget_per_class']),
                                             first_observed_total_targets=int(first['total_targets']),
                                             first_observed_metric_value=number(first,key),
                                             saved_targets_vs_full_pool=7800-int(first['total_targets']),
                                             scope='first qualifying measured budget; random method uses mean over repeats, not guarantee for every repeat'))
    write_csv(out/'coverage_efficiency.csv', efficiencies)
    model_rows = []
    grouped = defaultdict(list)
    for row in runs: grouped[row['source'], int(row['budget_per_class']), row['method'], row['model']].append(row)
    for (source, budget, method, model), rows in sorted(grouped.items()):
        assert len(rows) == 3
        model_rows.append(dict(source=source, budget_per_class=budget, total_targets=6*budget, method=method, model=model, seeds=3,
                               **{key:float(np.mean([number(r,key) for r in rows])) for key in ('accuracy','macro_f1','errors','confusion_type_count','confusion_breadth')+tuple(c+'_recall' for c in CLASSES)}))
    write_csv(out/'model_summary.csv', model_rows)
    categories = [dict(source=r['source'], budget_per_class=int(r['budget_per_class']), total_targets=int(r['total_targets']), method=r['method'], category=c,
                       recall=number(r,c+'_recall'), selection_p025=r.get(c+'_recall_selection_p025',''), selection_p975=r.get(c+'_recall_selection_p975','')) for r in summary for c in CLASSES]
    write_csv(out/'category_recall_summary.csv', categories)
    candidates = [k for k in tests[0] if 'holm' in k.lower()]
    assert candidates, 'New runner must provide full-family Holm correction.'
    preferred = [k for k in candidates if '18' in k or 'eighteen' in k or k == 'holm_p']
    holm_key = (preferred or candidates)[0]
    nonfull_tests = [r for r in tests if int(r['budget_per_class']) != 1300]
    assert len(nonfull_tests) == 18
    test_index = {(r['source'],int(r['budget_per_class'])):r for r in tests}
    deltas = []
    for source in ('local','online'):
        for budget in budgets:
            left, right = indexed[source,budget,'risk_coverage'], indexed[source,budget,'risk']
            for key in ('risk_lower','risk_midpoint','risk_upper','risk_unknown_fraction'):
                assert np.isclose(number(left,key),number(right,key),atol=1e-9,rtol=0), (source,budget,key)
            test = test_index.get((source,budget), {})
            deltas.append(dict(source=source,budget_per_class=budget,total_targets=6*budget,
                               presence_coverage_delta_pp=100*(number(left,'presence_coverage')-number(right,'presence_coverage')),
                               support_completion_delta_pp=100*(number(left,'support_completion')-number(right,'support_completion')),
                               other_label_support_delta_pp=100*(number(left,'other_label_support_completion')-number(right,'other_label_support_completion')),
                               macro_f1_delta_pp=100*(number(left,'macro_f1')-number(right,'macro_f1')),
                               additional_errors=number(left,'errors')-number(right,'errors'),
                               additional_confusion_types=number(left,'confusion_type_count')-number(right,'confusion_type_count'),
                               equal_error_breadth_delta_pp=100*float(test.get('mean_equal_error_delta',0)),
                               holm_18_p=test.get(holm_key,''),positive_architectures=test.get('positive_architectures',''),
                               full_endpoint=budget==1300, correction_family='18 non-full source-budget tests' if budget!=1300 else 'full endpoint excluded'))
    write_csv(out/'combo_vs_risk_deltas.csv',deltas)
    out.joinpath('figures').mkdir(exist_ok=True)
    plot_curves(out,summary,[('presence_coverage','Fixed-dictionary presence coverage (%)',100),('support_completion','Support-cap20 completion (%)',100)],'coverage_all_budgets')
    plot_curves(out,summary,[('risk_midpoint','Mean risk score (interval midpoint)',1),('q5_fraction','Q5 fraction (%)',100)],'risk_all_budgets')
    plot_curves(out,summary,[('accuracy','Accuracy (%)',100),('macro_f1','Macro-F1 (%)',100)],'model_all_budgets')
    plot_curves(out,summary,[('errors','Errors found (24-run average)',1),('confusion_type_count','Distinct confusion types (24-run average)',1)],'diagnostics_all_budgets')
    figures = sorted((out/'figures').glob('*all_budgets.png'))
    lines = ['# QE项目：三种测试选样策略的全量预算对照','',
             '本轮以同一7800目标候选池开展实验，六类别各1300目标。分别使用local与online两套已有VLM标注，固定每类71个预定义非unknown状态、总426状态；逐项出现覆盖与每状态20个证据的支持完成度分开计算。两标注源分开评价，不拼接标签制造100%覆盖。','',
             '三个策略为无风险配额的覆盖贪心、风险配额分层无放回随机、同风险配额下的覆盖贪心。每类预算为20、40、100、200、400、600、800、1000、1200、1300，分别对应120至7800个目标。非全池的风险随机组重复1000次；全池只有一种目标集合，因此仅保留一次。所有模型指标复用8架构×3训练种子的24份既有分类预测，没有新模型推理、训练或微调。','',
             '为与上一轮一致，报告和风险曲线的主要风险分使用区间中点。完整CSV同时保留下界、中点、上界和未知比例；中点是固定展示规则，不代表未知状态已经确定或其真实风险期望，不确定性并未消除。','',
             '## 全量终点：三种方法必然汇合','',
             '当每类预算达到1300，三个方法均选入整个候选池，目标集合、覆盖、风险、各模型指标一致。此时没有继续选样的空间，覆盖方法的优势不可能在终点继续增大。相同模型在其他子集上的F1不同，说明测试条件不同，不表示模型参数变好或变差。','']
    endpoint_rows=[]
    for source in ('local','online'):
        row=indexed[source,1300,'coverage']
        endpoint_rows.append([source,'7800',f"{number(row,'presence_coverage'):.4%}",f"{number(row,'support_completion'):.4%}",f"{number(row,'risk_midpoint'):.4f}",f"{number(row,'macro_f1'):.4%}"])
    lines += md_table(['标注源','目标数','固定字典出现覆盖','支持完成度','平均风险分（中点）','24模型平均Macro-F1'],endpoint_rows)
    lines += ['','全池覆盖值是现有候选池可达到的上限，未出现状态仍保留在426分母中。达到全部可达状态的100%不等于预定义字典100%，更不等于特征组合全部覆盖或部署充分性。','', '## 覆盖效率：达到全池支持完成度的比例','',
              '下表的95%与100%相对于各标注源的全池支持完成度终点，不相对于绝对100%覆盖。只报告10个实测预算点中首次达标的位置，不能推出中间未测试预算的最小需求。风险随机组使用1000次抽样均值，不能保证每次抽样都达标。','']
    efficiency_rows=[]
    for source in ('local','online'):
        for method in METHODS:
            items={r['fraction_of_full_pool_endpoint']:r for r in efficiencies if r['source']==source and r['method']==method and r['metric']=='support_completion'}
            efficiency_rows.append([source,NAMES[method],items[.95]['first_observed_total_targets'],items[1.0]['first_observed_total_targets']])
    lines += md_table(['标注源','策略','首次达到全池支持完成度95%的目标数','首次达到全池支持完成度100%的目标数'],efficiency_rows)
    lines += ['','## 控制风险后的诊断增益','',
              '组合组与风险随机组在每个预算下具有相同的已测风险分区间均值及未知比例；因此可比较增加覆盖后的附加诊断价值。源数据集构成及未测语义因素仍可能不同，不作“只改变覆盖”的因果解释。无约束覆盖组的风险构成不同，不能仅凭发现更多错误断言其选样更有效。','',
              '主检验比较组合组与风险随机组在相同错误数下的混淆类型广度，先在架构内平均三个种子，再以8架构开展精确符号翻转检验。18个非全池来源—预算检验统一做Holm校正；全池终点差异为0，不纳入该检验族。这里使用本轮结果，未沿用此前6个预算设置的校正p值。','']
    delta_rows=[]
    for r in deltas:
        p = '' if r['holm_18_p']=='' else f"{float(r['holm_18_p']):.6f}"
        delta_rows.append([r['source'],r['total_targets'],f"{r['support_completion_delta_pp']:+.3f}",f"{r['additional_errors']:+.3f}",f"{r['equal_error_breadth_delta_pp']:+.3f}",p or '终点不检验'])
    lines += md_table(['标注源','目标数','支持完成度增益/百分点','额外发现错误/24模型平均','等错误数广度增益/百分点','Holm校正p'],delta_rows)
    sig=[r for r in deltas if not r['full_endpoint'] and r['holm_18_p']!='' and float(r['holm_18_p'])<.05]
    pos=[r for r in sig if r['equal_error_breadth_delta_pp']>0]
    lines += ['',f'本轮18项检验中，{len(sig)}项达到Holm校正p＜0.05，其中{len(pos)}项为组合组的正向广度差异。结论仅适用于本候选池、预算和混淆类型代理指标；不把覆盖率或原始错误数增加直接等同于稳健优越。','',
              '检验分辨率存在明确限制：以8架构为独立统计单位的双侧精确符号翻转检验，最小可能p值为2/256＝0.0078125，而18项Holm检验第一步要求p≤0.05/18＝0.0027778。因此本轮检验设计即使全部架构正向，也无法得到0.05层面的校正显著性。“未显著”不能解释为已经证明没有诊断收益；应报告效应量与方向，并在后续预先设计更多独立评估单位或更集中的确认性问题。不会为了获得显著性事后缩小本轮检验族。','',
              '## 完整结果与停止规则','',
              '各策略按明确预算停止；候选池耗尽时停止。本轮不包含覆盖达标后的自动提前停止。达到全池支持完成度之前存在稀缺状态；该完成度达到终点之后新增目标可能增加样本数量和错误总数，但不能再提高已封顶的支持指标。','',
              'summary.csv保存全部60个来源—预算—方法配置；model_summary.csv保存每架构跨种子均值；category_recall_summary.csv保存各类别Recall；full_pool_metrics.csv保存全量终点；coverage_efficiency.csv保存实测预算效率；combo_vs_risk_deltas.csv保存风险匹配后的差异；selection与test_sets保存可复用的选样索引和目标manifest。图中阴影是风险随机组选样的2.5%—97.5%分位范围，不是模型总体性能的置信区间。','',
              '## 全预算覆盖及模型指标','']
    all_rows=[[r['source'],int(r['total_targets']),NAMES[r['method']],f"{number(r,'presence_coverage'):.3%}",f"{number(r,'support_completion'):.3%}",f"{number(r,'risk_midpoint'):.3f}",f"{number(r,'macro_f1'):.3%}"] for r in sorted(summary,key=lambda r:(r['source'],int(r['budget_per_class']),r['method']))]
    lines += md_table(['标注源','目标数','策略','出现覆盖','支持完成度','风险分（中点）','Macro-F1'],all_rows)
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text)
    docx_from_md(text,out/'report.docx',figures)
    claims={'full_endpoint_identity':{'evidence':'full_pool_metrics.csv; all 18 common metrics + six recalls equality assertions','status':'verified'},
            'efficiency':{'evidence':'coverage_efficiency.csv','scope':'measured budget points only; risk method means'},
            'new_multiple_testing':{'evidence':'primary_tests.csv','family':18,'holm_field':holm_key,'significant_tests':len(sig)},
            'no_new_inference':{'evidence':'prediction_input_freeze.sha256.json and runner protocol; existing predictions reused'},
            'frozen_summary_inputs':frozen_inputs}
    (out/'full_report_claim_evidence.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n')
    assert all(sha(Path(p))==v for p,v in frozen_inputs.items())
    generated=[out/name for name in ('report.md','report.docx','full_pool_metrics.csv','coverage_efficiency.csv','model_summary.csv','category_recall_summary.csv','combo_vs_risk_deltas.csv','full_report_claim_evidence.json')]+list((out/'figures').glob('*all_budgets.*'))
    hashes={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),**frozen_inputs,**{str(p):sha(p) for p in generated}}
    (out/'full_report_freeze.sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    audit={'status':'pass','summary_settings':len(summary),'model_run_rows':len(runs),'architecture_summary_rows':len(model_rows),'new_holm_family':18,'full_pool_methods_equal':full_equality,'report_generated':True}
    (out/'full_report_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    hashes[str(out/'full_report_audit.json')] = sha(out/'full_report_audit.json')
    (out/'full_report_freeze.sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__': main()
