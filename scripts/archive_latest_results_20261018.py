#!/usr/bin/env python3
"""Create a verified, writing-oriented snapshot of the latest result sets."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
BASE=ROOT/'results/最新实验结果汇总_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=BASE
    if out.exists():out=out.with_name(out.name+'_'+datetime.now().strftime('%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    records=[]

    def copy_file(source,target):
        target=out/target;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        original=sha(source);copied=sha(target)
        assert original==copied,str(source)
        records.append(dict(归档路径=str(target.relative_to(out)),原始路径=str(source),字节数=target.stat().st_size,SHA256=copied))

    def copy_folder(source,target):
        for source_file in sorted(source.rglob('*')):
            if not source_file.is_file() or '__pycache__' in source_file.parts:continue
            if source_file.suffix in {'.pyc','.tmp'}:continue
            copy_file(source_file,Path(target)/source_file.relative_to(source))

    mappings={
        'other_classes_specific_experiment_20261018':'01_四类专属特征最新实验',
        'cat_joint_recent_five_20261018':'02_猫类共同分层_五模型_共享71状态',
        'cat_joint_stratification_20261018':'03_猫类共同分层_八模型_共享71状态',
        'dog_cat_f1_zoom_20261018':'04_狗猫早期单类对照_历史参考',
        'other_classes_specific_coverage_v2_20261018':'05_特征字典/当前确认版',
        'other_classes_llm_feature_definition_20261018':'05_特征字典/定义来源与修订记录',
        'coco_five_dimension_fine_selection_20261015':'06_基础数据与模型预测/共享状态实验基准',
    }
    for source,target in mappings.items():
        copy_folder(ART/source,target)
        print('[copied]',source,flush=True)
    models=['yolo26s_cls','pe_core_b16','siglip2_base','mobileclip2_s0','aimv2_large']
    for model in models:
        for seed in [20260916,20260917,20260918]:
            source=ART/f'recent_five_models_20261009/runs/{model}/{seed}/B_outputs.npz'
            copy_file(source,Path('06_基础数据与模型预测/后续五模型预测')/model/str(seed)/source.name)
    scripts=[
        'run_other_classes_specific_experiment_20261018.py',
        'annotate_other_classes_specific_v2_20261018.py',
        'analyze_other_classes_specific_experiment_20261018.py',
        'build_specific_coverage_dictionary_20261018.py',
        'refine_specific_coverage_dictionary_20261018.py',
        'propose_other_class_fine_features_llm_20261018.py',
        'run_category_five_dimension_pilot.py',
        'run_coco_fine_coverage_experiment_20261015.py',
        'run_qe_three_strategy_full_experiment.py',
        'archive_latest_results_20261018.py',
    ]
    for name in scripts:copy_file(ROOT/'scripts'/name,Path('07_程序与配置/scripts')/name)
    for name in ['category_five_dimension_pilot_v1.json','recent_five_models_20261009.json']:
        copy_file(ROOT/'configs'/name,Path('07_程序与配置/configs')/name)
    for source in sorted(ART.glob('other_classes_specific_experiment_20261018*.log')):
        copy_file(source,Path('08_运行日志')/source.name)
    figures={
        '01_四类专属特征最新实验':['四类_五模型_Q1-Q5_F1','四类_Q1-Q5_累计覆盖率'],
        '02_猫类共同分层_五模型_共享71状态':['猫类_后续五模型_共同分层_Q1-Q5_F1','猫类_Q1-Q5_累计覆盖率曲线','猫类_覆盖率提升_同配额对照'],
    }
    for folder,names in figures.items():
        for name in names:
            for ext in ['png','svg']:
                copy_file(out/folder/f'{name}.{ext}',Path('09_写作常用图')/f'{name}.{ext}')
    data=pd.read_csv(out/'01_四类专属特征最新实验/final_summary.csv')
    coverage=pd.read_csv(out/'01_四类专属特征最新实验/coverage_summary.csv')
    names=dict(horse='马',sheep='羊',person='人',car='汽车')
    rows=[]
    for _,r in data.iterrows():
        c=r.class_name
        x=coverage.query('class_name == @c and namespace == "all" and method == "expanded_greedy"').iloc[0]
        rows.append(dict(类别=names[c],候选数=400,入选数=200,每层入选数=40,共享状态数=71,
            专属状态数=int(r.specific_states),总状态数=int(r.total_states),
            Q1类别F1百分比=r.Q1_mean_F1_pct,Q5类别F1百分比=r.Q5_mean_F1_pct,
            F1下降百分点=-r.F1_delta_pp,出现覆盖率百分比=x.presence_pct,
            出现覆盖提升百分点=r.presence_gain_pp,支持完成度百分比=x.support_completion_pct,
            支持完成度提升百分点=r.support_completion_gain_pp,
            共享支持较旧选样变化百分点=r.shared_support_change_vs_old_pp,
            专属字段平均未知百分比=r.pool_unknown_pct))
    pd.DataFrame(rows).to_csv(out/'写作汇总表_四类最新结果.csv',index=False,encoding='utf-8-sig')
    intro='''# 最新实验结果汇总（2026-10-18）

优先阅读[写作结果摘要](写作结果摘要.md)，图表集中在[09_写作常用图](09_写作常用图)。四类最新数值见[写作汇总表](写作汇总表_四类最新结果.csv)。

## 目录与使用顺序

| 目录 | 保存内容 | 写作口径 |
|---|---|---|
| 01_四类专属特征最新实验 | 马、羊、人、汽车的完整新实验；图、表、1600目标标注、原始响应、证据图、配额、入选名单和核验记录 | 本轮主结果；共享71状态＋各类专属状态 |
| 02_猫类共同分层_五模型_共享71状态 | 猫类后续五模型、共同分层/固定负类对照、累计覆盖及1000次随机对照 | 前一版共享71状态实验，不能写成本轮专属字典结果 |
| 03_猫类共同分层_八模型_共享71状态 | 旧八模型的猫类共同分层结果 | 与五模型结果分开注明模型集合 |
| 04_狗猫早期单类对照_历史参考 | 早期狗猫图及绘图数据 | 原图指标为Macro-F1，不能写成最新单类别F1 |
| 05_特征字典 | 当前确认版、LLM原稿、助手修订、原风险锁定核验 | 说明字典如何形成及哪些字段仅参与覆盖 |
| 06_基础数据与模型预测 | 旧共享状态基准与五模型各3种子的B预测 | 追溯目标、风险原始字段和冻结预测 |
| 07_程序与配置 | 本轮标注/选样/统计/绘图脚本、评分程序与配置 | 保留执行逻辑快照 |
| 08_运行日志 | 试跑、修复、逐目标标注、统计日志 | 试跑失败记录保留，正式结果排除了首轮多目标试跑 |
| 09_写作常用图 | 最新四类及猫类常用PNG/SVG | SVG可编辑，PNG用于预览 |

## 使用说明

本目录是结果副本，来源和逐文件SHA256见归档文件清单.csv；原项目结果保持原位置。新建的写作摘要与汇总表可以直接用于查数，但论文表述应保留实验边界。

本轮主实验对四类新增自动细粒度标注，模型阶段为冻结预测复评。狗、猫在本轮六类共同评测中仍沿用共享字典选样，不能写成六类均完成同一版专属扩展。F1按独立Q层计算；累计覆盖则保留此前层的全部目标。

复现脚本保留原项目路径与依赖。清单中的原始绝对路径、JSON审计中的原始路径用于追溯；移动文件夹后不会自动改写。文件夹保存结果和预测，不包含完整COCO原图、模型权重或Python环境，不能称为完全独立的运行环境。
'''
    (out/'00_开始这里.md').write_text(intro)
    summary=['# 写作结果摘要','',
        '## 方法与设置','',
        '在固定的COCO候选池中，对马、羊、人、汽车四类各400个目标进行逐目标自动细粒度视觉状态标注。类别专属特征用于覆盖编码和选样；风险评分沿用尺度、可见性、遮挡、截断、朝向五维等权规则。各类别内部按风险中点排序并五等分，未知风险保留上下界，风险同分时按目标ID处理。每类选取200个目标，各风险层40个，并保持风险分数与不确定性联合单元的原配额。覆盖对照为1000次同预算、同风险单元配额的随机选样。','',
        '评测沿用五个分类模型（YOLO26s、PE-Core-B16、SigLIP2-Base、MobileCLIP2-S0、AIMv2-Large）各三个种子的冻结预测。将六类中同一风险层的目标合并为平衡评测集，从六分类混淆矩阵计算每类F1；先按种子计算，再取三种子均值。下表再对五个模型均值取算术平均。','',
        '## 可直接查阅的结果','',
        '| 类别 | Q1 F1 | Q5 F1 | 下降(pp) | 出现覆盖 | 相对随机提升(pp) | 支持完成度 | 相对随机提升(pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        summary.append(f"|{r['类别']}|{r['Q1类别F1百分比']:.2f}%|{r['Q5类别F1百分比']:.2f}%|{r['F1下降百分点']:.2f}|{r['出现覆盖率百分比']:.2f}%|{r['出现覆盖提升百分点']:+.2f}|{r['支持完成度百分比']:.2f}%|{r['支持完成度提升百分点']:+.2f}|")
    summary+=['',
        '在四类、五模型共20条三种子均值曲线上，Q5的F1均低于Q1，但部分中间层发生回升，因此可写“总体下降”，不宜写“严格单调下降”。相对同风险配额随机对照，四类的出现覆盖及支持完成度均有提高。','',
        '加入专属状态后，相对旧的仅共享状态驱动选样，马、羊、人、汽车分别替换59、42、62、57个入选目标。专属支持完成度提高，但共享状态支持完成度分别下降1.27、0.56、2.39、1.27个百分点，需说明这一取舍。羊类专属出现覆盖相对旧选样没有增加，其收益主要体现在状态支持数量。','',
        '## 写作时保留的边界','',
        '- 细粒度状态由模型自动标注，未测量人工一致性或标注准确率；用户确认的是特征定义与取值。',
        '- 分类器阶段是冻结预测重新统计，不是本轮重新训练或推理。',
        '- 各Q的F1独立计算；覆盖曲线为Q1至Qk累计，两者横轴含义不同。',
        '- 出现覆盖为状态至少出现1次；支持完成度为平均min(状态样本数/20,1)，不等于达到20例的状态比例。',
        '- 累计覆盖非下降是固定字典下的性质；方法收益应依据同预算、同配额对照，不能只凭曲线上升。',
        '- 各类字典状态数不同，不能仅按覆盖率绝对值跨类排名；也不能与旧71状态覆盖率直接比较增减。',
        '- F1下降说明这些风险层在当前模型上的性能差异，单凭下降不能证明测试集质量更高，也未给出显著性结论。',
        '- 猫类旧五模型/八模型实验及早期狗猫Macro-F1图可作背景对照，引用时注明其模型、字典和指标版本。','',
        '## 数字来源','',
        '- 01_四类专属特征最新实验/final_summary.csv：端点F1、覆盖提升及专属/共享取舍。',
        '- 01_四类专属特征最新实验/coverage_summary.csv：覆盖率、支持完成度及达到K阈值比例。',
        '- 01_四类专属特征最新实验/model_F1_by_seed.csv、confusion_matrices.json：逐模型、逐种子结果。',
        '- 01_四类专属特征最新实验/coverage_cumulative_Q1_Q5.csv：累计覆盖图数据。',
        '- 原始claim_evidence.md、research_log.md、final_audit.json一并归档，记录来源与核验。','']
    (out/'写作结果摘要.md').write_text('\n'.join(summary))
    for name in ['00_开始这里.md','写作结果摘要.md','写作汇总表_四类最新结果.csv']:
        p=out/name;records.append(dict(归档路径=name,原始路径='本次根据已归档结果整理',字节数=p.stat().st_size,SHA256=sha(p)))
    with (out/'归档文件清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for r in records:assert sha(out/r['归档路径'])==r['SHA256']
    audit=dict(status='pass',created_at=datetime.now().astimezone().isoformat(),folder=str(out),
        files_verified=len(records),bytes=sum(r['字节数'] for r in records),originals_preserved=True,
        method='copy then SHA256 equality; all archived payloads rechecked',
        manifest_sha256=sha(out/'归档文件清单.csv'))
    (out/'归档校验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
