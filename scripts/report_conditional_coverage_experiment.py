"""Aggregate executed data into bounded claims, tables, and a Chinese report."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import conditional_coverage_common as cc


LABELS={'R':'风险配额随机','G':'完整贪心','SR':'静态稀缺度','KC':'ViT k-center',
        'CS':'条件替换','AS':'持续替换'}


def save(out,name,frame):
    frame.to_csv(out/name,index=False,encoding='utf-8-sig')


def summarize(out):
    assert json.loads((out/'validation.json').read_text())['status']=='pass'
    plan=json.loads((out/'protocol.json').read_text())
    seeds=plan['phases']['retrospective']['selection_seeds']
    runs=pd.read_csv(out/'selection_runs.csv')
    timing=pd.read_csv(out/'timing_replays.csv')
    diag=pd.read_csv(out/'external_diagnostic_runs.csv')
    B=plan['statistics']['bootstrap_replicates']
    rng=np.random.default_rng(plan['statistics']['bootstrap_seed'])
    preparation=json.loads((out/'preparation_cost.json').read_text())
    calibration=preparation['calibration_seconds']
    loading=preparation['context_loading_seconds']
    # Technical replays summarized without treating identical choices as new quality runs.
    replay=timing.groupby(['class_name','method','seed'],as_index=False).agg(
        timing_replays=('seconds','size'),median_replay_seconds=('seconds','median'))
    save(out,'timing_replay_summary.csv',replay)
    summary=[];class_summary=[];complete=[]
    for method in cc.METHODS:
        part=runs[runs.method==method]
        for c in cc.old.CLASSES:
            p=part[part.class_name==c]
            class_summary.append(dict(class_name=c,method=method,quality_runs=len(p),
                pass_fraction=p.final_pass.mean(),occurrence_pct=100*p.all_occurrence.mean(),
                support_pct=100*p.all_support_completion.mean(),
                mean_swaps=p.accepted_swaps.mean(),median_seconds=p.seconds.median()))
        for seed,p in part.groupby('seed'):
            if method in ['G','SR','KC']:
                warm=sum(replay[(replay.method==method)&(replay.class_name==c)].median_replay_seconds.iloc[0]
                         for c in cc.old.CLASSES)
            else:
                warm=p.seconds.sum()
            # On this calibration pool G already produces W; do not charge a second G run.
            incremental_calibration=0. if method=='G' else calibration
            complete.append(dict(method=method,seed=int(seed),all_four_classes_pass=bool(p.final_pass.all()),
                warm_seconds=float(warm),first_task_component_accounting_seconds=float(warm+incremental_calibration+loading),
                mean_class_swaps=float(p.accepted_swaps.mean()),
                sum_swaps=int(p.accepted_swaps.sum()),mean_occurrence_pct=float(100*p.all_occurrence.mean()),
                mean_support_pct=float(100*p.all_support_completion.mean()),
                initial_all_four_pass=bool(p.initial_pass.fillna(False).all()) if method in ['R','CS','AS'] else ''))
        p=pd.DataFrame([r for r in complete if r['method']==method])
        summary.append(dict(method=method,name=LABELS[method],complete_quality_runs=len(p),
            joint_pass_fraction=p.all_four_classes_pass.mean(),
            mean_occurrence_pct=p.mean_occurrence_pct.mean(),mean_support_pct=p.mean_support_pct.mean(),
            mean_swaps_per_class=part.accepted_swaps.mean(),
            median_warm_seconds=p.warm_seconds.median(),
            median_first_task_component_accounting_seconds=p.first_task_component_accounting_seconds.median(),
            mean_pair_scores_per_class=part.pair_scores.mean(),
            mean_add_scores_per_class=part.add_candidate_scores.mean()))
    summary=pd.DataFrame(summary);complete=pd.DataFrame(complete)
    save(out,'method_summary.csv',summary)
    save(out,'class_method_summary.csv',pd.DataFrame(class_summary))
    save(out,'complete_selection_runs.csv',complete)
    costs=[]
    cs=complete[complete.method=='CS'].set_index('seed').loc[seeds]
    for comparator in ['G','AS','R','SR','KC']:
        other=complete[complete.method==comparator].set_index('seed')
        reference=(other.loc[seeds].warm_seconds.to_numpy() if comparator in ['AS','R'] else
                   np.full(len(seeds),other.warm_seconds.iloc[0]))
        ratio=cs.warm_seconds.to_numpy()/reference
        indices=rng.integers(0,len(seeds),size=(B,len(seeds)))
        bootstrap=np.median(ratio[indices],axis=1)
        low,high=np.quantile(bootstrap,[.025,.975])
        costs.append(dict(comparison=f'CS_minus_{comparator}',comparator=comparator,
            median_paired_warm_time_ratio=float(np.median(ratio)),
            median_warm_time_reduction_pct=float(100*(1-np.median(ratio))),
            ratio_bootstrap_p025=float(low),ratio_bootstrap_p975=float(high),
            interval_unit='selection_seed_conditional_on_fixed_pools_and_baseline_timing',
            passes_proposed_cost_gate=bool(np.median(ratio)<=.7 and high<1)))
    costs=pd.DataFrame(costs)
    save(out,'paired_cost_comparisons.csv',costs)
    # Timing robustness uses three total replays for the first ten seeds.
    robust=[]
    replaytot=replay.groupby(['method','seed']).median_replay_seconds.sum()
    for comparator in ['G','AS','R','SR','KC']:
        a=np.array([replaytot['CS',s] for s in seeds[:10]])
        b=(np.array([replaytot[comparator,s] for s in seeds[:10]]) if comparator in ['AS','R'] else
           np.full(10,replaytot[comparator,-1]))
        robust.append(dict(comparator=comparator,selection_seeds=10,replays_per_seed=3,
            median_paired_time_ratio=float(np.median(a/b)),
            median_time_reduction_pct=float(100*(1-np.median(a/b)))))
    save(out,'timing_robustness_comparisons.csv',pd.DataFrame(robust))
    amortized=[]
    for _,r in summary.iterrows():
        for tasks in [1,10,100]:
            amortized.append(dict(method=r.method,hypothetical_new_tasks=tasks,
                calibration_seconds=calibration,per_task_selection_seconds=r.median_warm_seconds,
                per_task_calibration_plus_selection=calibration/tasks+r.median_warm_seconds,
                observed_new_tasks=0,interpretation='scenario_not_observed_new_datasets'))
    save(out,'calibration_amortization_scenarios.csv',pd.DataFrame(amortized))
    cold=pd.read_csv(out/'cold_start_runs.csv')
    cold_summary=cold.groupby('method',as_index=False).agg(
        replays=('replay','size'),context_loading_seconds=('context_loading_seconds','median'),
        algorithm_seconds=('algorithm_seconds','median'),
        cold_pool_pipeline_seconds=('cold_pool_pipeline_seconds','median'),
        peak_process_RSS_KiB=('peak_process_RSS_KiB','median'))
    save(out,'cold_start_summary.csv',cold_summary)
    # Reference-image bootstrap: discovery selections/conditions remain fixed.
    with np.load(out/'reference_bootstrap_confirmed_counts.npz',allow_pickle=False) as z:
        boot={k:z[k].copy() for k in z.files}
    diagnostic_summary=[];diagnostic_differences=[];paired=[]
    for scope in ['all','specific']:
        method_boot={}
        method_point={}
        for method in cc.METHODS:
            p=diag[(diag.scope==scope)&(diag.method==method)]
            runs_boot=np.stack([boot[f'{method}__s{int(s)}__{scope}'] for s in p.seed])
            method_boot[method]=runs_boot.mean(0)
            method_point[method]=p.confirmed.mean()
            low,high=np.quantile(method_boot[method],[.025,.975])
            diagnostic_summary.append(dict(method=method,scope=scope,selection_runs=len(p),
                **{k:float(p[k].mean()) for k in ['proposed','evaluable','confirmed','not_confirmed',
                                                'insufficient','deduplicated_confirmed','evaluable_classes']},
                confirmation_fraction=float(p.confirmed.sum()/p.evaluable.sum()) if p.evaluable.sum() else '',
                reference_image_bootstrap_p025=float(low),reference_image_bootstrap_p975=float(high),
                bootstrap_interpretation='resampling_quantiles_not_calibrated_CI_due_distinct_image_support_gate',
                data_role='historical_reanalysis_not_new_confirmation'))
        for comparator in ['G','AS','R','SR','KC']:
            delta=method_boot['CS']-method_boot[comparator]
            low,high=np.quantile(delta,[.025,.975]);one_sided=np.quantile(delta,.05)
            diagnostic_differences.append(dict(scope=scope,comparator=comparator,
                point_difference=float(method_point['CS']-method_point[comparator]),
                reference_image_bootstrap_p025=float(low),reference_image_bootstrap_p975=float(high),
                one_sided_95_lower=float(one_sided),margin_minus_two_exceeded=bool(one_sided>-2),
                fresh_noninferiority_claim_allowed=False,formal_noninferiority_conclusion_allowed=False,
                reason='historical_reference_and_uncalibrated_discontinuous_support_bootstrap'))
            p=diag[(diag.scope==scope)&(diag.method=='CS')].set_index('seed').loc[seeds]
            q=diag[(diag.scope==scope)&(diag.method==comparator)].set_index('seed')
            for seed in seeds:
                ref=q.loc[seed if comparator in ['AS','R'] else -1]
                paired.append(dict(scope=scope,seed=seed,comparator=comparator,
                    confirmed_difference=float(p.loc[seed].confirmed-ref.confirmed),
                    proposed_difference=float(p.loc[seed].proposed-ref.proposed),
                    evaluable_difference=float(p.loc[seed].evaluable-ref.evaluable)))
    ds=pd.DataFrame(diagnostic_summary);dd=pd.DataFrame(diagnostic_differences)
    save(out,'external_diagnostic_summary.csv',ds)
    save(out,'paired_diagnostic_comparisons.csv',dd)
    save(out,'paired_differences.csv',pd.DataFrame(paired))
    # Preserve uncertainty from model and selection seeds as separate summaries.
    metrics=pd.read_csv(out/'model_metrics_by_seed.csv')
    metric_names=['precision','recall','f1','one_vs_rest_accuracy','top1_accuracy']
    for name in metric_names:
        metrics[name]=pd.to_numeric(metrics[name],errors='coerce')
    keys=['method','class_name','model','Q']
    by_selection=metrics.groupby(keys+['selection_seed'],as_index=False)[metric_names].mean()
    by_prediction=metrics.groupby(keys+['prediction_seed'],as_index=False)[metric_names].mean()
    save(out,'model_metrics_selection_seed_summary.csv',by_selection)
    save(out,'model_metrics_prediction_seed_summary.csv',by_prediction)
    mean=by_selection.groupby(keys)[metric_names].mean().add_suffix('_mean')
    sd=by_selection.groupby(keys)[metric_names].std().add_suffix('_selection_seed_sd')
    model_sd=by_prediction.groupby(keys)[metric_names].std().add_suffix('_prediction_seed_sd')
    quality_count=by_selection.groupby(keys).size().rename('selection_runs')
    save(out,'model_metrics_summary.csv',pd.concat([mean,sd,model_sd,quality_count],axis=1).reset_index())
    # Joint success is the conjunction over the four classes, not a pooled class rate.
    sensitivity=pd.read_csv(out/'sensitivity_runs.csv')
    sensitivity_settings=['axis','budget','support_cap','fraction','dictionary']
    sruns=sensitivity.groupby(sensitivity_settings+['method','seed'],as_index=False).agg(
        joint_pass=('final_pass','all'),warm_seconds=('seconds','sum'),
        mean_swaps=('accepted_swaps','mean'),mean_occurrence=('all_occurrence','mean'),
        mean_support=('all_support_completion','mean'))
    ss=sruns.groupby(sensitivity_settings+['method'],as_index=False).agg(
        complete_quality_runs=('seed','size'),joint_pass_fraction=('joint_pass','mean'),
        median_seconds=('warm_seconds','median'),mean_swaps=('mean_swaps','mean'),
        occurrence=('mean_occurrence','mean'),support=('mean_support','mean'))
    save(out,'sensitivity_summary.csv',ss)
    save(out,'sensitivity_complete_runs.csv',sruns)
    missing=pd.read_csv(out/'missingness_runs.csv')
    ms=missing.groupby(['class_name','missing_rate'],as_index=False).agg(
        repetitions=('repeat','size'),observed_pass_fraction=('final_pass','mean'),
        clean_occurrence=('clean_occurrence','mean'),clean_support=('clean_support','mean'))
    save(out,'missingness_summary.csv',ms)
    stop=runs.groupby(['class_name','method','stop_reason'],as_index=False).size()
    save(out,'stop_reasons.csv',stop)
    gates=dict(correctness='pass',coverage='pass' if summary.set_index('method').loc['CS','joint_pass_fraction']>=.95 else 'fail',
        warm_cost_vs_G='pass' if costs.set_index('comparator').loc['G','passes_proposed_cost_gate'] else 'fail',
        warm_cost_vs_AS='pass' if costs.set_index('comparator').loc['AS','passes_proposed_cost_gate'] else 'fail',
        fresh_diagnostic_noninferiority='not_evaluated_no_fresh_data',
        new_data_phase='not_started_P2_cost_gate_failed_and_no_frozen_fresh_pools',
        claim_scope='retrospective_fixed_pool_only')
    cc.dump(out/'decision_gate_results.json',gates)
    make_report(out,summary,ds,dd,costs,ss,gates)
    cc.log(out,'report_complete',decision_gates=gates)


def make_report(out,summary,ds,dd,costs,ss,gates):
    sm=summary.set_index('method');d=ds[ds.scope=='all'].set_index('method')
    specific=ds[ds.scope=='specific'].set_index('method')
    cp=costs.set_index('comparator');diff=dd[(dd.scope=='all')&(dd.comparator=='G')].iloc[0]
    lines=['# 风险先行、覆盖条件替换：实际实验结果',
      '',f'生成时间：{cc.old.now()}。数据、方法、停止规则均以本目录冻结协议为准。',
      '', '**结论：条件替换实现了覆盖达标，并比持续替换减少搜索；但在本次历史候选池、主阈值下，它没有比完整贪心更快，也没有在历史参考集上显示高于随机选样的平均诊断数量。当前证据不支持“同时省时且诊断更优”的完整主张。**',
      '', '## 1. 实际执行范围', '',
      '- 四个优化类别：马、羊、人、汽车，每类候选400、入选200；狗猫各固定200，最终每个测试集1200个目标。',
      '- 六方法；R/CS/AS各50个配对选样种子，G/SR/KC各一个确定性选集；共612个类别级主选样运行、153个完整六类测试集。',
      '- 预算/阈值/K/字典单因素敏感性4896个类别级运行；状态缺失扰动240个运行。',
      '- 5模型×3已有预测种子；全部复用绑定样本ID与标签的历史预测，没有新训练或新推理。',
      '- 历史Objects365参考集800个目标，按新同字段已知非状态对照规则重算；不能称为新数据独立确认。',
      '', '## 2. 主结果', '',
      '时间为四个优化类别的总选样时间。随机方法列50次完整运行的中位数；确定性基线列每类三次技术计时的中位数之和。覆盖是四类等权平均，达标率要求四类同时达标。', '',
      '| 方法 | 四类共同达标 | 出现覆盖 | 支持完成度 | 每类平均交换 | 总选样时间/ms | 历史确认条件实例 | 去重状态类型 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in cc.METHODS:
        r=sm.loc[method];q=d.loc[method]
        lines.append(f'| {method} {LABELS[method]} | {r.joint_pass_fraction:.0%} | {r.mean_occurrence_pct:.2f}% | {r.mean_support_pct:.2f}% | {r.mean_swaps_per_class:.2f} | {1000*r.median_warm_seconds:.2f} | {q.confirmed:.2f} | {q.deduplicated_confirmed:.2f} |')
    lines += ['', f'CS/G配对耗时比中位数为{cp.loc["G","median_paired_warm_time_ratio"]:.3f}，95%种子bootstrap区间为[{cp.loc["G","ratio_bootstrap_p025"]:.3f}, {cp.loc["G","ratio_bootstrap_p975"]:.3f}]。',
      f'CS相对AS的配对耗时降低中位数为{cp.loc["AS","median_warm_time_reduction_pct"]:.2f}%。这些区间条件于当前固定数据与基线计时，不是跨数据集总体区间。',
      '', '所有主设置R初始集合均未同时满足所属类别的两个覆盖阈值，因此CS在本主设置未获得“初始已达标、直接跳过”的机会。不能只凭停止机制预设一定加速。',
      '', '## 3. 历史诊断复核', '',
      '| 方法 | 提出 | 可评价 | 确认 | 支持不足 | 可评价中的确认比例 |',
      '|---|---:|---:|---:|---:|---:|']
    for method in cc.METHODS:
        r=d.loc[method]
        lines.append(f'| {method} | {r.proposed:.2f} | {r.evaluable:.2f} | {r.confirmed:.2f} | {r.insufficient:.2f} | {r.confirmation_fraction:.2%} |')
    lines += ['', f'CS−G的历史确认实例差为{diff.point_difference:.2f}条，参考图像配对bootstrap差值分布的2.5%—97.5%分位范围为[{diff.reference_image_bootstrap_p025:.2f}, {diff.reference_image_bootstrap_p975:.2f}]，5%分位数为{diff.one_sided_95_lower:.2f}。这些是重采样分位数，不能直接作为已校准的不劣性置信界。',
      '该统计按原图分层成簇重采样2000次，并重算20目标、20不同原图的支持条件；预测种子先平均，发现条件固定。支持不足在确认数量中贡献零。',
      '参考数据已在过去研究中分析过，所以本表只作回顾性比较。还发现预设统计的边界问题：有放回抽样会减少不同原图数量，20原图门槛使确认数产生系统性下降及不连续性；原始确认点估计甚至可能落在重采样分位范围之外。本轮保留原预设重采样输出、不事后更换有利统计，但不把该范围解释为覆盖率已验证的置信区间，也不据此作正式不劣性结论。新确认实验前须先验证该统计方案。',
      '确认实例包含跨模型重复，应同时查看去重状态类型。状态相关性也不等于状态独立导致错误；当前没有可支持因果解释的调整结果。',
      f'仅专属状态的次分析同样全部保留：CS平均确认{specific.loc["CS","confirmed"]:.2f}条，G为{specific.loc["G","confirmed"]:.2f}条，R为{specific.loc["R","confirmed"]:.2f}条，详见同一诊断主表的specific行。',
      '', '## 4. 敏感性结果', '',
      '下表只列CS；完整六方法及共同K20评价见原始CSV。预算变化始终沿用B200主阈值。', '',
      '| 变化 | 设置 | 四类共同达标 | 每类平均交换 | 总选样时间/ms |',
      '|---|---|---:|---:|---:|']
    for _,r in ss[ss.method=='CS'].iterrows():
        setting=({'budget':f'B={int(r.budget)}','threshold':f'W×{r.fraction:.0%}',
                  'support_cap':f'K={int(r.support_cap)}','dictionary':r.dictionary}[r.axis])
        lines.append(f'| {r.axis} | {setting} | {r.joint_pass_fraction:.0%} | {r.mean_swaps:.2f} | {1000*r.median_seconds:.2f} |')
    lines += ['', '敏感性计时按固定顺序单次执行，用于观察量级，主成本判断以随机顺序和技术重复的主实验为准。',
      '状态缺失扰动不是标注错误率；无编码有效值比例也不是语义准确率。自动标注缺少独立人工金标准。',
      '', '## 5. 成本口径和停止决定', '',
      '主表不含输入载入、历史字典构建或候选VLM标注。首次任务需加上阈值参照生成和输入载入，见preparation_cost.json、method_summary.csv。组件加总仅为估算；G在校准池上的本次运行已生成W，不重复收取第二次G费用。摊销1/10/100任务只是成本情景，不是实际完成了这些新数据任务。',
      '另在独立进程、首个固定种子上，随机方法顺序、每方法3次实际测量输入准备＋四类选样：cold_start_summary.csv。交换索引只向CS/AS计费，G/R/SR/KC不准备索引；该测量不含Python导入、结果核对或历史阈值校准，不能冒充完整API标注到推理的耗时。',
      '同样预算下，各方法分类预测数量相同，不能声称节省模型推理或候选标注。进程峰值RSS已保存，但不是每种算法独立的额外峰值内存，不能据此排名。',
      f'正式门槛：正确性{gates["correctness"]}；覆盖{gates["coverage"]}；相对G热启动成本{gates["warm_cost_vs_G"]}；相对AS热启动成本{gates["warm_cost_vs_AS"]}；新数据诊断不劣性未完成。',
      '本次未进入新C-select/C-ref采集标注和推理阶段：回顾性主成本门槛未通过，且没有符合冻结与隔离要求的两套新池。没有把历史数据改名为新验证。',
      '', '## 6. 可用结论与不可用结论', '',
      '- 可用：当前实现能保持预算和精细风险配额；在本次主设置达到覆盖阈值。',
      '- 可用：同搜索规则下，达标即停相对持续交换减少工作量；幅度只限本次数据和阈值。',
      '- 不支持：条件交换比完整贪心更省选样时间，或比随机选样发现更多可复现薄弱状态。',
      '- 未完成：在新数据上的诊断非劣性、总体测试充分性和标注语义准确率。',
      '- 不改变：模型权重未训练或修复，本实验不证明模型能力提高。',
      '', '## 7. 审核与复现', '',
      'validation.json记录了独立全量复算：预算与风险单元、交换轨迹、模型混淆矩阵、逐行指标、发现与历史参考条件。原始输入和最终集合保留SHA-256冻结。',
      '调试发现并修复了低精度log排序、背景索引和记录字段问题；修复均发生在主运行前，失败准备目录保留，主实验未读取预测来调整算法。',
      '完整复现命令见reproduce.sh；已冻结输出不能原地覆盖，重跑请设置一个新目录。',
      '', '## 8. 文件与图', '',
      f'- [实验协议]({out}/protocol.json)',
      f'- [完整核验]({out}/validation.json)',
      f'- [六方法主表]({out}/method_summary.csv)',
      f'- [历史诊断主表]({out}/external_diagnostic_summary.csv)',
      f'- [分类指标汇总]({out}/model_metrics_summary.csv)',
      f'- [敏感性汇总]({out}/sensitivity_summary.csv)',
      f'- [研究日志]({out}/research_log.md)',
      '',f'![覆盖、成本与历史诊断]({out}/figures/coverage_cost_diagnostics.svg)',
      '',f'![四指标分层表现]({out}/figures/four_metrics_grid.svg)',
      '',f'![敏感性]({out}/figures/sensitivity.svg)']
    (out/'report_zh.md').write_text('\n'.join(lines)+'\n')
    report_path=cc.ROOT/'docs/风险先行条件覆盖替换_实验结果_20261105.md'
    shutil.copyfile(out/'report_zh.md',report_path)
    claims=['# Claim–evidence ledger','',
      '| ID | 主张 | 状态 | 证据与边界 |','|---|---|---|---|',
      '| C1 | 实现保持预算和精细风险配额 | supported | validation.json；当前输入和实现 |',
      '| C2 | CS主设置覆盖达标 | supported | method_summary.csv；四个固定类别、50种子 |',
      '| C3 | CS比AS省搜索时间 | supported | paired_cost_comparisons.csv；相同S0与接受规则 |',
      '| C4 | CS比G省时间 | contradicted | paired_cost_comparisons.csv；本次主设置 |',
      f'| C5 | CS比R有更高历史确认数量 | unsupported | external_diagnostic_summary.csv；{d.loc["CS","confirmed"]:.2f} vs {d.loc["R","confirmed"]:.2f}，无平均优势 |',
      '| C6 | CS在新数据保持诊断效用 | missing | 无新C-select/C-ref，不以历史参考代替 |',
      '| C7 | 模型能力提高或标签准确 | unsupported | 未训练模型、无独立语义金标准 |']
    (out/'claim_evidence.md').write_text('\n'.join(claims)+'\n')
    (out/'review.md').write_text('# Result integrity review\n\n'
      '- Citation integrity: no new external literature claims introduced.\n'
      '- Result traceability: pass; see validation.json and raw CSV/NPZ outputs.\n'
      '- Claim scope: pass within historical fixed-pool scope; fresh generalization withheld.\n'
      '- Diagnostic bootstrap calibration: unresolved; distinct-image eligibility is discontinuous and resampling reduces unique images. Quantiles are descriptive, not validated confidence bounds; no formal noninferiority claim.\n'
      '- Reviewer role: same agent using independently implemented whole-set / prediction checks.\n'
      '- Outcome: proposed full efficiency-plus-diagnostic claim is unsupported; cost gate versus G failed.\n')


if __name__=='__main__':
    args=cc.parsed_args(__doc__).parse_args()
    summarize(args.out)
