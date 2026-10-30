#!/usr/bin/env python3
"""Report equal-risk comparisons, acknowledging identical greedy strategies."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize_qe_three_strategy_full_experiment import read_csv,write_csv,sha,md_table,docx_from_md

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/QE_ThreeStrategy_EqualRatio_20261013'
DATA=ROOT/'data/QE_VisualState_CoverRisk_v1_20261013'
SCORES=ROOT/'artifacts/QE_FeatureRisk_Diagnostics_20261013/target_scores.csv'
CLASSES=('dog','cat','horse','sheep','person','car')
DIMS=('scale','visibility','occlusion','truncation','orientation')
METHODS=('coverage','risk','risk_coverage')
NAMES={'coverage':'配额约束覆盖','risk':'配额约束随机','risk_coverage':'风险＋覆盖（同覆盖组）'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();out=args.output
    summary=read_csv(out/'summary.csv');tests=read_csv(out/'primary_tests.csv');runs=read_csv(out/'model_run_metrics.csv')
    budgets=sorted({int(r['budget_per_class']) for r in summary});assert budgets==[20,40,100,200,400,600,800,1000,1200]
    assert len(summary)==54 and len(tests)==18 and len(runs)==1296
    indexed={(r['source'],int(r['budget_per_class']),r['method']):r for r in summary}
    scores={(r['annotation_backend'],r['sample_id']):r for r in read_csv(SCORES)};assert len(scores)==15600
    input_paths=[out/'summary.csv',out/'primary_tests.csv',out/'model_run_metrics.csv',out/'protocol.json',out/'coverage_combination_equivalence.json',SCORES,DATA/'manifest.csv',out/'merged_predictions.npz']
    freeze={str(p):sha(p) for p in input_paths};generated=[]
    ratio_rows=[]
    for source in ('local','online'):
        for budget in budgets:
            greedy=indexed[source,budget,'coverage'];combo=indexed[source,budget,'risk_coverage'];risk=indexed[source,budget,'risk']
            for key in greedy:
                if key not in ('method',):assert greedy[key]==combo[key],(source,budget,key)
            for row in (greedy,combo,risk):
                for tier in range(1,6):assert np.isclose(float(row[f'q{tier}_fraction']),.2,atol=1e-12)
            for bound in ('lower','midpoint','upper'):
                assert np.isclose(float(greedy['risk_'+bound]),float(risk['risk_'+bound]),atol=1e-9,rtol=0)
            assert np.isclose(float(greedy['risk_unknown_fraction']),float(risk['risk_unknown_fraction']),atol=1e-12)
            path=out/'selection'/f'{source}_n{budget*6}.npz';freeze[str(path)]=sha(path)
            with np.load(path) as z:
                assert np.array_equal(z['coverage'],z['risk_coverage'])
                for method in METHODS:
                    selected=z[method];selected=selected[None] if selected.ndim==2 else selected
                    for ci,category in enumerate(CLASSES):
                        ratio_rows.append(dict(source=source,budget_per_class=budget,total_targets=6*budget,method=method,category=category,targets_per_repetition=budget,class_fraction=1/6,repetitions=len(selected),**{f'Q{tier}_targets_per_repetition':budget//5 for tier in range(1,6)},**{f'Q{tier}_fraction':.2 for tier in range(1,6)}))
    path=out/'equal_ratio_proportions.csv';write_csv(path,ratio_rows);generated.append(path)
    main_rows=[]
    for row in summary:
        main_rows.append({k:row[k] for k in ('source','budget_per_class','total_targets','method','repetitions','presence_coverage','support_completion','risk_lower','risk_midpoint','risk_upper','q5_fraction','risk_unknown_fraction','accuracy','macro_f1','errors','confusion_type_count')})
    path=out/'all_budget_results.csv';write_csv(path,main_rows);generated.append(path)
    with np.load(out/'merged_predictions.npz') as z:pred={k:z[k] for k in z.files}
    predindex={str(s):i for i,s in enumerate(pred['sample_ids'])}
    primarydir=out/'primary_1200';primarydir.mkdir(exist_ok=True);manifest_index=[];rep0_metrics=[];primarydim=[]
    for source in ('local','online'):
        for method in METHODS:
            src=out/'test_sets'/f'{source}_{method}_n1200.csv';freeze[str(src)]=sha(src);records=read_csv(src);assert len(records)==1200
            assert Counter(r['class_name'] for r in records)==Counter(dict.fromkeys(CLASSES,200))
            for r in records:
                actual=scores[source,r['sample_id']];assert r['annotation_backend']==source
                for bound in ('lower','midpoint','upper'):
                    r['score_'+bound]=actual['global_risk_'+bound]
                for key,value in actual.items():
                    if key.endswith('_label') or '_risk_' in key or '_contribution_' in key or key.endswith('_known_weight'):r[key]=value
                r['risk_tier']=actual['risk_tier'];r['risk_tier_integer']=actual['risk_tier_integer'];r['high_risk_q5']=str(int(actual['risk_tier_integer'])==5)
                r['dataset_root']=str(DATA);r['selection_repetition']='0';r['selection_kind']='one fixed greedy set' if method!='risk' else 'first of1000 random sets, not random mean';r['global_scoring_scheme']='equal_global'
            assert Counter((r['class_name'],int(r['risk_tier_integer'])) for r in records)==Counter({(c,t):40 for c in CLASSES for t in range(1,6)})
            path=primarydir/f'{source}_{method}.csv';write_csv(path,records);generated.append(path)
            npz=out/'selection'/f'{source}_n1200.npz'
            manifest_index.append(dict(source=source,method=method,path=str(path),targets=1200,repetition=0,selection_npz=str(npz),dataset_root=str(DATA),score_source=source,score_table=str(SCORES),risk_mean_requires_all_repetitions=method=='risk'))
            gi=[predindex[r['sample_id']] for r in records];true=np.array([int(r['label']) for r in records]);accuracy=[];f1=[];errors=[]
            for predicted in pred['predictions'][:,gi]:
                confusion=np.zeros((6,6),dtype=int);np.add.at(confusion,(true,predicted),1);diag=np.diag(confusion);den=confusion.sum(0)+confusion.sum(1)
                f1.append(np.divide(2*diag,den,out=np.zeros(6),where=den>0).mean());accuracy.append((predicted==true).mean());errors.append((predicted!=true).sum())
            layers=Counter(int(r['risk_tier_integer']) for r in records)
            rep0_metrics.append(dict(source=source,method=method,total_targets=1200,repetition=0,model_runs=24,accuracy=float(np.mean(accuracy)),macro_f1=float(np.mean(f1)),errors=float(np.mean(errors)),risk_lower=float(np.mean([float(r['score_lower']) for r in records])),risk_midpoint=float(np.mean([float(r['score_midpoint']) for r in records])),risk_upper=float(np.mean([float(r['score_upper']) for r in records])),Q5_targets=layers[5],Q5_fraction=layers[5]/1200,description='actual exported repetition0; risk random mean in all_budget_results is over1000 sets'))
            for dimension in DIMS:
                primarydim.append(dict(source=source,method=method,dimension=dimension,total_targets=1200,repetition=0,**{bound+'_points':float(np.mean([float(r[dimension+'_contribution_'+bound+'_0_20']) for r in records])) for bound in ('lower','midpoint','upper')},unknown_fraction=float(np.mean([float(r[dimension+'_known_weight'])<1 for r in records]))))
    for name,rows in [('primary_manifest_index.csv',manifest_index),('primary_1200_rep0_metrics.csv',rep0_metrics),('primary_1200_rep0_dimension_scores.csv',primarydim)]:
        path=out/name;write_csv(path,rows);generated.append(path)
    primarysummary=[r for r in main_rows if int(r['budget_per_class'])==200];path=out/'primary_1200_summary.csv';write_csv(path,primarysummary);generated.append(path)
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for col,source in enumerate(('local','online')):
        for row,(key,label,scale) in enumerate([('presence_coverage','Fixed-state presence coverage (%)',100),('support_completion','Support-cap20 completion (%)',100),('macro_f1','Macro-F1 (%)',100)]):
            ax=axes[row,col]
            for method,color,labelmethod in [('coverage','#D55E00','Quota coverage = risk + coverage'),('risk','#777777','Same quota random')]:
                values=[indexed[source,b,method] for b in budgets];x=[int(v['total_targets']) for v in values];y=[scale*float(v[key]) for v in values];ax.plot(x,y,'o-',markersize=3,color=color,label=labelmethod)
                if method=='risk' and key+'_selection_p025' in values[0]:ax.fill_between(x,[scale*float(v[key+'_selection_p025']) for v in values],[scale*float(v[key+'_selection_p975']) for v in values],color=color,alpha=.13)
            ax.set_title(source if row==0 else '');ax.set_ylabel(label);ax.set_xlabel('Total targets');ax.grid(alpha=.2)
            if row==0:ax.legend(fontsize=8)
    fig.suptitle('Two independent strategies; coverage and combination are identical',fontsize=12);fig.tight_layout(rect=(0,0,1,.97))
    out.joinpath('figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):
        path=out/'figures'/f'equal_ratio_two_independent_strategies.{ext}';fig.savefig(path,dpi=180);generated.append(path)
    plt.close(fig)
    lines=['# QE项目：三组相同类别与风险比例的公平测试','',
           '本轮重新构造所有集合：六类别等量，三组每个类别的Q1—Q5各20%。各组进一步共享风险分中点与未知维数联合格的精确配额。覆盖组与风险＋覆盖组在相同约束下使用相同贪心算法、同分规则，因此所有预算下有完全相同的目标、轨迹和指标。虽然保留三个组名，实际只有两种独立策略，不将重合曲线故意拆开，也不检验组合优于配额覆盖。','',
           '两组每一步允许选择的候选相同、计算覆盖增益的方法相同、增益相同时的选择规则也相同，所以每一步都会选到同一目标。保留三个组名用于展示用户要求的相同比例约束；若今后要比较三种独立策略，必须先定义真正不同的层内选样目标，而非只更换名称。本轮未另行发明第三种算法。','',
           '覆盖组和组合组都从满足共同配额的候选中，按封顶20证据的凹覆盖增益选样；风险组在相同格配额内无放回随机1000次。主预算为1200目标，六类各200，每类每层40，全体每层240。预算按规定数量停止，不是随机遇到什么就选什么。','',
           '两标注源分别评价，逐项状态覆盖分母固定426；评分采用五维各20分的equal_global，身体可见性与身体朝向直接计分，头部和腿部等细字段保留但没有直接进入本主评分。风险分中点用于主展示，同时保留上下界与未知比例；Q5为每类相对高分20%，不等于绝对高风险或真实失败概率。','',
           '## 主预算：1200目标的公平比较','',
           '风险组的下表覆盖与模型结果是1000次选样均值。六份导出的主测试集则各自对应实际第0次集合，primary_1200_rep0_metrics.csv独立重算其模型指标，不能用随机组均值冒充这份集合的表现。','']
    primarytable=[]
    for r in primarysummary:primarytable.append([r['source'],NAMES[r['method']],f"{100*float(r['presence_coverage']):.3f}",f"{100*float(r['support_completion']):.3f}",f"{float(r['risk_lower']):.3f}/{float(r['risk_midpoint']):.3f}/{float(r['risk_upper']):.3f}",f"{100*float(r['q5_fraction']):.1f}",f"{100*float(r['risk_unknown_fraction']):.3f}",f"{100*float(r['macro_f1']):.3f}",f"{float(r['errors']):.3f}"])
    lines+=md_table(['来源','组名','出现覆盖%','支持完成%','风险下界/中点/上界','Q5%','未知风险%','Macro-F1%','发现错误数'],primarytable)
    lines+=['','虽然各组总风险分区间及未知比例匹配，各维贡献仍可能不同。主集合逐维分数在primary_1200_rep0_dimension_scores.csv；中点展示不消除未知状态，且来源组成及未测条件可以不同，不作只改变覆盖的因果推断。','',
            '## 诊断结果与检验范围','',
            '复用既有8架构×3种子的24份分类预测，没有重新训练或模型推理。测试子集F1较低是测试条件的表现，不代表模型参数退化；发现更多错误也不能单独证明方法更有效。主检验使用风险匹配下的等错误数混淆广度，18个来源—预算检验统一Holm校正。覆盖组与组合组相同，不作为两个独立检验重复计算。','']
    testtable=[]
    for t in tests:
        cov=indexed[t['source'],int(t['budget_per_class']),'coverage'];rnd=indexed[t['source'],int(t['budget_per_class']),'risk']
        testtable.append([t['source'],int(t['budget_per_class'])*6,f"{100*(float(cov['support_completion'])-float(rnd['support_completion'])):+.3f}",f"{100*float(t['mean_equal_error_delta']):+.3f}",f"{float(t['holm_eighteen_p']):.6f}"])
    lines+=md_table(['来源','目标数','同配额贪心支持增益/百分点','等错误数混淆广度增益/百分点','18项Holm校正p'],testtable)
    sig=sum(float(t['holm_eighteen_p'])<.05 for t in tests)
    lines+=['',f'本轮校正后显著项为{sig}/18。8架构双侧精确符号翻转检验的最小p＝2/256＝0.0078125，高于18项Holm第一步阈值0.05/18＝0.0027778，因此本设计无法达到该校正显著性。未显著不代表无效；应描述效应量与方向，不能事后缩小检验族以追求显著。','',
            '公平修正后的结论是：同风险配额下，贪心覆盖选样与随机选样的覆盖和诊断差异可以比较；“覆盖”与“风险＋覆盖”在本定义下为同一算法，不能据此支撑三个独立方法的竞争，也不能声称组合优于配额覆盖。','',
            '## 全部已测预算','',
            '本轮预算每类20、40、100、200、400、600、800、1000、1200，仅含18个来源—预算设置、54个组名结果，没有7800全池选样终点。既有全池上限local出现覆盖95.5399%、支持完成89.5657%，online99.5305%与92.5469%，仅作为上一轮候选池参考，不是本轮新增测试集的结果。','']
    rows=[[r['source'],r['total_targets'],NAMES[r['method']],f"{100*float(r['presence_coverage']):.3f}",f"{100*float(r['support_completion']):.3f}",f"{float(r['risk_midpoint']):.3f}",f"{100*float(r['macro_f1']):.3f}"] for r in main_rows]
    lines+=md_table(['来源','目标数','组名','出现覆盖%','支持完成%','风险中点','Macro-F1%'],rows)
    lines+=['','primary_manifest_index.csv列出6份主预算实体manifest、标注源与NPZ位置；风险组全部1000次索引保存在selection/{source}_n1200.npz，不复制百万条目标明细。图中只显示两条独立曲线，阴影为随机选样2.5%—97.5%分位区间。']
    text='\n'.join(lines)+'\n';path=out/'report.md';path.write_text(text);generated.append(path);path=out/'report.docx';docx_from_md(text,path,[out/'figures/equal_ratio_two_independent_strategies.png']);generated.append(path)
    audit=dict(status='pass',summary_rows=54,ratio_table_rows=len(ratio_rows),independent_strategies=2,coverage_combination_identical_all18=True,all_three_Q1_to_Q5_equal20percent=True,all_three_exact_total_risk_matched=True,primary_manifests=6,primary_targets_per_manifest=1200,primary_per_class=200,primary_class_tier_targets=40,random_rep0_model_metrics_recomputed=True,new_full_pool_selected=False)
    path=out/'equal_ratio_report_audit.json';path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');generated.append(path)
    claims=dict(fairness='equal_ratio_proportions.csv and frozenselection quotas',equivalence='coverage_combination_equivalence.json;same orderedIDs verified',primary_rep0='primary_manifest_index.csv;primary_1200_rep0_metrics.csv;actual annotationsource scores verified',scope='two independent algorithms,54 group-name settings, finite pool, frozen predictions',statistics='primary_tests.csv Holm_eighteen_p18 test family;8unit resolution limitation',no_causal_claim=True)
    path=out/'equal_ratio_report_claim_evidence.json';path.write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n');generated.append(path)
    assert all(sha(Path(p))==v for p,v in freeze.items())
    freeze[str(Path(__file__).resolve())]=sha(Path(__file__).resolve());freeze.update({str(p):sha(p) for p in generated})
    (out/'equal_ratio_report_freeze.sha256.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
