#!/usr/bin/env python3
"""Post-hoc label sensitivity using immutable predictions, with explicit support gates."""
from collections import Counter
import itertools,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from automatic_strengthening_common import labels_for
from dual_vlm_common import CLASSES,MODELS,SEEDS,read_csv,write_csv,write_json,sha256,now
from analyze_sixclass_difficulty import risk,MAPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/label_disagreement_sensitivity_20261005'
PROTOCOL=ROOT/'docs/label_disagreement_sensitivity_protocol_20261005.md'
POOLS={'B':ROOT/'artifacts/sixclass_independent_confirmation_20261004',
       'C':ROOT/'artifacts/sixclass_voc_external_confirmation_20261005',
       'D':ROOT/'artifacts/objects365_independent_confirmation_20261005'}
GROUPS=('all','risk_agree','shared_agree')
LABELS=dict(zip(CLASSES,range(6)))


def values(v):return set(v) if isinstance(v,list) else {v}


def f1(y,p):
    confusion=np.bincount(6*y+p,minlength=36).reshape(6,6)
    diagonal=np.diag(confusion);denominator=confusion.sum(0)+confusion.sum(1)
    return float(np.divide(2*diagonal,denominator,out=np.zeros(6,dtype=float),where=denominator>0).mean())


def image_interval(y,pred,indices,masks,rng,draws=2000):
    """Joint full-pool resampling and filtering keeps shared images correlated."""
    full=[];subset=[]
    for _ in range(draws):
        sample={tier:np.concatenate([rng.choice(indices[tier][c],len(indices[tier][c]),replace=True) for c in CLASSES]) for tier in (1,5)}
        selected={tier:sample[tier][masks[sample[tier]]] for tier in (1,5)}
        if any(len(np.unique(y[selected[t]]))<6 for t in (1,5)):continue
        whole=np.mean([f1(y[sample[5]],p[sample[5]])-f1(y[sample[1]],p[sample[1]]) for p in pred])
        restricted=np.mean([f1(y[selected[5]],p[selected[5]])-f1(y[selected[1]],p[selected[1]]) for p in pred])
        full.append(whole);subset.append(restricted)
    if not subset:return dict(bootstrap_valid=0)
    lo,hi=np.percentile(subset,[2.5,97.5]);dl,dh=np.percentile(np.asarray(subset)-full,[2.5,97.5])
    return dict(bootstrap_valid=len(subset),ci95_low=float(lo),ci95_high=float(hi),change_ci95_low=float(dl),change_ci95_high=float(dh))


def analyze():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[PROTOCOL,Path(__file__)]
    for path in POOLS.values():
        inputs+=[path/'manifests/candidate_pool.csv',path/'transfer/weak_states_A_frozen.csv',path/'configs/shared_object_schema.json']
        inputs+=list((path/'annotations').glob('*/*/parsed/*.json'))+list((path/'benchmark').glob('**/candidate_outputs.npz'))
    freeze={str(p):sha256(p) for p in inputs}
    previous=OUT/'input_freeze.sha256.json'
    if previous.exists():assert json.loads(previous.read_text())==freeze
    else:write_json(previous,freeze)
    write_json(OUT/'registration.json',dict(created_at=now(),posthoc=True,bootstrap=2000,support_per_class_tier_min=5,
        state_min=20,non_state_reference_min=20,groups=list(GROUPS),original_strata_fixed=True,no_new_inference=True,
        risk_fields=list(MAPS),manual_review=False))
    sample_rows=[];field_rows=[];distribution=[];run_rows=[];summary=[];rank_rows=[];state_rows=[];state_summary=[]
    rng=np.random.default_rng(20261005)
    for pool,path in POOLS.items():
        print(f'[{now()}] analyze {pool}',flush=True)
        rows=read_csv(path/'manifests/candidate_pool.csv');y=np.asarray([LABELS[r['class_name']] for r in rows]);ids=[r['sample_id'] for r in rows]
        labels,failures=labels_for(rows,path);assert not failures
        shared=[r['feature_id'] for r in json.loads((path/'configs/shared_object_schema.json').read_text())['universal_features']]
        assert len(shared)==14
        agree={field:np.asarray([values(labels['local'][iid][field])==values(labels['online'][iid][field]) and
            'unknown' not in values(labels['local'][iid][field])|values(labels['online'][iid][field]) for iid in ids]) for field in shared}
        unknown={field:np.asarray(['unknown' in values(labels['local'][iid][field])|values(labels['online'][iid][field]) for iid in ids]) for field in shared}
        masks={'all':np.ones(len(rows),dtype=bool),'risk_agree':np.logical_and.reduce([agree[f] for f in MAPS]),
               'shared_agree':np.logical_and.reduce([agree[f] for f in shared])}
        for i,row in enumerate(rows):
            sample_rows.append(dict(pool=pool,sample_id=ids[i],class_name=row['class_name'],risk_agree=bool(masks['risk_agree'][i]),
                shared_agree=bool(masks['shared_agree'][i]),agree_fields=sum(v[i] for v in agree.values()),unknown_fields=sum(v[i] for v in unknown.values())))
        for field in shared:
            for c in CLASSES:
                idx=np.flatnonzero(y==LABELS[c]);field_rows.append(dict(pool=pool,class_name=c,field=field,n=len(idx),
                    agree_known=int(agree[field][idx].sum()),unknown=int(unknown[field][idx].sum()),
                    disagree_known=int((~agree[field][idx]&~unknown[field][idx]).sum())))
        predictions=[]
        for model,seed in itertools.product(MODELS,SEEDS):
            with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as cache:
                assert cache['sample_ids'].tolist()==ids and np.isfinite(cache['logits']).all()
                predictions.append(cache['logits'].argmax(1))
        predictions=np.asarray(predictions)
        for source in ('local','online'):
            risks=[risk(row,labels[source][iid]) for row,iid in zip(rows,ids)]
            tiers=np.zeros(len(rows),dtype=int)
            for c in CLASSES:
                idx=np.flatnonzero(y==LABELS[c]);ordered=sorted(idx,key=lambda i:(risks[i]['h_midpoint'],ids[i]))
                for tier,group in enumerate(np.array_split(ordered,5),1):tiers[group]=tier
            fixed={tier:{c:np.flatnonzero((tiers==tier)&(y==LABELS[c])) for c in CLASSES} for tier in (1,5)}
            original_gap=np.asarray([f1(y[tiers==5],p[tiers==5])-f1(y[tiers==1],p[tiers==1]) for p in predictions])
            original_ranking=None
            for group,mask in masks.items():
                for c,tier in itertools.product(CLASSES,range(1,6)):
                    idx=np.flatnonzero((y==LABELS[c])&(tiers==tier)&mask);denom=int(((y==LABELS[c])&(tiers==tier)).sum())
                    distribution.append(dict(pool=pool,source=source,group=group,class_name=c,tier=tier,n=len(idx),original_n=denom,
                        retention=len(idx)/denom,mean_H=float(np.mean([risks[i]['h_midpoint'] for i in idx])) if len(idx) else None,
                        mean_area=float(np.mean([float(rows[i]['bbox_area_ratio']) for i in idx])) if len(idx) else None,
                        mean_unknown_dims=float(np.mean([risks[i]['unknown_dimensions'] for i in idx])) if len(idx) else None))
                minimal=min(int(((y==LABELS[c])&(tiers==tier)&mask).sum()) for c,tier in itertools.product(CLASSES,(1,5)))
                record=dict(pool=pool,source=source,group=group,retained=int(mask.sum()),total=len(rows),retention=float(mask.mean()),
                    Q1_n=int((mask&(tiers==1)).sum()),Q5_n=int((mask&(tiers==5)).sum()),minimum_class_tier_support=minimal,evaluable=minimal>=5)
                if minimal>=5:
                    delta=[];high=[];low=[]
                    for run,(model,seed) in enumerate(itertools.product(MODELS,SEEDS)):
                        p=predictions[run];q1=mask&(tiers==1);q5=mask&(tiers==5)
                        lo=f1(y[q1],p[q1]);hi=f1(y[q5],p[q5]);gap=hi-lo;delta.append(gap);high.append(hi);low.append(lo)
                        run_rows.append(dict(pool=pool,source=source,group=group,model=model,seed=seed,Q1_f1=lo,Q5_f1=hi,
                            delta=gap,change_from_full=gap-original_gap[run]))
                    rank=np.asarray(high).reshape(8,3).mean(1)
                    if group=='all':original_ranking=rank
                    tau=float(kendalltau(original_ranking,rank).statistic)
                    record.update(mean_delta=float(np.mean(delta)),mean_Q1=float(np.mean(low)),mean_Q5=float(np.mean(high)),
                        change_from_full=float(np.mean(delta-original_gap)),negative_runs=int((np.asarray(delta)<0).sum()),ranking_tau=tau,
                        **image_interval(y,predictions,fixed,mask,rng))
                    rank_rows.extend(dict(pool=pool,source=source,group=group,model=model,high_risk_f1=float(value),
                        rank=int(np.argsort(-rank).tolist().index(i)+1),tau_vs_full=tau) for i,(model,value) in enumerate(zip(MODELS,rank)))
                summary.append(record)
        # Field-specific consensus: keep a paired common combo set for full vs filtered comparisons.
        weak=read_csv(path/'transfer/weak_states_A_frozen.csv')
        for w in weak:
            source=w['source'];c=w['class_name'];field=w['feature'];value=w['value'];model=w['model']
            if field=='target_scale':
                from run_multiclass_object_selection_benchmark import scale_state
                membership=np.asarray([scale_state(float(r['bbox_area_ratio']))==value for r in rows]);eligible=np.ones(len(rows),dtype=bool)
            else:
                membership=np.asarray([value in values(labels[source][iid][field]) for iid in ids]);eligible=agree[field]
            population=(y==LABELS[c]);state=population&membership;subset=population&eligible;restricted=state&eligible
            n=int(restricted.sum());nonstate=int((subset&~membership).sum());evaluable=n>=20 and nonstate>=20
            for seed_i,seed in enumerate(SEEDS):
                p=predictions[list(MODELS).index(model)*3+seed_i];correct=p==y
                whole_gap=float(correct[state].mean()-correct[population].mean()) if state.any() else None
                gap=float(correct[restricted].mean()-correct[subset].mean()) if evaluable else None
                state_rows.append(dict(pool=pool,source=source,model=model,seed=seed,class_name=c,field=field,value=value,
                    full_state_n=int(state.sum()),consensus_state_n=n,consensus_class_n=int(subset.sum()),non_state_n=nonstate,
                    evaluable=evaluable,full_gap=whole_gap,consensus_gap=gap,
                    consensus_state_recall=float(correct[restricted].mean()) if evaluable else None,
                    consensus_class_recall=float(correct[subset].mean()) if evaluable else None))
        frame=pd.DataFrame([r for r in state_rows if r['pool']==pool])
        for source in ('local','online'):
            selected=frame[(frame.source==source)&frame.evaluable]
            combos=selected.groupby(['model','class_name','field','value'])[['full_gap','consensus_gap']].mean()
            state_summary.append(dict(pool=pool,source=source,original_combos=len(frame[frame.source==source])//3,evaluable_combos=len(combos),
                negative_consensus_combos=int((combos.consensus_gap<0).sum()),negative_full_same_combos=int((combos.full_gap<0).sum()),
                direction_flips=int(((combos.consensus_gap<0)!=(combos.full_gap<0)).sum()),
                mean_full_same_combo_gap=float(combos.full_gap.mean()) if len(combos) else None,
                mean_consensus_gap=float(combos.consensus_gap.mean()) if len(combos) else None))
    for name,data in [('sample_groups',sample_rows),('field_agreement',field_rows),('retention_distribution',distribution),
        ('run_metrics',run_rows),('risk_summary',summary),('model_ranks',rank_rows),('state_metrics',state_rows),('state_summary',state_summary)]:
        write_csv(OUT/'tables'/f'{name}.csv',data)
    plot(summary,sample_rows)
    lines=['# 标签分歧敏感性结果','',f'生成：{now()}。事后缓存分析，无新增VLM请求或模型推理。','',
        '原全池风险层固定，筛选后不重新划层；每类别Q1/Q5至少5图，否则不可评价。风险一致=四个VLM风险字段已知且一致；共享一致=14字段均已知且一致。','',
        '| 池 | 来源 | 组 | 保留数 | 保留率 | 最小类别/层支持 | Q5−Q1 F1(pp) | 源图区间(pp) | 负向run | 排序tau |',
        '|---|---|---|---:|---:|---:|---:|---|---|---:|']
    for r in summary:
        gap=f"{100*r['mean_delta']:+.2f}" if r['evaluable'] else '不可评价'
        ci=f"[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]" if r.get('bootstrap_valid',0) else '—'
        negative=f"{r['negative_runs']}/24" if r['evaluable'] else '—';tau=f"{r['ranking_tau']:.3f}" if r['evaluable'] else '—'
        lines.append(f"| {r['pool']} | {r['source']} | {r['group']} | {r['retained']} | {100*r['retention']:.2f}% | {r['minimum_class_tier_support']} | {gap} | {ci} | {negative} | {tau} |")
    lines+=['','## 冻结弱状态的字段一致子集','', '| 池 | 来源 | 原组合 | 可评价 | 原同组合负向 | 一致子集负向 | 方向改变 | 原同组合平均差(pp) | 一致子集平均差(pp) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in state_summary:
        a=f"{100*r['mean_full_same_combo_gap']:+.2f}" if r['evaluable_combos'] else '—';b=f"{100*r['mean_consensus_gap']:+.2f}" if r['evaluable_combos'] else '—'
        lines.append(f"| {r['pool']} | {r['source']} | {r['original_combos']} | {r['evaluable_combos']} | {r['negative_full_same_combos']} | {r['negative_consensus_combos']} | {r['direction_flips']} | {a} | {b} |")
    lines+=['','一致筛选改变样本组成，不是真值验证。组间Macro-F1还受类别保留量影响，完整类别/风险/面积分布见tables/retention_distribution.csv。',
        '字段一致状态结果须同时满足状态20图和非状态20图；全体结果在相同可评价组合集合上比较，避免换组合产生虚假稳定。',
        '区间为2000次类别/原层源图重采样，固定24模型预测，共享图像相关性保留；有效次数与筛选−全体差的配对区间见risk_summary.csv。',
        '没有新增主检验，不用这些事后子集p值替代既有结果。不能根据一致组更好推断哪一个VLM正确；未复现及不足支持结果保留。',
        '', '![敏感性](figures/label_sensitivity.png)']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    assert all(sha256(p)==digest for p,digest in freeze.items())
    write_json(OUT/'final_audit.json',dict(status='pass',pools=list(POOLS),targets=len(sample_rows),input_files_unchanged=len(freeze),
        risk_summary_rows=len(summary),state_summary_rows=len(state_summary),posthoc=True,no_new_inference=True,manual_review=False))


def plot(summary,samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.8,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    colors={'all':'#767676','risk_agree':'#3775BA','shared_agree':'#9A4D8E'}
    for ax,pool in zip(axes,POOLS):
        ax.axhline(0,color='.8',lw=.7)
        for source,shift,marker in [('local',-.10,'o'),('online',.10,'s')]:
            for j,group in enumerate(GROUPS):
                r=next(r for r in summary if r['pool']==pool and r['source']==source and r['group']==group)
                if r['evaluable']:
                    center=100*r['mean_delta'];lo=100*r['ci95_low'];hi=100*r['ci95_high']
                    ax.errorbar(j+shift,center,yerr=[[max(0,center-lo)],[max(0,hi-center)]],fmt=marker,color=colors[group],ms=4,capsize=2)
                else:ax.text(j+shift,-2,'NA',fontsize=7,ha='center')
        ax.set(title=f'{pool}: '+{'B':'COCO','C':'VOC','D':'Objects365'}[pool],xticks=range(3),
            xticklabels=['All','Risk agree','14-field agree'],ylabel='Q5 minus Q1 Macro-F1 (pp)',ylim=(-55,5))
        ax.tick_params(axis='x',rotation=20)
        ax.text(0,-52,'o local   s online',fontsize=7)
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'label_sensitivity.svg',bbox_inches='tight');fig.savefig(folder/'label_sensitivity.png',dpi=200,bbox_inches='tight');plt.close(fig)
    shutil.copy2(Path(__file__),folder/Path(__file__).name)


if __name__=='__main__':analyze()
