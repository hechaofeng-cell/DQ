#!/usr/bin/env python3
"""Reproduce expanded-state selection, matched coverage control and cached F1."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import run_other_classes_specific_experiment_20261018 as exp
import annotate_other_classes_specific_v2_20261018 as annotation

OUT=exp.OUT
CN=dict(horse='马',sheep='羊',person='人',car='汽车')
DISPLAY=['YOLO26s','PE-Core-B16','SigLIP2-Base','MobileCLIP2-S0','AIMv2-Large']
COLORS=['#1769AA','#D55E00','#009E73','#7B2CBF','#B28A00']
MARKERS=['o','s','^','D','v']


def write(name,data):
    frame=pd.DataFrame(data);frame.to_csv(OUT/name,index=False,encoding='utf-8-sig');return frame


def prepare_select():
    rows,schemas=exp.load()
    for source,expected in json.loads((OUT/'input_freeze.json').read_text()).items():
        assert exp.sha(source)==expected,'Frozen input changed: '+source
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert exp.sha(exp.ROOT/'configs/category_five_dimension_pilot_v1.json')==protocol['risk_config_sha256']
    for c in exp.TARGETS:
        snapshot=OUT/'frozen_dictionaries'/f'{c}.json'
        original=exp.DICTIONARY/c/'coverage_features.json'
        assert snapshot.read_bytes()==original.read_bytes(),'Dictionary snapshot mismatch'
    native={r['sample_id']:r['features'] for r in map(json.loads,(exp.SOURCE/'native_features.jsonl').read_text().splitlines())}
    common_states=json.loads((exp.SOURCE/'state_dictionary.json').read_text())['main_states']
    shared=json.loads((exp.DICTIONARY/'shared_features_reference.json').read_text())['features']
    shared_groups={f['feature_id']:f['major_group'] for f in shared}
    config=json.loads((exp.ROOT/'configs/category_five_dimension_pilot_v1.json').read_text())
    reference=pd.read_csv(exp.ROOT/'artifacts/cat_joint_recent_five_20261018/risk_scores_and_tiers.csv').set_index('sample_id')
    with np.load(exp.SOURCE/'selection.npz',allow_pickle=False) as z:
        assert z['sample_ids'].ravel().tolist()==[r['sample_id'] for r in rows]
        old_selected=z['E4'][0].copy();random=z['E3'].copy()
    assert old_selected.shape==(6,200) and random.shape==(1000,6,200)
    selected=old_selected.copy();tiers=np.zeros(2400,dtype=int)
    data={};scores_out=[];trace_out=[];annotations=[];unknown=[];unknown_Q=[];input_hashes={}
    quota_out=[];state_rows=[];coverage=[];cumulative=[];group_stats=[];ablation=[]
    for ci,c in enumerate(exp.CLASSES):
        subset=rows[ci*400:(ci+1)*400];ids=[r['sample_id'] for r in subset]
        risk=[exp.scoring.score_sample(float(r['bbox_area_ratio']),native[r['sample_id']],c,'equal_parts',config) for r in subset]
        for sid,r in zip(ids,risk):
            for k in ['score_lower','score_midpoint','score_upper']:
                assert np.isclose(r[k],reference.loc[sid,k],atol=1e-10,rtol=0)
        structure=exp.experiment.risk_structure(risk,ids,200);tier=structure['tiers']
        assert np.array_equal(tier,reference.loc[ids,'Q'].to_numpy())
        tiers[ci*400:(ci+1)*400]=tier
        common=exp.experiment.state_matrix(subset,[native[sid] for sid in ids],common_states)
        if c not in exp.TARGETS:
            for i,(row,r) in enumerate(zip(subset,risk)):
                scores_out.append(dict(sample_id=row['sample_id'],class_name=c,Q=int(tier[i]),selected=int(i in selected[ci]),**r))
            continue
        fields=schemas[c]['features'];states=[];values=[];groups=[]
        for f in fields:
            for v in f['values']:
                if v['value_id']=='unknown':continue
                states.append(('specific',f['feature_id'],v['value_id']))
                groups.append(f['major_group'])
        for row in subset:
            path=OUT/'annotations_v2'/c/f'{row["sample_id"]}_parsed.json'
            a=json.loads(path.read_text())
            assert a['sample_id']==row['sample_id'] and a['category']==c
            assert a['dictionary_sha256']==exp.sha(exp.DICTIONARY/c/'coverage_features.json')
            assert a['original_sha256']==row['image_sha256'] and a['crop_sha256']==row['crop_sha256']
            annotation.validate(a['features'],fields)
            raw_path=path.with_name(f'{row["sample_id"]}_attempt{a["attempt"]}_raw.json')
            raw_response=json.loads(raw_path.read_text())
            assert raw_response['done_reason']=='stop'
            assert a['features']==json.loads(raw_response['message']['content']),'Parsed output differs from raw VLM response'
            evidence=path.with_name(f'{row["sample_id"]}_evidence.jpg')
            assert exp.sha(evidence)==a['evidence_sha256']
            input_hashes[str(path)]=exp.sha(path)
            input_hashes[str(raw_path)]=exp.sha(raw_path)
            input_hashes[str(evidence)]=exp.sha(evidence)
            values.append(a['features']);annotations.append(a)
        local=np.zeros((400,len(states)),dtype=np.uint8)
        for i,v in enumerate(values):
            for j,(_,field,state) in enumerate(states):
                local[i,j]=state in v[field] if isinstance(v[field],list) else v[field]==state
        matrix=np.concatenate([common,local],axis=1)
        all_states=[('shared',f,v) for f,v in common_states]+states
        all_groups=[shared_groups[f] for f,v in common_states]+groups
        chosen,trace=exp.experiment.old.greedy(dict(matrix=matrix,sample_ids=np.array(ids)),200,structure)
        selected[ci]=chosen
        assert np.bincount(tier[chosen],minlength=6)[1:].tolist()==[40]*5
        # All controls match every exact score+uncertainty cell, not just Q.
        for cell,members in structure['cells'].items():
            assert int(np.isin(chosen,members).sum())==structure['quota'][cell]
            assert np.all(np.isin(random[:,ci],members).sum(axis=1)==structure['quota'][cell])
            quota_out.append(dict(class_name=c,Q=cell[0],risk_midpoint=cell[1]/1e8,uncertainty_width=cell[2]/1e8,candidates=len(members),quota=structure['quota'][cell]))
        trace_out.extend(dict(class_name=c,**r) for r in trace)
        for i,(row,r) in enumerate(zip(subset,risk)):
            scores_out.append(dict(sample_id=row['sample_id'],class_name=c,Q=int(tier[i]),selected=int(i in chosen),**r))
        counts=matrix[chosen].sum(0);baseline=matrix[random[:,ci]].sum(1)
        pool=matrix.sum(0)
        for j,(namespace,field,value) in enumerate(all_states):
            state_rows.append(dict(class_name=c,namespace=namespace,field=field,value=value,major_group=all_groups[j],pool_support=int(pool[j]),selected_support=int(counts[j]),random_support_mean=float(baseline[:,j].mean())))
        missing_fields=[]
        for f in fields:
            missing=np.array([v[f['feature_id']]==['unknown'] if isinstance(v[f['feature_id']],list) else v[f['feature_id']]=='unknown' for v in values])
            missing_fields.append(missing)
            unknown.append(dict(class_name=c,field=f['feature_id'],major_group=f['major_group'],pool_unknown_count=int(missing.sum()),pool_unknown_pct=100*missing.mean(),selected_unknown_pct=100*missing[chosen].mean()))
        missing_matrix=np.array(missing_fields).T
        for q in range(1,6):
            unknown_Q.append(dict(class_name=c,Q=q,pool_targets=80,selected_targets=40,
                pool_unknown_pct=float(100*missing_matrix[tier==q].mean()),
                selected_unknown_pct=float(100*missing_matrix[chosen[tier[chosen]==q]].mean())))
        for namespace,mask in [('all',np.ones(len(all_states),dtype=bool)),('shared',np.arange(len(all_states))<71),('specific',np.arange(len(all_states))>=71)]:
            for method,ct in [('expanded_greedy',counts[None]),('same_quota_random',baseline),('shared_only_greedy',matrix[old_selected[ci]].sum(0)[None])]:
                ct=ct[:,mask]
                coverage.append(dict(class_name=c,namespace=namespace,method=method,states=int(mask.sum()),selected_targets=200,repetitions=len(ct),
                    presence_pct=float((ct>0).mean()*100),support_completion_pct=float(np.minimum(ct/20,1).mean()*100),
                    support_threshold_pct=float((ct>=20).mean()*100),pool_present_states=int((pool[mask]>0).sum()),
                    retained_pool_available_pct=float(((ct>0)&(pool[mask]>0)).sum(1).mean()/(pool[mask]>0).sum()*100)))
        for group in sorted(set(all_groups)):
            mask=np.array(all_groups)==group
            group_stats.append(dict(class_name=c,major_group=group,states=int(mask.sum()),presence_pct=float((counts[mask]>0).mean()*100),support_completion_pct=float(np.minimum(counts[mask]/20,1).mean()*100),random_presence_pct=float((baseline[:,mask]>0).mean()*100),random_support_completion_pct=float(np.minimum(baseline[:,mask]/20,1).mean()*100)))
        previous=np.zeros(matrix.shape[1],dtype=int);prev_random=np.zeros_like(baseline)
        for q in range(1,6):
            ix=chosen[tier[chosen]<=q]
            rx=random[:,ci][tier[random[:,ci]]<=q].reshape(1000,40*q)
            a=matrix[ix].sum(0);b=matrix[rx].sum(1)
            assert len(ix)==40*q and np.all(a>=previous) and np.all(b>=prev_random)
            previous=a;prev_random=b
            for metric,current,control in [('presence',(a>0).mean(),(b>0).mean(1)),('support_completion',np.minimum(a/20,1).mean(),np.minimum(b/20,1).mean(1))]:
                lo,hi=np.quantile(control,[.025,.975])
                cumulative.append(dict(class_name=c,Q=q,metric=metric,n_cumulative=40*q,states=matrix.shape[1],selected_pct=100*current,random_mean_pct=100*control.mean(),random_p025_pct=100*lo,random_p975_pct=100*hi,improvement_pp=100*(current-control.mean())))
        ablation.append(dict(class_name=c,changed_selected_targets=200-len(set(chosen)&set(old_selected[ci])),specific_fields=len(fields),specific_states=local.shape[1],total_states=matrix.shape[1],pool_unknown_pct=100*np.mean([r['pool_unknown_pct']/100 for r in unknown if r['class_name']==c])))
        np.savez_compressed(OUT/f'{c}_state_matrix.npz',sample_ids=np.array(ids),matrix=matrix,specific_matrix=local,tiers=tier,selected=chosen,random=random[:,ci])
        exp.dump(OUT/f'{c}_state_dictionary.json',dict(states=all_states,major_groups=all_groups,unknown_excluded=True,support_cap=20))
        data[c]=dict(matrix=matrix,selected=chosen,states=all_states)
        print('[selection]',c,matrix.shape,'changed',ablation[-1]['changed_selected_targets'],flush=True)
    write('risk_scores_and_tiers.csv',scores_out);write('greedy_trace.csv',trace_out);write('risk_cell_quotas.csv',quota_out)
    write('state_support.csv',state_rows);write('unknown_diagnostics.csv',unknown)
    write('unknown_by_Q.csv',unknown_Q)
    write('coverage_summary.csv',coverage);write('coverage_cumulative_Q1_Q5.csv',cumulative)
    write('coverage_major_groups.csv',group_stats);write('selection_ablation.csv',ablation)
    (OUT/'specific_annotations.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in annotations))
    np.savez_compressed(OUT/'selection.npz',sample_ids=np.array([r['sample_id'] for r in rows]).reshape(6,400),expanded=selected,old_shared_only=old_selected,random=random,tiers=tiers)
    freeze_paths=[OUT/n for n in ['selection.npz','risk_scores_and_tiers.csv','coverage_summary.csv','coverage_cumulative_Q1_Q5.csv','specific_annotations.jsonl']]
    freeze_paths+=[Path(__file__),Path(annotation.__file__),Path(exp.__file__),OUT/'protocol.json',OUT/'input_freeze.json']
    exp.dump(OUT/'pre_evaluation_freeze.json',dict(source_sha256={**input_hashes,**{str(p):exp.sha(p) for p in freeze_paths}},scores_and_tiers_unchanged=True,annotation_targets=len(annotations),expanded_greedy_ignores_classifier_predictions=True))
    return rows,tiers,selected,old_selected


def evaluate(rows,tiers,selected,old_selected):
    ids=[r['sample_id'] for r in rows];truth=np.array([int(r['label']) for r in rows])
    outputs=[];hashes={};predictions=[]
    for model in exp.MODELS:
        for seed in exp.SEEDS:
            p=exp.ROOT/f'artifacts/recent_five_models_20261009/runs/{model}/{seed}/B_outputs.npz'
            with np.load(p,allow_pickle=False) as z:
                cache_ids=z['sample_ids'].astype(str).tolist()
                assert len(set(cache_ids))==2400 and set(cache_ids)==set(ids)
                index={sid:i for i,sid in enumerate(cache_ids)};order=[index[sid] for sid in ids]
                assert np.array_equal(z['labels'][order],truth)
                logits=z['logits'][order];assert logits.shape==(2400,6) and np.isfinite(logits).all()
                predictions.append(logits.argmax(1))
            hashes[str(p)]=exp.sha(p)
    predictions=np.array(predictions)
    cohorts=dict(pool=np.arange(2400),expanded=(selected+np.arange(6)[:,None]*400).ravel(),shared_only=(old_selected+np.arange(6)[:,None]*400).ravel())
    members=[];confusions=[]
    for cohort,indices in cohorts.items():
        n=80 if cohort=='pool' else 40
        for q in range(1,6):
            take=indices[tiers[indices]==q]
            assert np.bincount(truth[take],minlength=6).tolist()==[n]*6
            members.extend(dict(cohort=cohort,Q=q,sample_id=ids[i],class_name=exp.CLASSES[truth[i]]) for i in take)
            for run,pred in enumerate(predictions):
                model=exp.MODELS[run//3];seed=exp.SEEDS[run%3]
                cm=confusion_matrix(truth[take],pred[take],labels=np.arange(6))
                tp=cm.diagonal();fp=cm.sum(0)-tp;fn=cm.sum(1)-tp
                f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(6,dtype=float),where=(2*tp+fp+fn)>0)
                outputs.append(dict(cohort=cohort,Q=q,model=model,seed=seed,class_name='macro',f1_pct=100*f1.mean(),tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum()),targets=len(take)))
                for ci,c in enumerate(exp.CLASSES):
                    outputs.append(dict(cohort=cohort,Q=q,model=model,seed=seed,class_name=c,f1_pct=100*f1[ci],tp=int(tp[ci]),fp=int(fp[ci]),fn=int(fn[ci]),targets=int(cm[ci].sum())))
                confusions.append(dict(cohort=cohort,Q=q,model=model,seed=seed,matrix=cm.tolist()))
    raw=write('model_F1_by_seed.csv',outputs)
    previous=pd.read_csv(exp.ROOT/'artifacts/cat_joint_recent_five_20261018/all_class_metrics_runs.csv')
    previous=previous.query('condition == "joint"').copy()
    previous['cohort']=previous['group'].map({'pool':'pool','coverage':'shared_only'})
    baseline=raw.query('cohort in ["pool", "shared_only"] and class_name != "macro"')
    matched=baseline.merge(previous,on=['cohort','class_name','model','seed','Q'],validate='one_to_one')
    assert len(matched)==len(baseline)
    assert np.allclose(matched.f1_pct.to_numpy(),100*matched.f1.to_numpy(),atol=1e-10,rtol=0)
    summary=raw.groupby(['cohort','class_name','model','Q'],as_index=False).agg(f1_mean_pct=('f1_pct','mean'),f1_sd_pct=('f1_pct','std'))
    summary.to_csv(OUT/'model_F1_summary.csv',index=False,encoding='utf-8-sig')
    trends=[]
    for (cohort,c,m),frame in summary.groupby(['cohort','class_name','model']):
        y=frame.sort_values('Q').f1_mean_pct.to_numpy()
        trends.append(dict(cohort=cohort,class_name=c,model=m,Q1_f1_pct=y[0],Q5_f1_pct=y[-1],Q5_minus_Q1_pp=y[-1]-y[0],intermediate_increases=int((np.diff(y)>1e-10).sum())))
    write('F1_trends.csv',trends);write('evaluation_membership.csv',members)
    exp.dump(OUT/'confusion_matrices.json',confusions)
    exp.dump(OUT/'prediction_audit.json',dict(status='pass',runs=15,models=exp.MODELS,seeds=exp.SEEDS,targets=2400,ids_labels_matched=True,
        historical_pool_and_shared_only_F1_matches=True,historical_F1_crosschecks=len(matched),new_inference=False,new_training=False,source_sha256=hashes))
    return summary


def style():
    font=Path('/home/hcf/.local/share/fonts/NotoSansCJKsc-Regular.otf')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],
        'svg.fonttype':'none','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'savefig.facecolor':'white'})


def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    path=OUT/f'{name}.svg';root=ET.parse(path).getroot()
    assert len(root.findall('.//{http://www.w3.org/2000/svg}text'))>0
    assert not root.findall('.//{http://www.w3.org/2000/svg}image')
    assert path.stat().st_size<1_000_000


def plots(summary):
    style();fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,c,letter in zip(axes.flat,exp.TARGETS,'ABCD'):
        for model,name,color,marker in zip(exp.MODELS,DISPLAY,COLORS,MARKERS):
            part=summary.query('cohort == "expanded" and class_name == @c and model == @model').sort_values('Q')
            x=part.Q.to_numpy();y=part.f1_mean_pct.to_numpy();sd=part.f1_sd_pct.to_numpy()
            ax.plot(x,y,color=color,marker=marker,linewidth=1.8,markersize=4.5,label=name)
            ax.fill_between(x,np.clip(y-sd,0,100),np.clip(y+sd,0,100),color=color,alpha=.08,linewidth=0)
        sub=summary.query('cohort == "expanded" and class_name == @c')
        lo=max(0,np.floor((sub.f1_mean_pct.min()-sub.f1_sd_pct.max()-3)/5)*5)
        hi=min(100,np.ceil((sub.f1_mean_pct.max()+sub.f1_sd_pct.max()+3)/5)*5)
        ax.set_ylim(lo,max(lo+10,hi));ax.yaxis.set_major_locator(MultipleLocator(2 if hi-lo<=40 else 5 if hi-lo<=60 else 10))
        ax.set_xticks(range(1,6),[f'Q{i}' for i in range(1,6)])
        ax.set_xlim(.8,5.2);ax.set_title(f'{letter}  {CN[c]}类',loc='left',fontsize=14,fontweight='bold')
        ax.set_xlabel('独立风险层（Q1低 → Q5高）');ax.set_ylabel('类别 F1（%）');ax.grid(axis='y',color='#E7EBEF',linewidth=.6)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=5,frameon=False)
    fig.suptitle('四类专属状态选样后的 Q1–Q5 模型表现',fontsize=19,y=.98)
    fig.text(.5,.93,'六类共同分层评测；每类每层40个目标；五模型各3个种子；线为均值，阴影为±1标准差',ha='center',color='#52606D')
    fig.text(.5,.025,'复用冻结模型预测；每个Q独立计算；纵轴显示局部范围。未对分数或入选目标施加F1递减要求。',ha='center',fontsize=9,color='#52606D')
    fig.subplots_adjust(top=.86,bottom=.2,left=.08,right=.98,wspace=.24,hspace=.40)
    save(fig,'四类_五模型_Q1-Q5_F1')
    cov=pd.read_csv(OUT/'coverage_cumulative_Q1_Q5.csv');fig,axes=plt.subplots(2,4,figsize=(17,8.2))
    for col,c in enumerate(exp.TARGETS):
        for row,metric,title in [(0,'presence','累计出现覆盖率'),(1,'support_completion','累计支持完成度 K=20')]:
            ax=axes[row,col];part=cov.query('class_name == @c and metric == @metric').sort_values('Q')
            x=part.Q.to_numpy();current=part.selected_pct.to_numpy();random=part.random_mean_pct.to_numpy()
            ax.fill_between(x,part.random_p025_pct.to_numpy(),part.random_p975_pct.to_numpy(),color='#8795A2',alpha=.16,linewidth=0)
            ax.plot(x,current,'-o',color='#1769AA',linewidth=2,markersize=4,label='专属状态覆盖选样')
            ax.plot(x,random,'--s',color='#7D8993',linewidth=1.5,markersize=3.5,label='同配额随机（1000次均值）')
            ax.set_title(f'{CN[c]}｜{title}',loc='left',fontsize=12)
            ax.set_xticks(range(1,6),['Q1','Q1–2','Q1–3','Q1–4','Q1–5'],rotation=20)
            ax.set_xlabel('累计纳入风险层');ax.set_ylabel('比例（%）')
            lo=max(0,np.floor((min(current.min(),part.random_p025_pct.min())-4)/10)*10)
            hi=min(100,np.ceil((max(current.max(),part.random_p975_pct.max())+4)/10)*10)
            ax.set_ylim(lo,max(lo+10,hi));ax.grid(axis='y',color='#E7EBEF',linewidth=.6)
            ax.text(.98,.04,f"固定 {int(part.states.iloc[0])} 个状态",ha='right',transform=ax.transAxes,color='#52606D',fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
    fig.suptitle('四类｜专属状态＋共享状态的累计覆盖',fontsize=20,y=.98)
    fig.text(.5,.93,'每层40个；累计40 / 80 / 120 / 160 / 200个；共享71状态＋各类专属状态；灰带为随机选样分位区间',ha='center',color='#52606D')
    fig.text(.5,.025,'出现覆盖：至少1例；支持完成度：平均 min(状态例数/20,1)。未知不计覆盖；固定字典含候选池未出现状态。',ha='center',fontsize=9,color='#52606D')
    fig.subplots_adjust(top=.85,bottom=.23,left=.055,right=.985,wspace=.3,hspace=.52)
    save(fig,'四类_Q1-Q5_累计覆盖率')


def report():
    coverage=pd.read_csv(OUT/'coverage_summary.csv')
    trends=pd.read_csv(OUT/'F1_trends.csv');ablation=pd.read_csv(OUT/'selection_ablation.csv')
    unknown=pd.read_csv(OUT/'unknown_diagnostics.csv')
    lines=['# 四类专属特征实验结果','',
        '使用已确认的马、羊、人、汽车字典，完成1600个真实目标的逐目标本地Qwen3-VL标注；原五维风险规则保持不变。','',
        '每类400个候选、每层80个；按原风险分数/不确定性联合配额选200个，每层40个。随机对照1000次，预算与所有风险单元配额完全一致。','',
        '五个模型各3个种子，复用匹配2400个目标的冻结预测重新计算；没有重新训练或推理。每层合并六类各40个目标，以六分类混淆矩阵计算各类别F1。狗、猫仍用先前共享字典选样；四类使用本次扩展字典。','',
        '## 同配额覆盖对照','',
        '| 类别 | 总状态数 | 当前出现覆盖 | 随机出现覆盖 | 提升(pp) | 当前支持完成度 | 随机支持完成度 | 提升(pp) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    final=[]
    for c in exp.TARGETS:
        a=coverage.query('class_name == @c and namespace == "all" and method == "expanded_greedy"').iloc[0]
        b=coverage.query('class_name == @c and namespace == "all" and method == "same_quota_random"').iloc[0]
        lines.append(f'|{CN[c]}|{int(a.states)}|{a.presence_pct:.2f}%|{b.presence_pct:.2f}%|{a.presence_pct-b.presence_pct:+.2f}|{a.support_completion_pct:.2f}%|{b.support_completion_pct:.2f}%|{a.support_completion_pct-b.support_completion_pct:+.2f}|')
        t=trends.query('cohort == "expanded" and class_name == @c')
        r=ablation.query('class_name == @c').iloc[0]
        new_specific=coverage.query('class_name == @c and namespace == "specific" and method == "expanded_greedy"').iloc[0]
        old_specific=coverage.query('class_name == @c and namespace == "specific" and method == "shared_only_greedy"').iloc[0]
        new_shared=coverage.query('class_name == @c and namespace == "shared" and method == "expanded_greedy"').iloc[0]
        old_shared=coverage.query('class_name == @c and namespace == "shared" and method == "shared_only_greedy"').iloc[0]
        final.append(dict(class_name=c,presence_gain_pp=a.presence_pct-b.presence_pct,support_completion_gain_pp=a.support_completion_pct-b.support_completion_pct,
            specific_presence_gain_vs_old_pp=new_specific.presence_pct-old_specific.presence_pct,
            specific_support_gain_vs_old_pp=new_specific.support_completion_pct-old_specific.support_completion_pct,
            shared_support_change_vs_old_pp=new_shared.support_completion_pct-old_shared.support_completion_pct,
            Q1_mean_F1_pct=t.Q1_f1_pct.mean(),Q5_mean_F1_pct=t.Q5_f1_pct.mean(),F1_delta_pp=t.Q5_minus_Q1_pp.mean(),models_end_lower=int((t.Q5_minus_Q1_pp<0).sum()),intermediate_increases=int(t.intermediate_increases.sum()),**{k:r[k] for k in ['specific_fields','specific_states','total_states','changed_selected_targets','pool_unknown_pct']}))
    summary=write('final_summary.csv',final)
    lines+=['','支持完成度是平均min(n/20,1)，不是“达到20例的状态比例”；后者另存coverage_summary.csv。各类字典分母不同，覆盖率应与本类同字典随机对照比较，不能直接按绝对百分比跨类排名。','',
        '## 加入专属状态前后的选样比较','',
        '两种入选集合均按本次完整标注计算，预算和风险单元配额相同；旧方法只用共享状态驱动选样，新方法加入专属状态。','',
        '| 类别 | 替换入选目标数 | 专属出现覆盖变化(pp) | 专属支持完成度变化(pp) | 共享支持完成度变化(pp) |',
        '|---|---:|---:|---:|---:|']
    for r in final:
        lines.append(f'|{CN[r["class_name"]]}|{int(r["changed_selected_targets"])}|{r["specific_presence_gain_vs_old_pp"]:+.2f}|{r["specific_support_gain_vs_old_pp"]:+.2f}|{r["shared_support_change_vs_old_pp"]:+.2f}|')
    lines+=['',
        '## F1走势（五模型均值）','',
        '| 类别 | Q1 F1 | Q5 F1 | Q5−Q1(pp) | 末端下降模型数 | 中间回升次数(共20段) |','|---|---:|---:|---:|---:|---:|']
    for r in final:
        lines.append(f'|{CN[r["class_name"]]}|{r["Q1_mean_F1_pct"]:.2f}%|{r["Q5_mean_F1_pct"]:.2f}%|{r["F1_delta_pp"]:+.2f}|{r["models_end_lower"]}/5|{r["intermediate_increases"]}|')
    lines+=['','F1各Q独立计算；覆盖图是Q1至Qk累计，两者不能混为一谈。三种子标准差不是数据集抽样置信区间。没有为使曲线下降而改变风险排序或挑选模型。','',
        '## 自动标注与范围','',
        '首轮多目标数值编码试跑出现不可见鞋类被猜测的问题，已完整保留并排除；最终仅使用annotations_v2逐目标、具名字段和约束取值输出。逐目标标注仍可能错误，本实验未测量人工一致性或标注准确率。','',
        '缺失unknown不计覆盖。coverage_major_groups.csv保留五大字典组结果，unknown_diagnostics.csv逐字段记录未知比例。新字典结果不能与旧71状态覆盖率直接作增减比较。','',
        '累计覆盖不下降是固定字典下的数学性质；方法效果以同预算、同配额随机对照与共享字典选样消融为依据。F1下降本身不能证明测试集质量更高。','',
        '## 图与原始表','',
        '- [四类F1图](四类_五模型_Q1-Q5_F1.svg)','- [四类累计覆盖图](四类_Q1-Q5_累计覆盖率.svg)',
        '- final_summary.csv、model_F1_by_seed.csv、model_F1_summary.csv、confusion_matrices.json',
        '- specific_annotations.jsonl、state_support.csv、risk_scores_and_tiers.csv、selection.npz',
        '- pre_evaluation_freeze.json、prediction_audit.json、final_audit.json','']
    (OUT/'开始这里.md').write_text('\n'.join(lines))
    (OUT/'claim_evidence.md').write_text('# Claims and evidence\n\n'
        '| Claim | Evidence | Boundary |\n|---|---|---|\n'
        '| 1600 independent target annotations | annotations_v2/*/*_parsed.json and raw local responses; specific_annotations.jsonl | automatic, no measured annotation accuracy |\n'
        '| Risk scores and Q memberships unchanged | risk_scores_and_tiers.csv vs frozen reference; pre_evaluation_freeze.json | same candidate pool and configuration |\n'
        '| Coverage differences | coverage_summary.csv / coverage_cumulative_Q1_Q5.csv; saved matrices and 1000 matched draws | class-specific fixed denominators, not old71-state comparisons |\n'
        '| F1 trends | model_F1_by_seed.csv and confusion_matrices.json; prediction_audit.json | cached prediction re-evaluation; same-Q six-class mixture; no causal claim from decline |\n')
    files=[p for p in OUT.glob('*.svg')]
    exp.dump(OUT/'final_audit.json',dict(status='pass',annotated_targets=1600,classes=exp.TARGETS,annotations_protocol='independent-named-v2',
        initial_multi_target_smoke_excluded=True,risk_scores_and_Q_unchanged=True,
        each_class_pool=400,each_class_selected=200,each_class_each_Q_selected=40,
        random_repeats=1000,exact_risk_cell_quotas_match=True,cumulative_counts_monotonic=True,
        classifier_runs=15,cached_prediction_re_evaluation=True,no_new_training_or_inference=True,
        automatic_annotation_accuracy_not_measured=True,
        figures={p.name:dict(bytes=p.stat().st_size,sha256=exp.sha(p),editable_text=True) for p in files},
        source_sha256={str(p):exp.sha(p) for p in [Path(__file__),Path(annotation.__file__),Path(exp.__file__),OUT/'final_summary.csv',OUT/'model_F1_by_seed.csv',OUT/'coverage_summary.csv',OUT/'pre_evaluation_freeze.json']}))
    exp.dump(OUT/'annotation_progress_v2.json',dict(complete=True,total=1600,per_class=dict.fromkeys(exp.TARGETS,400),
        final_data_source='annotations_v2',initial_smoke_excluded=True))
    with (OUT/'research_log.md').open('a') as f:
        f.write('\n## Completed execution\n\n'
            'Commands: scripts/annotate_other_classes_specific_v2_20261018.py; scripts/analyze_other_classes_specific_experiment_20261018.py. '
            'Initial batch smoke failed semantics and is excluded. Independent named schema v2 is final. '
            'Risk, exact quotas, cached prediction IDs/labels and cumulative monotonicity asserted. '
            'Raw multi-target failures remain in annotations; independent target responses in annotations_v2. '
            'All charts contain editable text, no raster elements, and are below1MB.\n')
    print(summary.to_string(index=False),flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    rows,tiers,selected,old_selected=prepare_select()
    summary=evaluate(rows,tiers,selected,old_selected)
    plots(summary);report()


if __name__=='__main__':main()
