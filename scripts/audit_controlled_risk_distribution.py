#!/usr/bin/env python3
"""Reconstruct classification scores and audit the risk-mixture report artifacts."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/controlled_risk_distribution_20261005'
CLASSES=['dog','cat','horse','sheep','person','car']
POOLS={'B':ROOT/'artifacts/sixclass_independent_confirmation_20261004',
       'C':ROOT/'artifacts/sixclass_voc_external_confirmation_20261005',
       'D':ROOT/'artifacts/objects365_independent_confirmation_20261005'}


def main():
    checks=0;max_gap=0.
    def close(actual,expected,tol=1e-6):
        nonlocal checks,max_gap
        a,b=np.asarray(actual,float),np.asarray(expected,float)
        assert a.shape==b.shape,(a.shape,b.shape)
        gap=float(np.max(np.abs(a-b))) if a.size else 0.
        assert gap<=tol,(gap,tol)
        max_gap=max(max_gap,gap);checks+=a.size
    frames={name:pd.read_csv(BASE/'tables'/f'{name}.csv') for name in (
        'model_summary','class_recall','panel_summary','distribution_contrasts','class_panel','class_panel_contrasts')}
    risk=pd.read_csv(ROOT/'artifacts/ranking_attribute_validation_20261005/tables/risk_assignments.csv')
    quota_targets=0;direct_checks=0;reconstructed_scores=0;main_top2=True
    for file in sorted((BASE/'raw').glob('*.npz')):
        pool,source,n=file.stem.split('_');total=int(n[1:]);budget=total//6
        rows=pd.read_csv(POOLS[pool]/'manifests/candidate_pool.csv')
        y=rows.class_name.map(dict(zip(CLASSES,range(6)))).to_numpy()
        prior=risk[(risk.pool==pool)&(risk.source==source)&(risk.unknown_mode=='midpoint')].set_index('sample_id')
        tier=prior.loc[rows.sample_id].tier.to_numpy()
        with np.load(BASE/'sampling'/f'{pool}_{source}.npz') as z:
            high=z['high'][:,:,:budget];low=z['low'][:,:,:budget]
        assert high.shape==low.shape==(1000,6,budget)
        assert np.all(y[high]==np.arange(6)[None,:,None]) and np.all(y[low]==np.arange(6)[None,:,None])
        assert np.all(tier[high]==5) and np.all(tier[low]<5)
        both=np.concatenate([high,low],axis=2)
        assert np.all(np.diff(np.sort(both,axis=2),axis=2)>0)
        quota_targets+=both.size
        with np.load(BASE/'clusters'/f'{pool}.npz') as z:predictions=z['predictions']
        with np.load(file) as z:
            raw=z['metrics'].astype(float);recalls=z['class_recall'].astype(float)
            names=z['metric_names'].tolist();proportions=z['proportions'];models=z['models'].tolist()
        assert raw.shape==(21,1000,24,14) and recalls.shape==(21,1000,24,6)
        assert np.array_equal(proportions,np.arange(0,101,5)) and np.isfinite(raw).all()
        close(raw[...,0],raw[...,2]);close(raw[4,:,:,7:9],np.zeros((1000,24,2)))
        exact=np.empty((21,1000,24));nh=budget*proportions//100;nl=budget-nh
        # Independently histogram saved target predictions before extracting each mixture.
        for run,pred in enumerate(predictions):
            hi=np.eye(6,dtype=np.int32)[pred[high]].cumsum(axis=2)
            lo=np.eye(6,dtype=np.int32)[pred[low]].cumsum(axis=2)
            hi=np.pad(hi,((0,0),(0,0),(1,0),(0,0)))
            lo=np.pad(lo,((0,0),(0,0),(1,0),(0,0)))
            cm=(hi[:,:,nh,:]+lo[:,:,nl,:]).transpose(2,0,1,3)
            assert np.all(cm.sum(-1)==budget)
            tp=np.diagonal(cm,axis1=-2,axis2=-1).astype(float)
            incoming=cm.sum(-2)
            precision=np.divide(tp,incoming,out=np.zeros_like(tp),where=incoming>0)
            rec=tp/budget;f1=2*tp/(budget+incoming)
            exact[:,:,run]=f1.mean(-1)
            for key,value in [('accuracy',tp.sum(-1)/total),('macro_precision',precision.mean(-1)),
                              ('macro_recall',rec.mean(-1)),('macro_f1',f1.mean(-1))]:
                close(raw[:,:,run,names.index(key)],value)
            close(recalls[:,:,run],rec);reconstructed_scores+=21*1000
            if run in (0,23):
                for p in (0,4,11,20):
                    for repeat in (0,17,999):
                        chosen=np.concatenate([np.r_[high[repeat,c,:nh[p]],low[repeat,c,:nl[p]]] for c in range(6)])
                        assert len(chosen)==len(set(chosen.tolist()))==total
                        _,rr,ff,_=precision_recall_fscore_support(y[chosen],pred[chosen],labels=range(6),zero_division=0)
                        close(ff.mean(),exact[p,repeat,run],tol=1e-12)
                        close(rr,recalls[p,repeat,run]);direct_checks+=1
        f=raw.reshape(21,1000,8,3,14).mean(3)
        rc=recalls.reshape(21,1000,8,3,6).mean(3)
        scores=exact.reshape(21,1000,8,3).mean(3);point=scores.mean(1)
        ranks=rankdata(-point,axis=-1,method='average')
        draw_ranks=rankdata(-scores,axis=-1,method='average')
        cutoff=np.sort(scores,axis=-1)[...,-2:];cutoff=cutoff[...,:1]
        above=scores>cutoff;ties=scores==cutoff
        top=above+ties*(2-above.sum(-1,keepdims=True))/ties.sum(-1,keepdims=True)
        lookup={k:df[(df.pool==pool)&(df.source==source)&(df.total==total)] for k,df in frames.items()}
        for p,ratio in enumerate(proportions):
            for mi,model in enumerate(models[::3]):
                row=lookup['model_summary'][(lookup['model_summary'].model==model)&(lookup['model_summary'].high_risk_percent==ratio)].iloc[0]
                close(row['rank'],ranks[p,mi],tol=1e-12)
                close(row[['rank_sampling_low','rank_sampling_high']],np.percentile(draw_ranks[p,:,mi],[2.5,97.5]),tol=1e-12)
                close(row.top2_frequency,top[p,:,mi].mean(),tol=1e-12)
                close(row.point_kendall_vs20,kendalltau(point[p],point[4]).statistic,tol=1e-12)
                for j,key in enumerate(names):
                    value=f[p,:,mi,j]
                    close(row[[key+'_mean',key+'_sampling_low',key+'_sampling_high']],
                          np.r_[value.mean(),np.percentile(value,[2.5,97.5])])
                for ci,name in enumerate(CLASSES):
                    crow=lookup['class_recall'][(lookup['class_recall'].model==model)&(lookup['class_recall'].high_risk_percent==ratio)&(lookup['class_recall'].class_name==name)].iloc[0]
                    value=rc[p,:,mi,ci]
                    close(crow[['recall_mean','recall_sampling_low','recall_sampling_high']],np.r_[value.mean(),np.percentile(value,[2.5,97.5])])
            row=lookup['panel_summary'][lookup['panel_summary'].high_risk_percent==ratio].iloc[0]
            for j,key in enumerate(names):
                value=f[p,:,:,j].mean(-1)
                close(row[[key+'_mean',key+'_sampling_low',key+'_sampling_high']],np.r_[value.mean(),np.percentile(value,[2.5,97.5])])
            if total==120:
                order=np.argsort(-point[p]);main_top2&=models[3*order[0]]=='convnext_tiny' and models[3*order[1]]=='swin_t'
        for mi,model in list(enumerate(models[::3]))+[(None,'panel_mean')]:
            row=lookup['distribution_contrasts'][lookup['distribution_contrasts'].model==model].iloc[0]
            for j,key in enumerate(names):
                delta=f[-1,:,:,j]-f[0,:,:,j]
                value=delta.mean(-1) if mi is None else delta[:,mi]
                close(row[[key+'_delta',key+'_sampling_low',key+'_sampling_high']],np.r_[value.mean(),np.percentile(value,[2.5,97.5])])
        for ci,name in enumerate(CLASSES):
            value=recalls[:,:,:,ci].mean(2)
            row=lookup['class_panel_contrasts'][lookup['class_panel_contrasts'].class_name==name].iloc[0]
            delta=value[-1]-value[0]
            close(row[['recall_at0','recall_at100','recall_delta','recall_sampling_low','recall_sampling_high']],
                  np.r_[value[0].mean(),value[-1].mean(),delta.mean(),np.percentile(delta,[2.5,97.5])])
            for p,ratio in enumerate(proportions):
                row=lookup['class_panel'][(lookup['class_panel'].class_name==name)&(lookup['class_panel'].high_risk_percent==ratio)].iloc[0]
                close(row[['recall_mean','recall_sampling_low','recall_sampling_high']],
                      np.r_[value[p].mean(),np.percentile(value[p],[2.5,97.5])])
        print(f'{pool}/{source}/n{total}: quotas, exact scores, ranks, paired intervals PASS',flush=True)
    assert main_top2
    freeze=json.loads((BASE/'input_freeze.sha256.json').read_text())
    for file,digest in freeze.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    figures=list((BASE/'figures').glob('*.svg'));assert len(figures)==7
    for file in figures:
        tree=ET.parse(file)
        assert not tree.findall('.//{http://www.w3.org/2000/svg}image'),file
        assert tree.findall('.//{http://www.w3.org/2000/svg}text'),file
        assert file.with_suffix('.png').exists() and file.with_suffix('.py').exists()
    report=(BASE/'report_zh.md').read_text()
    links=re.findall(r'\]\(([^)]+)\)',report)
    for link in links:assert (BASE/link).exists(),link
    assert len(re.findall(r'^!\[',report,re.M))==7
    # Trace every displayed numeric table cell back to its detailed CSV row.
    blocks=re.findall(r'((?:^\|.*\n)+)',report,re.M);assert len(blocks)==4
    table_cells=0
    for table,block in enumerate(blocks):
        for line in block.strip().splitlines()[2:]:
            cell=[v.strip() for v in line.strip('|').split('|')];pool,source=cell[:2]
            panel=frames['panel_summary']
            panel=panel[(panel.pool==pool)&(panel.source==source)&(panel.total==120)].set_index('high_risk_percent')
            delta=frames['distribution_contrasts']
            delta=delta[(delta.pool==pool)&(delta.source==source)&(delta.model=='panel_mean')].set_index('total')
            if table==0:
                expected=[f'{100*panel.loc[0].macro_f1_mean:.2f}',f'{100*panel.loc[100].macro_f1_mean:.2f}',
                          f'{100*delta.loc[120].macro_f1_delta:+.2f}',
                          f'[{100*delta.loc[120].macro_f1_sampling_low:+.2f}, {100*delta.loc[120].macro_f1_sampling_high:+.2f}]']
                assert cell[2:]==expected
            elif table==1:
                rank=frames['model_summary']
                rank=rank[(rank.pool==pool)&(rank.source==source)&(rank.total==120)]
                span=rank.pivot(index='model',columns='high_risk_percent',values='rank')
                assert cell[2:4]==['ConvNeXt-Tiny','ConvNeXt-Tiny']
                assert cell[4:]==[f'{rank.point_kendall_vs20.min():.3f}',f'{(span.max(axis=1)-span.min(axis=1)).max():.0f}']
            elif table==2:
                expected=[f'{100*panel.loc[0].failed_state_breadth_mean:.2f}',f'{100*panel.loc[100].failed_state_breadth_mean:.2f}',
                          f'{100*panel.loc[100].equal_error_confusion_delta_mean:+.2f}',f'{100*panel.loc[100].equal_error_cluster_delta_mean:+.2f}']
                assert cell[2:]==expected
            else:
                expected=[f'{100*delta.loc[120].macro_f1_delta:+.2f}',f'{100*delta.loc[480].macro_f1_delta:+.2f}',
                          f'[{100*delta.loc[480].macro_f1_sampling_low:+.2f}, {100*delta.loc[480].macro_f1_sampling_high:+.2f}]']
                assert cell[2:]==expected
            table_cells+=len(cell)-2
    with ZipFile(BASE/'report_zh.docx') as z:
        doc=ET.fromstring(z.read('word/document.xml'))
        namespace={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                   'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        assert len(doc.findall('.//w:tbl',namespace))==4
        assert len(doc.findall('.//a:blip',namespace))==7
    result=dict(status='pass',candidate_quota_targets_checked=quota_targets,
        classifier_scores_reconstructed=reconstructed_scores,direct_sklearn_checks=direct_checks,
        summary_numeric_checks=checks,max_raw_float32_difference=max_gap,
        report_table_data_cells_verified=table_cells,
        freeze_files_unchanged=len(freeze),svg_figures=7,docx_tables=4,docx_figures=7,
        mean_top2_stable_at_all_126_main_points=bool(main_top2),
        intervals='95% repeated-sampling ranges conditional on fixed finite candidate pools',
        reports=['report_zh.md','report_zh.docx'])
    (BASE/'final_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
