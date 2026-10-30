#!/usr/bin/env python3
"""Independent histogram reconstruction, new controls tables and Word audits."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from run_random_stratification_ablation import BASE,ROOT,POOLS,CLASSES,MODELS,SEEDS,VARIANTS
from automatic_strengthening_common import labels_for
from analyze_sixclass_difficulty import MAPS
from run_class_stratified_error_controls import holm

PAPER=ROOT/'docs/visual_state_testing_paper_controls_20261006.md'
OLD=ROOT/'docs/visual_state_testing_paper.md'


def table(text,n):
    return text.split(f'**表{n} ',1)[1].split('\n\n',1)[1].split('\n\n',1)[0]


def histogram_scores(y,pred,chosen):
    # chosen repeat x layer x targets; use one histogram, independently of main class loops.
    count,tiers,n=chosen.shape
    cells=y[chosen]*6+pred[chosen]
    cells+=np.arange(count*tiers).reshape(count,tiers,1)*36
    cm=np.bincount(cells.reshape(-1),minlength=count*tiers*36).reshape(count,tiers,6,6)
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    support=cm.sum(-1);incoming=cm.sum(-2)
    return (2*tp/(support+incoming)).mean(-1),tp/support


def main():
    audit_count=0;direct_checks=0;member_count=0;max_gap=0.
    def close(a,b,tol=1e-12):
        nonlocal audit_count,max_gap
        a,b=np.asarray(a,float),np.asarray(b,float);assert a.shape==b.shape,(a.shape,b.shape)
        gap=float(np.max(np.abs(a-b))) if a.size else 0.
        assert gap<=tol,(gap,tol);audit_count+=a.size;max_gap=max(max_gap,gap)
    random=pd.read_csv(BASE/'tables/random_summary.csv')
    ablation=pd.read_csv(BASE/'tables/ablation_summary.csv')
    membership=pd.read_csv(BASE/'tables/membership.csv')
    profiles=pd.read_csv(BASE/'tables/ablation_run_metrics.csv')
    assert len(random)==6 and len(ablation)==42 and len(profiles)==5040 and len(membership)==504
    for pool,path in POOLS.items():
        rows=pd.read_csv(path/'manifests/candidate_pool.csv');y=rows.label.to_numpy(dtype=int)
        assert rows.class_name.map(dict(zip(CLASSES,range(6)))).tolist()==y.tolist()
        with np.load(BASE/'random_strata'/f'{pool}_orders.npz') as z:
            orders=z['orders'];assert z['sample_ids'].tolist()==rows.sample_id.tolist()
        repeats,c,tiers,k=orders.shape;assert (repeats,c,tiers)==(10000,6,5)
        assert np.all(np.sort(orders.reshape(repeats,-1),axis=1)==np.arange(len(rows)))
        assert np.all(y[orders]==np.arange(6)[None,:,None,None])
        member_count+=orders.size
        with np.load(BASE/'random_strata'/f'{pool}_metrics.npz') as z:
            score=z['f1'];recall=z['class_recall'];panel=z['panel_curve'];null=z['panel_delta']
            assert score.shape==(10000,24,5)
        pred=[]
        for run,(model,seed) in enumerate(itertools.product(MODELS,SEEDS)):
            with np.load(path/'benchmark/runs'/model/str(seed)/'candidate_outputs.npz') as z:prediction=z['logits'].argmax(1)
            pred.append(prediction)
            for start in range(0,10000,250):
                chosen=orders[start:start+250].transpose(0,2,1,3).reshape(-1,5,6*k)
                f1,rr=histogram_scores(y,prediction,chosen)
                close(f1,score[start:start+250,run])
                close(rr,recall[start:start+250,run],tol=1e-7)
            for repetition in (0,17,9999):
                for tier in (0,2,4):
                    selected=orders[repetition,:,tier].reshape(-1)
                    _,_,f1,_=precision_recall_fscore_support(y[selected],prediction[selected],labels=range(6),zero_division=0)
                    close(f1.mean(),score[repetition,run,tier]);direct_checks+=1
        pred=np.array(pred);close(panel,score.mean(1));close(null,panel[:,-1]-panel[:,0])
        labels,failures=labels_for(rows.to_dict('records'),path);assert not failures
        for source in ('local','online'):
            with np.load(BASE/'ablation'/f'{pool}_{source}.npz') as z:
                full_scores=z['f1'];variant_orders=z['orders'];parts=z['risk_parts']
                assert z['variants'].tolist()==VARIANTS and z['sample_ids'].tolist()==rows.sample_id.tolist()
            native=[]
            for row in rows.to_dict('records'):
                area=float(row['bbox_area_ratio']);size=0 if area>=.10 else 1 if area>=.03 else 2 if area>=.01 else 3
                item=[size]
                for field,mapping in MAPS.items():
                    value=labels[source][row['sample_id']][field]
                    item.append(max(mapping.values())/2 if value=='unknown' else mapping[value])
                native.append(item)
            close(parts,native)
            for vi,variant in enumerate(VARIANTS):
                scores=parts.sum(1)
                if variant=='equal_normalized':scores=np.array([v[0]/3+sum(v[j]/max(mapping.values()) for j,mapping in enumerate(MAPS.values(),1)) for v in parts])
                elif variant!='full':scores-=parts[:,['size',*MAPS].index(variant.removeprefix('drop_'))]
                for ci,name in enumerate(CLASSES):
                    indices=np.flatnonzero(y==ci)
                    expected=np.array(sorted(indices.tolist(),key=lambda i:(scores[i],rows.sample_id.iloc[i]))).reshape(5,k)
                    assert np.array_equal(variant_orders[vi,ci],expected)
                    for ti in (0,4):
                        old=set(variant_orders[0,ci,ti]);new=set(expected[ti])
                        r=membership[(membership.pool==pool)&(membership.source==source)&(membership.variant==variant)&(membership.class_name==name)&(membership.tier==ti+1)].iloc[0]
                        close([r.jaccard,r.retention],[len(old&new)/len(old|new),len(old&new)/len(old)])
                chosen=variant_orders[vi].transpose(1,0,2).reshape(1,5,6*k)
                for run,prediction in enumerate(pred):
                    f1,_=histogram_scores(y,prediction,chosen)
                    close(f1[0],full_scores[vi,run])
                delta=full_scores[vi,:,-1]-full_scores[vi,:,0]
                r=ablation[(ablation.pool==pool)&(ablation.source==source)&(ablation.variant==variant)].iloc[0]
                close(r.mean_delta_auc,delta.mean())
                close(r.change_vs_full,(delta-(full_scores[0,:,-1]-full_scores[0,:,0])).mean())
                assert r.negative_runs==int((delta<0).sum()) and r.positive_runs==int((delta>0).sum())
                means=delta.reshape(8,3).mean(1)
                exact=np.mean([abs(np.mean(means*sign))>=abs(means.mean())-1e-15 for sign in itertools.product((-1,1),repeat=8)])
                close(r.permutation_p,exact)
            r=random[(random.pool==pool)&(random.source==source)].iloc[0]
            delta=(full_scores[0,:,-1]-full_scores[0,:,0]).mean()
            close(r.observed_delta,delta)
            close([r.random_mean,r.random_sd,r.random_sampling_low,r.random_sampling_high,r.random_abs_max],
                  np.r_[null.mean(),null.std(ddof=1),np.percentile(null,[2.5,97.5]),np.abs(null).max()])
            extreme=int((np.abs(null)>=abs(delta)-1e-12).sum());assert extreme==r.extreme_count
            close(r.mc_two_sided_p,(extreme+1)/10001)
        p=ablation[(ablation.pool==pool)&(ablation.variant!='full')]
        close(p.holm_twelve_p,holm(p.permutation_p.tolist()))
        print(f'{pool}: random histogram reconstruction, quotas, all ablations, membership and p PASS',flush=True)
    close(random.holm_six_p,holm(random.mc_two_sided_p.tolist()))
    text=PAPER.read_text();old=OLD.read_text()
    for n in range(1,26):assert table(text,n)==table(old,n),n
    for left,right in [('## 2 相关工作','## 3 方法'),('## 参考文献','## 附录A')]:
        assert text.split(left,1)[1].split(right,1)[0]==old.split(left,1)[1].split(right,1)[0]
    assert re.findall(r'^\*\*表(\d+) ',text,re.M)==[str(i) for i in range(1,29)]
    assert re.findall(r'^!\[图(\d+) ',text,re.M)==[str(i) for i in range(1,18)]
    assert text.count('$$')==6
    # New numeric table cells trace directly to detailed outputs; old cells unchanged above.
    cells=0
    for line in table(text,26).splitlines()[2:]:
        p=[v.strip() for v in line.strip('|').split('|')];pool=p[0];source={'本地':'local','线上':'online'}[p[1]]
        r=random[(random.pool==pool)&(random.source==source)].iloc[0]
        expected=[f'{100*r.observed_delta:+.2f}',f'{100*r.random_mean:+.2f}',f'[{100*r.random_sampling_low:+.2f}, {100*r.random_sampling_high:+.2f}]',str(int(r.extreme_count)),f'{r.holm_six_p:.5f}']
        assert p[2:]==expected;cells+=5
    for vi,line in enumerate(table(text,27).splitlines()[2:]):
        p=[v.strip() for v in line.strip('|').split('|')]
        expected=[f'{100*ablation[(ablation.pool==pool)&(ablation.source==source)&(ablation.variant==VARIANTS[vi])].mean_delta_auc.iloc[0]:+.2f}' for pool in ('B','C','D') for source in ('local','online')]
        assert p[1:]==expected;cells+=6
    ms=pd.read_csv(BASE/'tables/membership_summary.csv')
    for line in table(text,28).splitlines()[2:]:
        p=[v.strip() for v in line.strip('|').split('|')];pool=p[0];source={'本地':'local','线上':'online'}[p[1]]
        a=ablation[(ablation.pool==pool)&(ablation.source==source)].set_index('variant')
        m=ms[(ms.pool==pool)&(ms.source==source)&(ms.variant=='drop_occlusion_level')&(ms.tier==5)].iloc[0]
        expected=[f'{100*a.loc["full"].mean_delta_auc:+.2f}',f'{100*a.loc["drop_occlusion_level"].mean_delta_auc:+.2f}',
            f'{100*a.loc["drop_occlusion_level"].change_vs_full:+.2f}',f'{m.jaccard:.3f}',f'{100*m.retention:.2f}']
        assert p[2:]==expected;cells+=5
    for file in (BASE/'figures').glob('*.svg'):
        e=ET.parse(file).getroot();ns={'s':'http://www.w3.org/2000/svg'}
        assert e.findall('.//s:text',ns) and not e.findall('.//s:image',ns)
        assert file.with_suffix('.png').exists() and file.with_suffix('.py').exists()
    with ZipFile(PAPER.with_suffix('.docx')) as z:
        e=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math',
            'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        assert len(e.findall('.//w:tbl',ns))==29
        assert len(e.findall('.//a:blip',ns))==17
        displays=len(e.findall('.//m:oMathPara',ns))
        # The three displayed equations are inline oMath nodes in numbered tab paragraphs.
        assert len(e.findall('.//m:oMath',ns))==len(re.findall(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)',text))+3
        for t in e.findall('.//w:tbl',ns):
            assert t.find('./w:tblPr/w:tblBorders/w:left',ns).get('{'+ns['w']+'}val')=='nil'
        texts=[x.text or '' for x in e.findall('.//w:t',ns)]
        assert not any('$' in t or '/home/hcf' in t for t in texts)
        bookmarks={x.get('{'+ns['w']+'}name') for x in e.findall('.//w:bookmarkStart',ns)}
        assert {f'ref_{i}' for i in range(1,14)}<=bookmarks
    freeze=json.loads((BASE/'input_freeze.sha256.json').read_text())
    for path,digest in freeze.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    result=dict(status='pass',checked_random_target_memberships=member_count,
        numerical_values_reconstructed=audit_count,direct_sklearn_checks=direct_checks,max_storage_difference=max_gap,
        input_hashes_unchanged=len(freeze),new_manuscript_numeric_cells=cells,
        retained_tables_unchanged=25,related_work_and_bibliography_unchanged=True,
        figures=17,new_svg_figures=2,word_three_line_tables=29,
        manuscript=str(PAPER),word=str(PAPER.with_suffix('.docx')),
        rendering='OOXML structure verified; page layout not rendered',
        interval_scope='random-grouping ranges and architecture/seed bootstrap remain separate',
        all_state_accuracy_validated=False)
    (BASE/'final_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
