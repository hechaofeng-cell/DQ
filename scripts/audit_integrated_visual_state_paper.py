#!/usr/bin/env python3
"""Audit retained nine tables and all six added result tables against cached raw outputs."""
import json,re,sys
from pathlib import Path
import pandas as pd
import audit_visual_state_paper_20261005 as old
from dual_vlm_common import sha256,write_json,now

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/visual_state_testing_empirical_paper_integrated_20261005.md'
D=ROOT/'artifacts/objects365_independent_confirmation_20261005'
OUT=ROOT/'artifacts/label_disagreement_sensitivity_20261005'
S={'本地':'local','线上':'online'}


def table(text,n):
    part=text.split(f'**表{n} ')[1].split('\n\n',1)[1]
    lines=[]
    for line in part.splitlines():
        if not line.startswith('|'):break
        lines.append([c.strip() for c in line.strip('|').split('|')])
    return lines[2:]


def main():
    old.PAPER=PAPER;old.run()
    text=PAPER.read_text();before=old.checked
    full=pd.read_csv(D/'confirmation/tables/full_stratum_metrics.csv')
    deltas=pd.read_csv(D/'confirmation/tables/difficulty_contrasts.csv')
    for row in table(text,10):
        for j,source in enumerate(S.values(),1):
            if row[0]=='Q5−Q1':
                value=deltas[(deltas.source==source)&(deltas.unknown_mode=='midpoint')].mean_delta_auc.iloc[0]
            else:value=full[(full.source==source)&(full.unknown_mode=='midpoint')&(full.tier==int(row[0][1:]))].macro_f1.mean()
            old.check(row[j],100*value)
    for row in table(text,11):
        source=S[row[1]];metric=old.METRICS[row[2]]
        if row[0]=='共同配额':
            f=pd.read_csv(D/'comparison/tables/comparisons.csv');r=f[(f.source==source)&(f.metric==metric)&(f.baseline=='random')].iloc[0];key='holm_primary4_p'
        else:
            f=pd.read_csv(D/'confirmation/tables/matched_equal_error_comparisons.csv');r=f[(f.source==source)&(f.metric==metric)].iloc[0];key='holm_four_p'
        old.check(row[3],r.mean_delta_auc,4);old.check(row[4],r[key],5)
    transfer=json.loads((D/'transfer/summary.json').read_text());states=pd.read_csv(D/'transfer/tables/state_validation.csv')
    for row in table(text,12):
        source=S[row[0]];r=next(r for r in transfer['summary'] if r['source']==source);v=next(r for r in transfer['comparisons'] if r['source']==source)
        old.check(row[1],r['frozen_state_models'],0);old.check(row[2],r['evaluable_state_models'],0)
        f=states[(states.source==source)&states.B_evaluable].groupby(['model','class_name','feature','value']).B_gap.mean()
        old.check(row[3],int((f<0).sum()),0);old.check(row[4],100*v['mean_delta_auc']);old.check(row[5],v['holm_two_p'],5)
    models=pd.read_csv(D/'confirmation/tables/model_robustness.csv')
    for row in table(text,13):
        for j,source in enumerate(S.values(),1):
            r=models[(models.source==source)&(models.model==old.MODELS[row[0]])].iloc[0]
            old.check(row[j],100*r.high_risk_f1)
    summary=pd.read_csv(OUT/'tables/risk_summary.csv');dist=pd.read_csv(OUT/'tables/retention_distribution.csv')
    for row in table(text,14):
        risk=summary[(summary.pool==row[0])&(summary.source=='local')&(summary.group=='risk_agree')].iloc[0]
        shared=summary[(summary.pool==row[0])&(summary.source=='local')&(summary.group=='shared_agree')].iloc[0]
        old.check(row[1],risk.total,0);old.check(row[2],risk.retained,0);old.check(row[3].rstrip('%'),100*risk.retention)
        old.check(row[4],shared.retained,0);old.check(row[5].rstrip('%'),100*shared.retention)
        f=dist[(dist.pool==row[0])&(dist.source=='local')&(dist.group=='risk_agree')&(dist.tier==5)]
        old.check(row[6],f.n.sum(),0)
        assert not summary[(summary.pool==row[0])&summary.group.isin(['risk_agree','shared_agree'])].evaluable.any()
    state=pd.read_csv(OUT/'tables/state_summary.csv')
    for row in table(text,15):
        r=state[(state.pool==row[0])&(state.source==S[row[1]])].iloc[0]
        for column,key in [(2,'evaluable_combos'),(3,'negative_full_same_combos'),(4,'negative_consensus_combos'),(5,'direction_flips')]:old.check(row[column],r[key],0)
        for column,key in [(6,'mean_full_same_combo_gap'),(7,'mean_consensus_gap')]:
            if r.evaluable_combos:old.check(row[column],100*r[key])
            else:assert row[column]=='—'
    # External/relative local links also resolve, not just the old absolute-file check.
    links=re.findall(r'\]\(([^)]+)\)',text);local=[]
    for value in links:
        if value.startswith(('https://','http://')):continue
        path=Path(value) if value.startswith('/') else PAPER.parent/value
        assert path.is_file(),str(path);local.append(str(path))
    mapping=json.loads((OUT/'input_freeze.sha256.json').read_text())
    assert all(sha256(path)==digest for path,digest in mapping.items())
    assert text.count('$$')%2==0 and '5400' in text
    result=dict(status='pass',manuscript=str(PAPER),checked_at=now(),tables=15,numeric_cells_checked=old.checked,
        new_numeric_cells=old.checked-before,local_links=len(local),figures=text.count('!['),input_freeze_files=len(mapping),
        no_new_training=True,no_new_vlm_requests=True,posthoc_consensus_analysis=True)
    write_json(OUT/'manuscript_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
