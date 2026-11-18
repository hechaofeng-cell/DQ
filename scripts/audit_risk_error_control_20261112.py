"""Reaggregate persisted CSVs independently of the experiment's summary functions."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import run_risk_error_control_20261112 as r

OUT,cc,study=r.OUT,r.cc,r.study


def freeze_code():
    cc.freeze(OUT/'audit_code_freeze.json',dict(files={str(Path(__file__)):study.sha(Path(__file__))}))


def primary():
    freeze_code();cc.verify_freeze(OUT/'primary_output_freeze.json')
    fields=['risk_source','coverage_source','threshold_mode','evaluator','reference','path','method','selection_seed']
    summed=defaultdict(lambda:[0,0]);universe={};count=0
    import csv
    with (OUT/'primary_diagnostic.csv').open(encoding='utf-8-sig',newline='') as handle:
        for v in csv.DictReader(handle):
            key=tuple(v[f] for f in fields);n=int(v['common_n']);yes=int(v['confirmed'])
            assert 0<=yes<=n
            summed[key][0]+=yes;summed[key][1]+=n
            uk=tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','class_name','selection_seed','model','evaluator','reference'])
            if uk in universe:assert universe[uk]==n
            else:universe[uk]=n
            count+=1
    for v in study.read(OUT/'primary_per_start.csv'):
        key=tuple(v[f] for f in fields)
        assert summed[key]==[int(float(v['confirmed'])),int(float(v['common_n']))]
    summary=study.read(OUT/'primary_summary.csv')
    for v in summary:
        base=tuple(v[f] for f in fields[:6]);a=[];b=[]
        for seed in r.previous.SEEDS:
            x=summed[base+(v['method'],str(seed))];y=summed[base+(v['comparator'],str(seed))]
            assert x[1]==y[1];a.append(x[0]);b.append(y[0])
        delta=np.array(a)-np.array(b)
        assert abs(float(delta.mean())-float(v['mean_difference']))<1e-10
        assert int((delta>0).sum())==int(v['positive_starts'])
        assert int((delta==0).sum())==int(v['ties'])
        assert int((delta<0).sum())==int(v['negative_starts'])
    # Selected error rate must reproduce the same frozen selection's old value.
    original={(v['risk_source'],v['coverage_source'],v['threshold_mode'],v['class_name'],v['selection_seed'],v['model'],v['method']):float(v['error_pct'])
              for v in study.read(r.previous.OUT/'diagnostic.csv') if v['method'] in r.METHODS and v['evaluator']==v['reference']=='A'}
    error_rows=0
    for v in study.read(OUT/'actual_error.csv'):
        if v['Q']!='0':continue
        key=tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','class_name','selection_seed','model','method'])
        assert abs(float(v['error_pct'])-original[key])<1e-10;error_rows+=1
    r.dump('primary_independent_audit.json',dict(status='pass',raw_rows=count,per_start_aggregates=len(summed),
        summary_rows=len(summary),shared_universe_groups=len(universe),frozen_error_rows_reproduced=error_rows))


def matched():
    freeze_code();cc.verify_freeze(OUT/'matched_output_freeze.json')
    rows=study.read(OUT/'matched_per_draw.csv')
    fields=['risk_source','coverage_source','threshold_mode','selection_seed','evaluator','reference','phase','method']
    grouped=defaultdict(list);balance={};nrows=0
    for v in rows:
        key=tuple(v[f] for f in fields);grouped[key].append(v)
        unit=tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','selection_seed','prediction_seed','repeat','evaluator','reference'])
        n=float(v['common_n'])
        if unit in balance:assert balance[unit]==n
        else:balance[unit]=n
    aggregate={}
    numeric=['confirmed','common_n','original_common_n','mean_selected_n','error_pct','total_error_targets']
    for key,rr in grouped.items():
        assert len(rr)==3*r.REPEATS
        aggregate[key]={f:sum(float(v[f]) for v in rr)/len(rr) for f in numeric}
    for v in study.read(OUT/'matched_per_start.csv'):
        key=tuple(v[f] for f in fields)
        for f in numeric:assert abs(float(v[f])-aggregate[key][f])<1e-9
    summary=study.read(OUT/'matched_summary.csv')
    for v in summary:
        left=[];right=[]
        for seed in r.previous.SEEDS:
            prefix=(v['risk_source'],v['coverage_source'],v['threshold_mode'],str(seed),v['evaluator'],v['reference'],v['phase'])
            x=aggregate[prefix+(v['method'],)];y=aggregate[prefix+(v['comparator'],)]
            left.append(x['confirmed']);right.append(y['confirmed'])
            if v['phase']=='matched':
                assert abs(x['mean_selected_n']-y['mean_selected_n'])<1e-10
                assert abs(x['total_error_targets']-y['total_error_targets'])<1e-10
                assert abs(x['error_pct']-y['error_pct'])<1e-10
        delta=np.array(left)-np.array(right)
        assert abs(float(delta.mean())-float(v['mean_difference']))<1e-9
        assert int((delta>1e-10).sum())==int(v['positive_starts'])
        assert int((np.abs(delta)<=1e-10).sum())==int(v['ties'])
        assert int((delta<-1e-10).sum())==int(v['negative_starts'])
    units={}
    for v in study.read(OUT/'matched_balance.csv'):
        key=tuple(v[f] for f in ['risk_source','coverage_source','threshold_mode','class_name','selection_seed','model','prediction_seed'])
        signature=tuple(v[f] for f in ['matched_n','matched_errors']+[f'Q{q}_matched_n' for q in range(1,6)])
        if key in units:assert units[key]==signature
        else:units[key]=signature
        assert sum(int(v[f'Q{q}_matched_n']) for q in range(1,6))==int(v['matched_n'])
        assert 0<=int(v['matched_errors'])<=int(v['matched_n'])<=200;nrows+=1
    r.dump('matched_independent_audit.json',dict(status='pass',draw_rows=len(rows),per_start_groups=len(grouped),
        summary_rows=len(summary),balanced_unit_rows=nrows,base_matching_units=len(units),
        identical_common_universe_original_and_matched=True,error_counts_sizes_and_rates_equal_after_matching=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['primary','matched'])
    args=parser.parse_args();{'primary':primary,'matched':matched}[args.stage]()
