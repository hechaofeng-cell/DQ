#!/usr/bin/env python3
"""Select a fixed v3.1 supplement from the re-annotated candidate pool."""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qe_quality.dta.io import atomic_csv, atomic_json

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={"000000244215","000000249619","000000275268","000000516265"}
STRATA=("tiny_target","small_target","multiple_dogs","boundary_target","person_cooccurrence")

def values(row,key):
    value=row.get(key,'')
    if not value:return []
    if value.startswith('['):
        try:return json.loads(value)
        except json.JSONDecodeError:return []
    return [value]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--schema',type=Path,default=ROOT/'configs/dog_feature_schema_v3_1.json');ap.add_argument('--baseline',type=Path,default=ROOT/'artifacts/dog500_features_v3_1_20260913/results.csv');ap.add_argument('--candidates',type=Path,default=ROOT/'artifacts/dog_candidate400_features_v3_1_20260913/results.csv');ap.add_argument('--metadata',type=Path,default=ROOT/'data/coco2017/dog_candidate400_20260912/candidate_manifest.csv');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/dog_v3_1_supplement_selection_20260913');ap.add_argument('--count',type=int,default=100)
    a=ap.parse_args();schema=json.loads(a.schema.read_text());defs=schema['universal_features'];keys=[d['feature_id'] for d in defs];allowed={d['feature_id']:set(d['possible_values']) for d in defs};types={d['feature_id']:d['value_type'] for d in defs}
    base=list(csv.DictReader(a.baseline.open(encoding='utf-8-sig',newline='')));cand=list(csv.DictReader(a.candidates.open(encoding='utf-8-sig',newline='')));meta={r['image_id']:r for r in csv.DictReader(a.metadata.open(encoding='utf-8-sig',newline=''))}
    def valid(r):return r.get('feature_status')=='ok' and r['image_id'] not in EXCLUDED
    cand=[r for r in cand if valid(r)]
    counts=Counter();
    for r in base:
        for k in keys:
            for v in values(r,k):counts[k,v]+=1
    gaps=[]
    for d in defs:
        k=d['feature_id']
        for state in d['possible_values']:
            if state in ('unknown','other'):continue
            n=counts[k,state]
            if n<30:gaps.append({'feature_id':k,'state':state,'baseline_count':n,'target_count':10 if n<10 else 30})
    remaining={(g['feature_id'],g['state']):max(0,g['target_count']-g['baseline_count']) for g in gaps}
    selected=[];strata=Counter()
    # First satisfy minimums with the highest current deficit contribution.
    def gain(r):
        score=0
        for k,s in remaining:
            if remaining[k,s]>0 and s in values(r,k):score+=1+remaining[k,s]/30
        return score
    pool=cand[:]
    while len(selected)<a.count and pool:
        eligible=[r for r in pool if strata[meta.get(r['image_id'],{}).get('selection_stratum','')] < 30]
        need_min=[st for st in STRATA if strata[st]<15]
        if need_min:
            eligible=[r for r in eligible if meta.get(r['image_id'],{}).get('selection_stratum') in need_min] or eligible
        best=max(eligible,key=lambda r:(gain(r),-int(r.get('feature_unknown_count') or 0),r['image_id']))
        selected.append(best);pool.remove(best);st=meta.get(best['image_id'],{}).get('selection_stratum','unknown');strata[st]+=1
        for k,s in remaining:
            if remaining[k,s]>0 and s in values(best,k):remaining[k,s]-=1
    a.output.mkdir(parents=True,exist_ok=False)
    selected_rows=[]
    for rank,r in enumerate(selected,1):
        st=meta.get(r['image_id'],{}).get('selection_stratum','unknown');contrib=[f'{k}={s}' for k,s in remaining if s in values(r,k)]
        selected_rows.append({'selection_rank':rank,'image_id':r['image_id'],'selection_stratum':st,'feature_unknown_count':r.get('feature_unknown_count',''),'gap_states':json.dumps(contrib,ensure_ascii=False),'image_id_source':'candidate400_v3.1'})
    atomic_csv(a.output/'selected_candidates.csv',selected_rows,list(selected_rows[0]))
    post=Counter(counts)
    for r in selected:
        for k in keys:
            for v in values(r,k):post[k,v]+=1
    coverage=[]
    for g in gaps:
        n=post[g['feature_id'],g['state']];coverage.append({**g,'selected_count':n-counts[g['feature_id'],g['state']],'post_count':n,'target_met':n>=g['target_count'],'remaining_needed':max(0,g['target_count']-n)})
    atomic_csv(a.output/'coverage_after_selection.csv',coverage,list(coverage[0]))
    atomic_csv(a.output/'remaining_gaps.csv',[r for r in coverage if not r['target_met']],list(coverage[0]))
    index=sum(min(post[d['feature_id'],s]/30,1) for d in defs for s in d['possible_values'] if s not in ('unknown','other'))/sum(s not in ('unknown','other') for d in defs for s in d['possible_values'])
    report={'schema_version':schema['schema_version'],'baseline_images':len(base),'candidate_pool':len(cand),'excluded_known_nonliving':len(EXCLUDED),'selected_count':len(selected),'strata':dict(strata),'baseline_gap_count':len(gaps),'resolved_gaps':sum(r['target_met'] for r in coverage),'remaining_gaps':sum(not r['target_met'] for r in coverage),'post_selection_coverage_index':index,'outputs':{'selected':'selected_candidates.csv','coverage':'coverage_after_selection.csv','remaining':'remaining_gaps.csv'}}
    atomic_json(a.output/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
