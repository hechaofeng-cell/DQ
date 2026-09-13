#!/usr/bin/env python3
"""Summarize dog_v3 feature states and identify low-frequency gaps."""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qe_quality.dta.io import atomic_csv, atomic_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schema', type=Path, default=ROOT/'configs/dog_feature_schema_v3.json')
    ap.add_argument('--results', type=Path, default=ROOT/'artifacts/dog500_features_v3_20260913')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    schema=json.loads(args.schema.read_text(encoding='utf-8'))
    defs=schema['universal_features']+schema.get('category_specific_features',[])
    rows=list(csv.DictReader((args.results/'results.csv').open(encoding='utf-8-sig',newline='')))
    out=args.output or args.results/'distribution'
    out.mkdir(parents=True,exist_ok=True)
    distributions=[]; gaps=[]; unknown=[]
    for d in defs:
        key=d['feature_id']; counts=Counter()
        for row in rows:
            value=row.get(key,'')
            if d['value_type']=='set':
                try: values=json.loads(value) if value else []
                except json.JSONDecodeError: values=[]
                counts.update(values or ['__missing__'])
            else: counts[value or '__missing__'] += 1
        denominator=len(rows)
        for state in d['possible_values']:
            n=counts[state]
            distributions.append({'feature_id':key,'feature_name_zh':d.get('feature_name_zh',''),'state':state,'count':n,'rate':round(n/denominator,6) if denominator else 0,'is_unknown':state=='unknown','band':'missing' if n==0 else ('sparse' if n<10 else ('weak' if n<30 else 'covered'))})
            if state not in {'unknown','other'} and n<30:
                gaps.append({'feature_id':key,'feature_name_zh':d.get('feature_name_zh',''),'state':state,'count':n,'rate':round(n/denominator,6) if denominator else 0,'gap_type':'missing' if n==0 else ('sparse' if n<10 else 'weak'),'target_count':10 if n<10 else 30,'needed':(10 if n<10 else 30)-n})
        unknown.append({'feature_id':key,'feature_name_zh':d.get('feature_name_zh',''),'unknown_count':counts['unknown'],'unknown_rate':round(counts['unknown']/denominator,6) if denominator else 0,'missing_row_count':counts['__missing__']})
    atomic_csv(out/'feature_state_distribution.csv',distributions,list(distributions[0]))
    atomic_csv(out/'low_frequency_gaps.csv',gaps,list(gaps[0]) if gaps else ['feature_id','feature_name_zh','state','count','rate','gap_type','target_count','needed'])
    atomic_csv(out/'unknown_summary.csv',unknown,list(unknown[0]))
    index=sum(min(r['count']/30,1) for r in distributions if not r['is_unknown'] and r['state']!='other')/sum(not r['is_unknown'] and r['state']!='other' for r in distributions)
    report={'schema_version':schema['schema_version'],'images':len(rows),'strict_ok':sum(r.get('feature_status')=='ok' for r in rows),'strict_rate':sum(r.get('feature_status')=='ok' for r in rows)/len(rows) if rows else 0,'evaluable_states':sum(not r['is_unknown'] and r['state']!='other' for r in distributions),'coverage_index':index,'gap_count':len(gaps),'missing_state_count':sum(r['gap_type']=='missing' for r in gaps),'sparse_state_count':sum(r['gap_type']=='sparse' for r in gaps),'weak_state_count':sum(r['gap_type']=='weak' for r in gaps),'outputs':{'distribution':'feature_state_distribution.csv','gaps':'low_frequency_gaps.csv','unknown':'unknown_summary.csv'}}
    atomic_json(out/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
