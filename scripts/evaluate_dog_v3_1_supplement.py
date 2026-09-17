#!/usr/bin/env python3
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qe_quality.dta.io import atomic_csv,atomic_json
EXCLUDED={"000000140444","000000246880","000000446990","000000471513"}
def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def vals(row,k):
 v=row.get(k,'')
 if not v:return []
 if v.startswith('['):
  try:return json.loads(v)
  except:return []
 return [v]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--schema',type=Path,default=ROOT/'configs/dog_feature_schema_v3_1.json');ap.add_argument('--baseline',type=Path,default=ROOT/'artifacts/dog500_features_v3_1_20260913/results.csv');ap.add_argument('--selected',type=Path,default=ROOT/'artifacts/dog_v3_1_supplement_selection_20260913/selected_candidates.csv');ap.add_argument('--candidate-results',type=Path,default=ROOT/'artifacts/dog_candidate400_features_v3_1_20260913/results.csv');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/dog_v3_1_combined_supplement_20260913');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 schema=json.loads(a.schema.read_text());defs=schema['universal_features'];keys=[d['feature_id'] for d in defs];base=read(a.baseline);cand={r['image_id']:r for r in read(a.candidate_results)};sel=read(a.selected);selected=[cand[r['image_id']] for r in sel];assert len(selected)==100
 combined=base+selected;clean=[r for r in combined if r['image_id'] not in EXCLUDED];
 atomic_csv(a.output/'combined_results_600.csv',combined,list(combined[0]));atomic_csv(a.output/'clean_results_596.csv',clean,list(clean[0]));atomic_csv(a.output/'excluded_baseline.csv',[r for r in base if r['image_id'] in EXCLUDED],list(base[0]))
 distribution=[];gaps=[]
 for d in defs:
  k=d['feature_id'];c=Counter(v for r in clean for v in vals(r,k))
  for st in d['possible_values']:
   if st in ('unknown','other'):continue
   n=c[st];distribution.append({'feature_id':k,'feature_name_zh':d.get('feature_name_zh',''),'state':st,'count':n,'rate':n/len(clean),'is_unknown':False,'band':'missing' if n==0 else ('sparse' if n<10 else ('weak' if n<30 else 'covered'))})
   if n<30:gaps.append({'feature_id':k,'feature_name_zh':d.get('feature_name_zh',''),'state':st,'count':n,'rate':n/len(clean),'gap_type':'missing' if n==0 else ('sparse' if n<10 else 'weak'),'target_count':10 if n<10 else 30,'needed':(10 if n<10 else 30)-n})
 atomic_csv(a.output/'feature_state_distribution_596.csv',distribution,list(distribution[0]));atomic_csv(a.output/'remaining_gaps_596.csv',gaps,list(gaps[0]))
 evaluable=sum(not r['is_unknown'] for r in distribution);coverage=sum(min(r['count']/30,1) for r in distribution)/evaluable
 strict600=sum(r.get('feature_status')=='ok' for r in combined)/len(combined);strict596=sum(r.get('feature_status')=='ok' for r in clean)/len(clean)
 unknown=[]
 for d in defs:
  c=Counter(v for r in clean for v in vals(r,d['feature_id']));unknown.append({'feature_id':d['feature_id'],'feature_name_zh':d.get('feature_name_zh',''),'unknown_count':c['unknown'],'unknown_rate':c['unknown']/len(clean),'missing_count':c['__missing__']})
 atomic_csv(a.output/'unknown_summary_596.csv',unknown,list(unknown[0]))
 report={'schema_version':schema['schema_version'],'combined_images':len(combined),'clean_images':len(clean),'supplement_selected':len(selected),'excluded_baseline_nonliving':len(EXCLUDED),'strict_feature_success_rate_600':strict600,'strict_feature_success_rate_596':strict596,'evaluable_states':evaluable,'clean_coverage_index_596':coverage,'remaining_gap_count_596':len(gaps),'missing_states_596':sum(r['gap_type']=='missing' for r in gaps),'sparse_states_596':sum(r['gap_type']=='sparse' for r in gaps),'weak_states_596':sum(r['gap_type']=='weak' for r in gaps),'evaluation_definition':{'strict_feature_success_rate':'rows with feature_status=ok / rows','coverage_index':'mean(min(state_count/30,1)) excluding unknown and other','unknown_rate':'unknown count / clean images'},'outputs':{'combined':'combined_results_600.csv','clean':'clean_results_596.csv','distribution':'feature_state_distribution_596.csv','gaps':'remaining_gaps_596.csv','unknown':'unknown_summary_596.csv'}}
 atomic_json(a.output/'report.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
