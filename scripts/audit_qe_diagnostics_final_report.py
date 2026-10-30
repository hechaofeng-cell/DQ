#!/usr/bin/env python3
"""Finalize report audit after the last fine-score presentation update."""
import json
from pathlib import Path
import numpy as np
from docx import Document
from audit_qe_feature_risk_diagnostics import ROOT,OUT,DIMS,CLASSES,read,digest,close,audit_report


def main():
    prior=json.loads((OUT/'independent_diagnostics_audit.json').read_text());assert prior['status']=='pass'
    old_freeze=json.loads((OUT/'independent_diagnostics_freeze.sha256.json').read_text())
    for name in ('result_freeze.sha256.json','selection_analysis_freeze.sha256.json','inventory_freeze.sha256.json'):
        path=str(OUT/name);assert digest(path)==old_freeze[path],('audited numeric freeze changed',name)
    final_freeze=json.loads((OUT/'report_freeze.sha256.json').read_text())
    for path,expected in final_freeze.items():assert digest(path)==expected,('final report frozen dependency changed',path)
    wide=read(OUT/'target_scores.csv');lookup={(r['annotation_backend'],r['sample_id']):r for r in wide}
    dims={(r['annotation_backend'],r['sample_id']):{d:tuple(float(r[d+'_risk_'+bound+'_0_100'])/100 for bound in ('lower','midpoint','upper'))+(float(r[d+'_known_weight']),) for d in DIMS} for r in wide}
    audit_report(wide,lookup,dims,read(OUT/'risk_tier_error_relationship.csv'))
    report=(OUT/'report.md').read_text()
    original=read(ROOT/'artifacts/fresh_vlm_feature_scoring_20261012/scores.csv')
    summary=read(OUT/'fine_score_summary.csv')
    for c in CLASSES:
        rows=[r for r in original if r['class_name']==c];assert len(rows)==10
        listed=next(r for r in summary if r['class_name']==c)
        mean=float(np.mean([float(r['score_midpoint']) for r in rows]))
        visibility=float(np.mean([20*float(r['visibility_midpoint']) for r in rows]))
        evaluable=sum(r['evaluable']=='True' for r in rows)
        close(listed['mean_score_midpoint'],mean,(c,'fresh mean'))
        close(listed['visibility_mean_contribution_points'],visibility,(c,'fresh visibility'))
        assert int(listed['evaluable_targets'])==evaluable
        cells=[c,listed['fine_fields_per_target'],'10/1300',str(evaluable),f'{mean:.2f}',f'{visibility:.2f}']
        assert '| '+' | '.join(cells)+' |' in report
    doc=Document(OUT/'report.docx');assert len(doc.tables)==7
    result=dict(status='pass',core_numeric_freezes_unchanged=True,final_report_frozen_dependencies=len(final_freeze),
        all_seven_tables_verified=True,fresh_six_class_score_table_recomputed_from_original60=True,
        fine_sample_unknown_midpoint_preserved=True,word_report_tables=7,
        earlier_core_audit=str(OUT/'independent_diagnostics_audit.json'))
    (OUT/'independent_final_report_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    paths=[Path(__file__).resolve(),OUT/'independent_final_report_audit.json',OUT/'report_freeze.sha256.json',OUT/'independent_diagnostics_audit.json',OUT/'independent_diagnostics_freeze.sha256.json']
    (OUT/'independent_final_report_freeze.sha256.json').write_text(json.dumps({str(p):digest(p) for p in paths},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
