#!/usr/bin/env python3
"""Read-only gates on the new supplement and its explicitly incomplete stages."""
import json
import re
from pathlib import Path
import numpy as np
from automatic_strengthening_common import *


def main():
    from independent_confirmation_common import verify
    frozen=verify(BASE/'existing_evidence_execution_freeze.sha256.json')
    audit=json.loads((BASE/'automatic_audit/audit.json').read_text())
    assert audit['status']=='pass' and audit['targets']==5400 and not audit['input_errors']
    official=read_csv(BASE/'automatic_audit/official_attribute_reference.csv');assert len(official)==24
    assert len(read_csv(BASE/'automatic_audit/dual_vlm_agreement.csv'))==270
    summary=read_csv(BASE/'automatic_audit/reference_summary.csv');doc=(ROOT/'docs/visual_state_automatic_evidence_update_20261005.md').read_text()
    checked=0
    for row in summary:
        values=[r for r in official if r['source']==row['source'] and r['feature']==row['feature']]
        totals={field:sum(int(r[field]) for r in values) for field in ('known','unknown','tp','tn','fp','fn')}
        assert int(row['known'])==totals['known'] and int(row['unknown'])==totals['unknown']
        for field,value in [('exact_agreement',(totals['tp']+totals['tn'])/totals['known']),
            ('positive_agreement',totals['tp']/(totals['tp']+totals['fn'])),('negative_agreement',totals['tn']/(totals['tn']+totals['fp']))]:
            assert abs(float(row[field])-value)<1e-12;assert f'{value:.2%}' in doc;checked+=1
    replay=json.loads((BASE/'regression_replay/audit.json').read_text())
    assert replay['status']=='pass' and replay['scope']=='full' and replay['comparisons']==99
    suite=read_csv(BASE/'regression_suite.csv');assert len(suite)==99 and len({(r['pool'],r['sample_id']) for r in suite})==31
    assert all(r['matches']=='True' for r in read_csv(BASE/'regression_replay/replay_results.csv'))
    stats=json.loads((BASE/'statistics/status.json').read_text())
    assert all(not r['inference_allowed'] for r in stats['risk_audits'])
    assert all(not r['usable_for_confirmatory_claim'] for r in stats['auc_audits'])
    assert '未收敛' in doc and '不作为可信论文结果' in doc
    availability=json.loads((BASE/'audit/sheep_all_cached_splits.json').read_text())
    assert all(r['unused']==0 for r in availability['splits'])
    assert not list(D.glob('benchmark/**/candidate_outputs.npz'))
    links=re.findall(r'\]\((/[^)]+)\)',doc);assert all(Path(p).is_file() for p in links)
    result=dict(status='pass_with_documented_limits',frozen_files_verified=frozen,summary_metric_cells_checked=checked,
        input_targets_checked=5400,reference_summary_rows=4,regression_assertions=99,unique_regression_images=31,
        new_D_E_executed=False,GLMM_inference_allowed=False,local_links_checked=len(links))
    write_json(BASE/'supplement_audit.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
