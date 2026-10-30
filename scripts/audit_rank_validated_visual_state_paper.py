#!/usr/bin/env python3
"""Audit retained results, new ranking/reference tables, and source freezes."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
import audit_integrated_visual_state_paper as previous
from analyze_ranking_attribute_validation import OUT, ROOT, MODELS, top_two_shares, ranks
from dual_vlm_common import sha256, write_json, now

PAPER = ROOT / 'docs/visual_state_testing_empirical_paper_rank_validated_20261005.md'


def main():
    previous.PAPER = PAPER
    # Keep the previous manuscript's audit file intact.
    original_writer = previous.write_json
    previous.write_json = lambda path, data: original_writer(OUT / 'retained_results_audit.json', data)
    previous.main()
    text = PAPER.read_text()
    before = previous.old.checked
    r = pd.read_csv(OUT / 'tables/ranking_summary.csv')
    for cells in previous.table(text, 16):
        row = r[r.pool == cells[0]].iloc[0]
        for column, key, digits in [(1,'q5_per_class',0),(2,'tau_min',3),(3,'tau_max',3),(5,'max_rank_change',0)]:
            previous.old.check(cells[column],row[key],digits)
        assert cells[4] == f'{row.same_top2_pairs}/15'
    attributes = pd.read_csv(OUT / 'tables/attribute_metrics.csv')
    for cells in previous.table(text, 17):
        field = {'遮挡':'occlusion_level','截断':'truncation_level'}[cells[0]]
        row = attributes[(attributes.field==field)&(attributes.source==previous.S[cells[1]])
                         &(attributes.aggregation=='target_weighted')].iloc[0]
        assert cells[2] == f'{int(row.known)}/{int(row.planned)}'
        for column,key in [(3,'agreement'),(4,'precision'),(5,'recall'),(6,'f1'),(7,'specificity')]:
            previous.old.check(cells[column],100*row[key])
        previous.old.check(cells[8],row.kappa,3)
    # Independent exact formula checks from confusion counts for every class/pooled row.
    for row in attributes[attributes.aggregation!='class_equal'].itertuples():
        assert row.known == row.tp + row.tn + row.fp + row.fn
        assert np.isclose(row.f1,2*row.tp/(2*row.tp+row.fp+row.fn))
        assert np.isclose(row.precision,row.tp/(row.tp+row.fp))
        assert np.isclose(row.recall,row.tp/(row.tp+row.fn))
    # Guard against manufacturing a unique best model from complete ties.
    assert np.allclose(top_two_shares(np.ones(8)),.25)
    assert np.allclose(ranks(np.ones(8)),4.5)
    frame = pd.read_csv(OUT / 'tables/model_ranks.csv')
    assert len(frame)==144
    for (pool,source,mode),part in frame.groupby(['pool','source','unknown_mode']):
        assert len(set(part.macro_f1))==8
        assert part.sort_values('rank').model.iloc[:2].tolist()==['convnext_tiny','swin_t']
    pair = pd.read_csv(OUT / 'tables/model_pair_differences.csv')
    top_pair = pair[(pair.model_left=='convnext_tiny')&(pair.model_right=='swin_t')]
    assert ((top_pair[top_pair.pool.isin(['B','C'])].ci95_low<0)
            &(top_pair[top_pair.pool.isin(['B','C'])].ci95_high>0)).all()
    assert (top_pair[top_pair.pool=='D'].ci95_low>0).all()
    # All new figures must preserve editable text and vector cells/marks.
    figures=[]
    for name in ('rank_sensitivity','attribute_reference'):
        svg=OUT/'figures'/f'{name}.svg'
        root=ET.parse(svg).getroot();ns={'s':'http://www.w3.org/2000/svg'}
        text_count=len(root.findall('.//s:text',ns));image_count=len(root.findall('.//s:image',ns))
        assert text_count>0 and image_count==0 and root.get('viewBox')
        figures.append(dict(path=str(svg),editable_text=text_count,raster_images=image_count))
    frozen_count=0
    for file in ('input_freeze.sha256.json','technical_amendment.sha256.json'):
        mapping=json.loads((OUT/file).read_text())
        assert all(sha256(p)==digest for p,digest in mapping.items())
        frozen_count+=len(mapping)
    # Verify the new numeric prose against cached summary endpoints.
    bounds={'B':('100.00%','98.50%','99.80%'),'C':('99.50%','99.95%','88.35%','97.90%'),
            'D':('99.95%','100.00%','35.05%','73.90%')}
    for pool in ('B','C','D'):
        for model in ('convnext_tiny','swin_t'):
            sub=frame[(frame.pool==pool)&(frame.model==model)]
            for value in (sub.top2_frequency.min(),sub.top2_frequency.max()):
                token=f'{100*value:.2f}%'
                assert token in text and token in bounds[pool]
    for row in attributes[attributes.aggregation=='target_weighted'].itertuples():
        assert f'[{100*row.f1_ci95_low:.2f}, {100*row.f1_ci95_high:.2f}]%' in text
    scale=pd.read_csv(OUT/'tables/scale_summary.csv')
    assert scale.targets.sum()==7800 and (scale.scale_changes==0).all() and (scale.clipping_scale_changes==0).all()
    audit=dict(status='pass',checked_at=now(),manuscript=str(PAPER),result_tables=17,
        numeric_table_cells_checked=previous.old.checked,new_numeric_cells=previous.old.checked-before,
        new_freeze_files=frozen_count,new_figures=figures,posthoc=True,all_state_accuracy_validated=False,
        broad_superiority_is_not_required=True)
    write_json(OUT/'manuscript_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
