#!/usr/bin/env python3
"""Audit new and retained manuscript evidence without writing to earlier artifacts."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
import audit_rank_validated_visual_state_paper as retained

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/visual_state_testing_empirical_paper_updated_20261006.md'
OLD=ROOT/'docs/visual_state_testing_empirical_paper_rank_validated_20261005.md'
BASE=ROOT/'artifacts/controlled_risk_distribution_20261005'
OUT=ROOT/'artifacts/visual_state_paper_updated_20261006'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Reuse established raw-result checks; redirect every audit write to this version.
    # Retained OUT stays unchanged because it is also the source-data directory.
    def redirect(path,value):
        name='retained_rank_audit.json' if Path(path).name=='manuscript_audit.json' else Path(path).name
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    preserved=[OLD,OLD.with_suffix('.docx')]
    preserved.extend((retained.OUT/name) for name in ('manuscript_audit.json','retained_results_audit.json','word_export_audit.json'))
    preserved=[p for p in preserved if p.exists()]
    hashes={str(p):digest(p) for p in preserved}
    retained.PAPER=PAPER
    retained.write_json=redirect
    retained.previous.write_json=redirect
    retained.main()
    text=PAPER.read_text();old=OLD.read_text()
    table=retained.previous.table;check=retained.previous.old.check
    before=retained.previous.old.checked
    for number in range(1,18):assert table(text,number)==table(old,number)
    assert text.split('## 2 相关工作')[1].split('## 3 方法')[0]==old.split('## 2 相关工作')[1].split('## 3 方法')[0]
    assert text.split('## 参考文献')[1].split('## 附录A')[0]==old.split('## 参考文献')[1].split('## 附录A')[0]
    panel=pd.read_csv(BASE/'tables/panel_summary.csv')
    contrasts=pd.read_csv(BASE/'tables/distribution_contrasts.csv')
    ranks=pd.read_csv(BASE/'tables/model_summary.csv')
    classes=pd.read_csv(BASE/'tables/class_panel_contrasts.csv')
    sources={'本地':'local','线上':'online'}
    for row in table(text,18):
        pool,source=row[0],sources[row[1]]
        p=panel[(panel.pool==pool)&(panel.source==source)&(panel.total==120)].set_index('high_risk_percent')
        d=contrasts[(contrasts.pool==pool)&(contrasts.source==source)&(contrasts.total==120)&(contrasts.model=='panel_mean')].iloc[0]
        for value,want in zip(row[2:5],[p.loc[0].macro_f1_mean,p.loc[100].macro_f1_mean,d.macro_f1_delta]):check(value,100*want)
        values=re.findall(r'[−+-]?\d+\.\d+',row[5]);assert len(values)==2
        for value,want in zip(values,[d.macro_f1_sampling_low,d.macro_f1_sampling_high]):check(value,100*want)
    for row in table(text,19):
        g=ranks[(ranks.pool==row[0])&(ranks.source==sources[row[1]])&(ranks.total==120)]
        for ratio in range(0,101,5):
            assert g[g.high_risk_percent==ratio].sort_values('rank').model.iloc[:2].tolist()==['convnext_tiny','swin_t']
        assert row[2:4]==['ConvNeXt-Tiny','ConvNeXt-Tiny']
        span=g.pivot(index='model',columns='high_risk_percent',values='rank')
        check(row[4],g.point_kendall_vs20.min(),3)
        check(row[5],(span.max(axis=1)-span.min(axis=1)).max(),0)
    for row in table(text,20):
        p=panel[(panel.pool==row[0])&(panel.source==sources[row[1]])&(panel.total==120)].set_index('high_risk_percent')
        for value,want in zip(row[2:],[p.loc[0].failed_state_breadth_mean,p.loc[100].failed_state_breadth_mean,
                                      p.loc[100].equal_error_confusion_delta_mean,p.loc[100].equal_error_cluster_delta_mean]):check(value,100*want)
    for row in table(text,21):
        c=contrasts[(contrasts.pool==row[0])&(contrasts.source==sources[row[1]])&(contrasts.model=='panel_mean')].set_index('total')
        check(row[2],100*c.loc[120].macro_f1_delta);check(row[3],100*c.loc[480].macro_f1_delta)
        values=re.findall(r'[−+-]?\d+\.\d+',row[4]);assert len(values)==2
        for value,want in zip(values,[c.loc[480].macro_f1_sampling_low,c.loc[480].macro_f1_sampling_high]):check(value,100*want)
    common=contrasts[(contrasts.total==120)&(contrasts.model=='panel_mean')]
    assert len(common)==6 and (common.macro_f1_sampling_high<0).all()
    assert f'{-100*common.macro_f1_delta.max():.2f}—{-100*common.macro_f1_delta.min():.2f}' in text
    individual=contrasts[(contrasts.total==120)&(contrasts.model!='panel_mean')]
    assert len(individual)==48 and (individual.macro_f1_delta<0).all()
    for pool,source,name in [('B','local','car'),('C','online','cat')]:
        r=classes[(classes.pool==pool)&(classes.source==source)&(classes.total==120)&(classes.class_name==name)].iloc[0]
        values=[r.recall_delta,r.recall_sampling_high]
        if name=='car':values.extend([r.recall_at0,r.recall_at100])
        for value in values:
            assert f'{100*value:.2f}' in text
        assert f'{100*r.recall_sampling_low:.2f}'.replace('-','−') in text
        assert r.recall_sampling_low<0<r.recall_sampling_high
    assert [int(x) for x in re.findall(r'^\*\*表(\d+) ',text,re.M)]==list(range(1,22))
    assert [int(x) for x in re.findall(r'^!\[图(\d+) ',text,re.M)]==list(range(1,16))
    assert text.count('$$')==6
    figure_paths=re.findall(r'^!\[[^]]+\]\(([^)]+)\)',text,re.M)
    assert len(figure_paths)==15 and all(Path(p).is_file() for p in figure_paths)
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if link.startswith(('http://','https://')):continue
        p=Path(link) if link.startswith('/') else PAPER.parent/link
        assert p.is_file(),str(p)
    for svg in (BASE/'figures').glob('*.svg'):
        e=ET.parse(svg).getroot();ns={'s':'http://www.w3.org/2000/svg'}
        assert e.findall('.//s:text',ns) and not e.findall('.//s:image',ns)
    assert json.loads((BASE/'final_audit.json').read_text())['status']=='pass'
    freeze=json.loads((BASE/'input_freeze.sha256.json').read_text())
    assert all(digest(Path(path))==want for path,want in freeze.items())
    with ZipFile(PAPER.with_suffix('.docx')) as z:
        e=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
            'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        counts=dict(tables=len(e.findall('.//w:tbl',ns)),embedded_figures=len(e.findall('.//a:blip',ns)),
                    editable_display_equations=len(e.findall('.//m:oMath',ns)))
        expected=len(re.findall(r'^\|[^\n]+\n\|[ :|\-]+\|\s*$',text,re.M))
        assert counts==dict(tables=expected,embedded_figures=15,editable_display_equations=3)
        cells=[n.text or '' for n in e.findall('.//w:t',ns)]
        assert any('固定预算风险混合' in x for x in cells)
        assert not any('$$' in x for x in cells)
    assert all(digest(Path(p))==want for p,want in hashes.items()),'Earlier artifact modified'
    audit=dict(status='pass',manuscript=str(PAPER),word=str(PAPER.with_suffix('.docx')),
        result_tables=21,numeric_table_cells_verified=retained.previous.old.checked,
        new_numeric_cells=retained.previous.old.checked-before,
        retained_result_tables_unchanged=17,retained_related_work_and_references_unchanged=True,
        risk_input_hashes_unchanged=len(freeze),prior_artifacts_unchanged=hashes,
        new_experiments=False,**counts,
        new_interval_scope='repeated sampling of fixed finite pools; not a mean confidence interval',
        rendering='Word structure audited; page layout not rendered')
    (OUT/'manuscript_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
