#!/usr/bin/env python3
"""Audit the closeout and package portable, credential-free cached evidence."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from docx import Document

import build_journal_revision as base
from build_submission_closeout import OUT, OLD, EXP, EN, ZH, SI, DOCS, ROOT


def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect():
    dest=OUT/'portable';dest.mkdir(exist_ok=True)
    shutil.copytree(EXP/'portable_inputs',dest/'portable_inputs',dirs_exist_ok=True)
    (dest/'scripts').mkdir(exist_ok=True)
    for name in ['run_scale_only_confirmation.py','reaggregate_selection_evidence.py']:
        shutil.copy2(ROOT/'scripts'/name,dest/'scripts'/name)
    archive=dest/'historical_implementation';archive.mkdir(exist_ok=True)
    for name in ['freeze_independent_sixclass_selection.py','run_multiclass_object_selection_benchmark.py',
                 'analyze_independent_sixclass.py','run_voc2012_confirmation_benchmark.py',
                 'analyze_sixclass_difficulty.py','sixclass_car_support.py','dual_vlm_common.py',
                 'annotate_dual_vlm.py','annotate_multiclass_online_sensenova.py']:
        shutil.copy2(ROOT/'scripts'/name,archive/name)
    shutil.copy2(DOCS/'dataset_license_inventory_20261007.md',dest/'dataset_license_inventory.md')
    (dest/'protocol').mkdir(exist_ok=True)
    for p in DOCS.glob('*protocol*2026100*.md'):shutil.copy2(p,dest/'protocol'/p.name)
    source=base.BASES['B']
    for folder in ['configs','prompts']:
        shutil.copytree(source/folder,dest/folder,dirs_exist_ok=True)
    # The service dictionary appended to both prompt templates is an exact
    # source transformation. Requests add the image_id/target_instance_id below.
    from annotate_dual_vlm import format_reminder
    for domain in ('old','car'):
        schema=json.loads((source/'configs'/f'{domain}_annotation_schema.json').read_text())
        compact={'schema_version':schema['schema_version'],'category':schema['category'],
            'features':[{k:item.get(k,True) for k in ('feature_id','value_type','possible_values','allow_unknown')}
                for item in schema['universal_features']+schema.get('category_specific_features',[])]}
        prompt=(source/'prompts'/f'{domain}.txt').read_text()+'\n特征字典如下：\n'+json.dumps(compact,ensure_ascii=False)+format_reminder(schema)
        (dest/'prompts'/f'{domain}_assembled_request.txt').write_text(prompt+'\nimage_id: <sample_id>\ntarget_instance_id: target_1\n')
    private=Path('/home/hcf/project/QE/artifacts/feature_pilot_20260908/api_config.local.json')
    config=json.loads(private.read_text())
    allowed=['temperature','top_p','max_tokens','seed','presence_penalty','frequency_penalty']
    requested={k:v for k,v in config.get('request_parameters',{}).items() if k in allowed}
    requested['max_tokens']=max(3000,int(requested.get('max_tokens',0)))
    dump(dest/'annotation_call_settings.json',{
        'local':{'model':'qwen3-vl:30b-a3b-instruct','temperature':0,'num_ctx':8192,'num_predict':2600,
                 'seed':20260911,'timeout_seconds':900,'max_retries':3,'retry_wait_seconds':20},
        'online':{'model':config['model'],'request_parameters':requested,'response_format':{'type':'json_object'},
                  'timeout_seconds':config.get('timeout_seconds',600),'max_retries':5,'retry_wait_seconds':30,
                  'workers':4,'start_interval_seconds':3},
        'scope':'Source-code local options and current credential-free online config snapshot; provider weights and complete immutable historical request parameters are not established.',
        'images_order':['original','marked_A','enlarged_target_crop'],
        'target_category_information':'Category set/car protocol plus class-suffixed sample_id; not blind recognition.'})
    packages=['numpy','pandas','scikit-learn','scipy','matplotlib','Pillow','python-docx','torch','torchvision']
    versions={name:importlib.metadata.version(name) for name in packages}
    dump(dest/'environment.json',{'python':platform.python_version(),'packages':versions,
                                'scope':'Observed analysis/export environment on 2026-10-07; not a claim of an immutable original training container.'})
    (dest/'requirements_reanalysis.txt').write_text('\n'.join(f'{name}=={versions[name]}' for name in ['numpy','pandas','scikit-learn'])+'\n')
    for pool,p in base.BASES.items():
        h=dest/'historical_selection'/pool;h.mkdir(parents=True,exist_ok=True)
        for filename in ['equal_error_metrics.csv','comparisons.csv','fixed_metrics.csv','random_metrics.csv','model_aware_orders.csv']:
            shutil.copy2(p/'comparison/tables'/filename,h/filename)
        for q in (p/'selection').glob('*'):
            if q.suffix in ['.csv','.json','.npz']:shutil.copy2(q,h/q.name)
        for folder in ['confirmation','transfer','robustness']:
            d=dest/'historical_results'/pool/folder;d.mkdir(parents=True,exist_ok=True)
            for q in (p/folder/'tables').glob('*.csv'):shutil.copy2(q,d/q.name)
        a=dest/'source_audits'/pool;a.mkdir(parents=True,exist_ok=True)
        for name in ['exclusions.csv','forbidden_coco_ids.json','forbidden_voc_ids.json','historical_manifest_provenance.json']:
            q=p/'audit'/name
            if q.exists():shutil.copy2(q,a/q.name)
        for name in ['registration.json','selection_plan.json','final_audit.json','preparation_audit.json','pre_prediction_freeze.sha256.json','prediction_freeze.sha256.json']:
            q=p/name
            if q.exists():shutil.copy2(q,a/q.name)
    # Actual sampled mixture metrics, not plot interpolations or reconstructed
    # aggregate means. Retain the performance fields and class recalls.
    d=dest/'mixture_cache';d.mkdir(exist_ok=True)
    for p in (ROOT/'artifacts/controlled_risk_distribution_20261005/raw').glob('*_n120.npz'):
        with np.load(p) as z:
            keep=[z['metric_names'].tolist().index(k) for k in ['accuracy','macro_precision','macro_recall','macro_f1']]
            values={k:z[k] for k in ['class_recall','classes','proportions','models','seeds']}
            np.savez_compressed(d/p.name,metrics=z['metrics'][...,keep],
                metric_names=z['metric_names'][keep],**values)
    shutil.copytree(OLD/'evidence_tables',dest/'historical_evidence_tables',dirs_exist_ok=True)
    shutil.copytree(EXP/'tables',dest/'expected_scale_tables',dirs_exist_ok=True)
    shutil.copy2(EXP/'audit.json',dest/'scale_analysis_audit.json')
    for name in ['case_manifest.json','practical_breadth.csv']:
        shutil.copy2(OUT/name,dest/name)
    inputs=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file():inputs.append({'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(dest/'manifest.sha256.json',inputs)
    return dest


def audit():
    _,_,_,_,_,_,old_checked=base.evidence_and_checks((DOCS/'visual_state_testing_journal_en_20261006.md').read_text())
    df=pd.read_csv(EXP/'tables/summary.csv')
    cond=pd.read_csv(EXP/'tables/area_condition_summary.csv')
    texts={'en':EN.read_text(),'zh':ZH.read_text()}
    for language,text in texts.items():
        assert '<!--' not in text
        table_pattern=r'\*\*Table (\d+)\.' if language=='en' else r'\*\*表(\d+) '
        fig_pattern=r'!\[Figure (\d+)\.' if language=='en' else r'!\[图(\d+) '
        assert re.findall(table_pattern,text)==list(map(str,range(1,14)))
        assert re.findall(fig_pattern,text)==list(map(str,range(1,7)))
        for match in re.findall(r'!\[.*?\]\((.*?)\)',text):assert (DOCS/match).resolve().is_file()
        assert len(base.table_blocks(text))==13
        assert len(re.findall(r'^\$\$$',text,re.M))==6
        assert len(re.findall(r'^#{4,}',text,re.M))==0
    # All original numeric tables remain identical after their index shift.
    original=base.table_blocks((DOCS/'visual_state_testing_journal_en_20261006.md').read_text())
    current=base.table_blocks(texts['en'])
    assert current[:4]+current[6:]==original
    checks=0
    for row,r in zip(base.cells(current[4])[1:],df.itertuples()):
        for value,expected,places in [(row[2],100*r.full_gap,2),(row[3],100*r.area_gap,2),
                                      (row[4],100*r.increment_mean,2),(row[6],r.holm_six_p,5)]:
            base.close(value,expected,places);checks+=1
        assert row[5]==f'[{100*r.ci_low:+.2f}, {100*r.ci_high:+.2f}]';checks+=2
    for row,r in zip(base.cells(current[5])[1:],df.itertuples()):
        sub=cond[(cond.pool==r.pool)&(cond.source==r.source)];valid=sub[sub.evaluable]
        for value,expected in [(row[2],len(sub)),(row[3],len(valid)),(row[4],int((valid.area_adjusted_recall_gap<0).sum()))]:
            base.close(value,expected,0);checks+=1
        base.close(row[5],100*valid.area_adjusted_recall_gap.mean(),2);checks+=1
    # Bilingual table numeric sequences match without relying on language labels.
    for a,b in zip(base.table_blocks(texts['en']),base.table_blocks(texts['zh'])):
        numeric=lambda s:re.findall(r'(?<![A-Za-z])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?',s)
        assert numeric(a)==numeric(b)
    abstract=texts['en'].split('## Abstract\n')[1].split('**Keywords:')[0]
    words=len(re.findall(r"\b[\w]+(?:[’'\-][\w]+)*\b",abstract));assert 150<=words<=250,words
    references=texts['en'].split('## References\n')[1].strip().split('\n\n');assert len(references)==16
    svg_records=[]
    for p in sorted((OUT/'figures').glob('*.svg')):
        tags=[n.tag.split('}')[-1] for n in ET.parse(p).getroot().iter()]
        assert tags.count('text')>0
        if p.stem!='06_real_target_cases':assert tags.count('image')==0
        svg_records.append(dict(file=p.name,text=tags.count('text'),images=tags.count('image')))
    documents=[]
    for p in [EN,ZH]:
        d=Document(p.with_suffix('.docx'))
        assert len(d.tables)==13 and len(d.inline_shapes)==6
        xml=d._element.xml
        assert xml.count('<m:oMath>')==3
        documents.append(dict(file=p.with_suffix('.docx').name,tables=13,pictures=6,display_equations=3))
    before=json.loads((EXP/'portable_inputs/original_input_hashes.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in before.items())
    report=dict(status='pass',old_numeric_cells=old_checked,new_numeric_cells=checks,
        abstract_words=words,main_tables=13,main_figures=6,references=16,
        old_numeric_tables_unchanged=True,bilingual_numeric_tables_match=True,
        original_inputs_unchanged=len(before),word_documents=documents,svg_records=svg_records,
        no_new_training_or_inference_or_vlm=True,human_ground_truth_review=False,
        author_facts_pending=True,public_repository_pending=True,
        latex_compiled=False,word_full_page_rendering=False)
    replay_file=OUT/'portable_replay_audit.json'
    if replay_file.exists():
        replay=json.loads(replay_file.read_text())
        assert replay['status']=='pass'
        report['portable_extraction_replay_passed']=True
        report['portable_scale_tables_verified']=len(replay['all_scale_tables_match'])
        report['portable_selection_contrasts_verified']=replay['historical_selection_contrasts_checked']
    dump(OUT/'audit.json',report)
    return report


def packages(dest):
    for p in [EN,ZH,SI]:assert p.with_suffix('.docx').exists()
    full=DOCS/'visual_state_testing_submission_materials_20261007.zip'
    with ZipFile(full,'w',ZIP_DEFLATED) as z:
        extra_figures={}
        documents=[EN,ZH,SI,DOCS/'投稿收口与尺度对照结果_20261007.md']
        for name in ['优点_20261007.md','局限性_20261007.md']:
            p=DOCS/name
            if p.exists():documents.append(p)
        for p in documents:
            # Markdown should resolve figures after extraction, independently
            # of the original local artifacts directory.
            def image_path(match):
                original=(p.parent/match[2]).resolve()
                assert original.is_file()
                extra_figures['figures/'+original.name]=original
                for ext in ['.svg','.pdf','.eps']:
                    sibling=original.with_suffix(ext)
                    if sibling.exists():extra_figures['figures/'+sibling.name]=sibling
                return '!['+match[1]+'](../figures/'+original.name+')'
            packaged=re.sub(r'!\[(.*?)\]\((.*?)\)',image_path,p.read_text())
            z.writestr('docs/'+p.name,packaged)
            z.write(p.with_suffix('.docx'),'docs/'+p.with_suffix('.docx').name)
        for folder in ['figures','latex','case_inputs']:
            for p in (OUT/folder).iterdir():z.write(p,folder+'/'+p.name)
        existing=set(z.namelist())
        for name,p in extra_figures.items():
            if name not in existing:z.write(p,name)
        for p in OUT.glob('*.md'):z.write(p,'review/'+p.name)
        for name in ['audit.json','case_manifest.json','practical_breadth.csv','portable_replay_audit.json','report_sync_audit.json']:
            p=OUT/name
            if p.exists():z.write(p,'review/'+name)
        for p in dest.rglob('*'):
            if p.is_file():z.write(p,'reproducibility/'+str(p.relative_to(dest)))
        for name in ['build_submission_closeout.py','package_submission_closeout.py','audit_portable_submission.py','sync_latest_reports.py']:
            z.write(ROOT/'scripts'/name,'source/'+name)
    portable=DOCS/'visual_state_testing_portable_reanalysis_20261007.zip'
    with ZipFile(portable,'w',ZIP_DEFLATED) as z:
        for p in dest.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(dest)))
    latex=DOCS/'visual_state_testing_sqj_latex_20261007.zip'
    with ZipFile(latex,'w',ZIP_DEFLATED) as z:
        for p in (OUT/'latex').iterdir():z.write(p,p.name)
    for p in [full,portable,latex]:
        with ZipFile(p) as z:assert z.testzip() is None
    print(json.dumps({'full':str(full),'portable':str(portable),'bytes':full.stat().st_size},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');args=p.parse_args()
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
    if not args.audit_only:
        dest=collect()
        shutil.copy2(DOCS/'复现入口说明_20261007.md',dest/'README.md')
        # README is added after collection; keep the manifest complete.
        entries=[{'path':str(q.relative_to(dest)),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='manifest.sha256.json']
        dump(dest/'manifest.sha256.json',entries)
        base.word(DOCS/'投稿收口与尺度对照结果_20261007.md',False)
        packages(dest)
