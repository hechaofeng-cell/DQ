#!/usr/bin/env python3
"""Logged, resumable no-human evidence-strengthening execution."""
import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from automatic_strengthening_common import *

STAGES = ('automatic_audit_existing', 'metadata', 'prepare', 'annotate_D', 'freeze_selection', 'infer',
          'confirm', 'compare', 'transfer', 'robustness', 'report_D', 'annotate_E', 'select_E', 'infer_E',
          'evaluate_E', 'automatic_audit_all', 'statistics', 'final_report')


def records():
    BASE.mkdir(parents=True, exist_ok=True)
    if not (BASE / 'registration.json').exists():
        weak = read_csv(B / 'transfer/weak_states_A_frozen.csv')
        by_condition = {}
        for row in weak:
            key = row['class_name'], row['feature'], row['value']
            by_condition.setdefault(key, []).append(row)
        selected = []
        for name in CLASSES:
            ranked = sorted((key for key in by_condition if key[0]==name), key=lambda k:(-len(by_condition[k]), k[1], k[2]))
            for key in ranked[:2]:
                models = {s:sorted({r['model'] for r in by_condition[key] if r['source']==s}) for s in ('local','online')}
                selected.append(dict(class_name=name, feature=key[1], value=key[2], frozen_models=models,
                                     old_combinations=len(by_condition[key]), primary_membership_source='local'))
        write_json(BASE / 'frozen_E_conditions.json', selected)
        value = dict(registered_at=now(), protocol_sha256=sha256(PROTOCOL), classes=list(CLASSES), models=list(MODELS), seeds=list(SEEDS),
            candidate_source='Open Images official train boxes', mids=MID, targets_per_class=400, technical_reserves_max_per_class=60,
            E_reservoir_max_per_class=300, E_case_target=30, E_reference_target=30, E_minimum_support_and_pairs=20,
            E_conditions_sha256=sha256(BASE / 'frozen_E_conditions.json'), state_models_A_sha256=sha256(B / 'transfer/weak_states_A_frozen.csv'),
            training_frozen=True, manual_review=False, annotation_rounds_max=2, replacements='same class, same scale preferred, hash rank; technical failure only',
            budgets_per_class=[20,40,60,80,100], random_replicates=1000, main_test_families={'risk':6,'random':4,'strong':12,'all_sensitivity':20},
            glmm=dict(implementation='statsmodels BinomialBayesMixedGLM.fit_vb', random_effects=['run_id','image_id'],
                      fixed_effects='risk_tier * dataset + class + architecture', fe_p=2,vcp_p=1,
                      optimizer='BFGS',maxiter=300,seed=20261005,interval='approximate variational posterior credible interval; not a frequentist p-value',
                      convergence_failure='report unconfirmed fit; do not switch model to seek significance'),
            diagnosis_bootstrap='legacy nonlinear rarefaction image bootstrap omitted; original architecture tests retained',
            data_shortage='stop D before new classifier predictions if <400 eligible per class; E shortage is unevaluable',
            independent_source_scope='OI distinct from COCO training, previously used in A discovery', stages=list(STAGES))
        write_json(BASE / 'registration.json', value)
    for name,text in {
        'paper_brief.md':'# 研究范围\n\n固定模型、多类别自动视觉状态测试；补强字段参考一致性、条件诊断复现与测试交付，不以模型修复或全面方法优势为目标。无新增人工复核。\n',
        'claim_evidence.md':'# 待确认的证据\n\n自动核对→automatic_audit/；新样本风险与选样→pool_D/；预选条件→suite_E/；补充模型→statistics/。完成前均不得写作支持。双VLM一致率不是标签准确率。\n',
        'review.md':'# 完整性门禁\n\n执行中。新结果尚未完成核验，不声明论文已补强完成。新文献/引用待对应统计实现和格式核对；保留全部既有负结果。\n',
    }.items():
        if not (BASE/name).exists(): (BASE/name).write_text(text)
    with (BASE/'research_log.md').open('a') as stream:
        stream.write(f'\n[{now()}] 启动/恢复无人工复核补强，PID={os.getpid()}。D=六类各400；E候选最多1800；24权重冻结。分阶段日志在logs/。\n')
    write_json(BASE/'process.json',dict(pid=os.getpid(),started_at=now(),log=str(BASE/'pipeline.log'),command=sys.argv))


def execute(stage, arguments, optional=False):
    marker = BASE/'stages'/f'{stage}.json'
    if marker.exists() and json.loads(marker.read_text()).get('status') in ('complete','unavailable'):
        return
    log=BASE/'logs'/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        process=subprocess.Popen([str(PYTHON),'-u',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'})
        write_json(BASE/'processes'/f'{stage}.json',dict(pid=process.pid,command=arguments,log=str(log),started_at=now()))
        while process.poll() is None:
            note(stage,child_pid=process.pid,log=str(log));time.sleep(20)
    if process.returncode:
        write_json(marker,dict(status='unavailable' if optional else 'failed',returncode=process.returncode,finished_at=now(),log=str(log)))
        if optional:
            note(stage,result='unavailable',log=str(log));return
        raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
    write_json(marker,dict(status='complete',finished_at=now(),log=str(log)))


def annotate(destination, rows):
    copy_setup(destination); annotation_inputs(rows,destination)
    for round_number in (1,2):
        labels, failures = labels_for(rows,destination)
        if not failures: return labels
        processes=[]
        for backend in ('local','online'):
            log=BASE/'logs'/f'annotate_{destination.name}_{backend}_round{round_number}.log';log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('a')
            p=subprocess.Popen([str(PYTHON),'-u',str(Path(__file__)), '--action','annotation_branch','--destination',str(destination),
                                '--backend',backend],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
            processes.append((backend,p,stream));write_json(BASE/'processes'/f'{destination.name}_{backend}.json',dict(pid=p.pid,log=str(log)))
        while any(p.poll() is None for _,p,_ in processes):
            counts={b:sum(len(list((destination/'annotations'/b/domain/'parsed').glob('*.json'))) for domain in ('old','car')) for b,_,_ in processes}
            note('annotate_'+destination.name,round=round_number,planned=len(rows),parsed_counts=counts);time.sleep(20)
        for b,p,stream in processes:
            stream.close()
            if p.returncode: raise RuntimeError(f'{destination.name}/{b}: failed annotation process')
    return labels_for(rows,destination)[0]


def annotate_d():
    rows=read_csv(D/'manifests/candidate_pool.csv');annotate(D,rows)
    used={r['sample_id'] for r in rows};reserves=read_csv(D/'manifests/technical_reserves.csv');replacements=[]
    from run_multiclass_object_selection_benchmark import scale_state
    for _ in range(60):
        _,failures=labels_for(rows,D)
        if not failures:break
        failed={r['sample_id'] for r in failures};changed=[]
        for i,row in enumerate(rows):
            if row['sample_id'] not in failed:continue
            options=[r for r in reserves if r['class_name']==row['class_name'] and r['sample_id'] not in used]
            same=[r for r in options if scale_state(float(r['bbox_area_ratio']))==scale_state(float(row['bbox_area_ratio']))]
            if not options:raise ValueError(f"Technical reserves exhausted for {row['class_name']}")
            replacement={**min(same or options,key=lambda r:int(r['selection_rank'])),'role':'candidate','replaces':row['sample_id']}
            used.add(replacement['sample_id']);changed.append(replacement);rows[i]=replacement
            replacements.append(dict(old_sample_id=row['sample_id'],new_sample_id=replacement['sample_id'],class_name=row['class_name'],
                                     reason='strict_annotation_unavailable_after_two_rounds',same_scale=bool(same)))
        write_csv(D/'manifests/replacement_mapping.csv',replacements)
        write_csv(D/'manifests/candidate_pool.csv',rows)
        annotate(D,changed)
    annotation_inputs(rows,D)
    _,failures=labels_for(rows,D)
    if failures:raise ValueError(f'Paired D annotations incomplete: {len(failures)}')
    assert Counter(r['class_name'] for r in rows)==Counter(dict.fromkeys(CLASSES,400))
    write_json(D/'annotation_audit.json',dict(status='pass',targets=len(rows),strict_per_backend=len(rows),replacements=len(replacements),manual_review=False))


def annotation_branch(destination, backend):
    import annotate_dual_vlm as adapter
    adapter.STUDY=destination
    adapter.job_path=lambda b,domain:destination/'annotations'/b/domain
    adapter.schema_path=lambda domain:destination/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')
    adapter.prompt_path=lambda domain:destination/'prompts'/f'{domain}.txt'
    for domain in ('old','car'):
        if read_csv(destination/'inputs'/f'{domain}.csv'):
            (adapter.run_local if backend=='local' else adapter.run_online)(domain,0,False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',default='pipeline',choices=('pipeline','annotation_branch','annotate_D','annotate_E'))
    parser.add_argument('--destination',type=Path);parser.add_argument('--backend',choices=('local','online'))
    args=parser.parse_args()
    if args.action=='annotation_branch':annotation_branch(args.destination,args.backend);return
    if args.action=='annotate_D':annotate_d();return
    if args.action=='annotate_E':
        rows=read_csv(E/'manifests/reservoir.csv')
        if rows:annotate(E,rows)
        _, failures=labels_for(rows,E);write_json(E/'annotation_audit.json',dict(status='complete_with_missing' if failures else 'pass',targets=len(rows),failures=failures))
        return
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records()
    try:
        execute('automatic_audit_existing',['scripts/audit_automatic_states.py'])
        execute('metadata',['scripts/prepare_automatic_strengthening.py','--stage','metadata'])
        availability=json.loads((BASE/'audit/availability.json').read_text())
        if availability.get('status')!='pass':
            write_json(BASE/'data_expansion_blocked.json',dict(status='blocked',audit=availability,
                reason='Prespecified unused Open Images quota is not available; no source substitution or old-image reuse',
                independent_work=['official attribute audit','cached B/C statistical supplements','regression test kit','figures and report']))
            from independent_confirmation_common import freeze,verify
            paths=[PROTOCOL,BASE/'registration.json',BASE/'frozen_E_conditions.json',BASE/'audit/availability.json']
            paths+=[ROOT/'scripts'/n for n in ('automatic_strengthening_common.py','run_automatic_strengthening.py','audit_automatic_states.py',
                'fit_automatic_statistics.py','analyze_automatic_strengthening.py','replay_visual_regression_suite.py')]
            paths+=[ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/m/'oi_gap_add0'/str(s)/'best.pt' for m,s in itertools.product(MODELS,SEEDS)]
            freeze(BASE/'existing_evidence_execution_freeze.sha256.json',paths)
            execute('statistics',['scripts/fit_automatic_statistics.py'],optional=True)
            verify(BASE/'existing_evidence_execution_freeze.sha256.json')
            execute('final_report',['scripts/analyze_automatic_strengthening.py','--stage','final_report'])
            execute('regression_replay',['scripts/replay_visual_regression_suite.py'])
            write_json(BASE/'progress.json',dict(status='partial_complete',stage='complete_existing_evidence',
                pending=['new dataset-source decision','D and E new-sample experiments'],pid=os.getpid(),heartbeat=now()))
            return
        # All implementation files are complete and frozen before expensive annotations/predictions.
        from independent_confirmation_common import freeze,verify
        paths=[PROTOCOL,BASE/'registration.json',BASE/'frozen_E_conditions.json']
        paths+=[ROOT/'scripts'/name for name in ('automatic_strengthening_common.py','prepare_automatic_strengthening.py',
            'run_automatic_strengthening.py','audit_automatic_states.py','analyze_automatic_strengthening.py','evaluate_automatic_conditions.py','fit_automatic_statistics.py')]
        paths+=[ROOT/'scripts'/name for name in ('analyze_independent_sixclass.py','freeze_independent_sixclass_selection.py',
            'annotate_dual_vlm.py','annotate_multiclass_online_sensenova.py','analyze_sixclass_difficulty.py','run_dog_local_pipeline.py')]
        paths+=[ROOT/'artifacts/sixclass_car_extension_20261004/training/runs'/m/'oi_gap_add0'/str(s)/'best.pt' for m,s in itertools.product(MODELS,SEEDS)]
        freeze(BASE/'execution_freeze.sha256.json',paths)
        execute('prepare',['scripts/prepare_automatic_strengthening.py'])
        execute('annotate_D',[str(Path(__file__)),'--action','annotate_D'])
        for stage in ('freeze_selection','infer','confirm','compare','transfer','robustness','report_D'):
            verify(BASE/'execution_freeze.sha256.json')
            execute(stage,['scripts/analyze_automatic_strengthening.py','--stage',stage])
        execute('annotate_E',[str(Path(__file__)),'--action','annotate_E'])
        for stage in ('select_E','infer_E','evaluate_E'):
            verify(BASE/'execution_freeze.sha256.json')
            execute(stage,['scripts/evaluate_automatic_conditions.py','--stage',stage])
        execute('automatic_audit_all',['scripts/audit_automatic_states.py','--include-D'])
        execute('statistics',['scripts/fit_automatic_statistics.py'],optional=True)
        verify(BASE/'execution_freeze.sha256.json')
        execute('final_report',['scripts/analyze_automatic_strengthening.py','--stage','final_report'])
        write_json(BASE/'progress.json',dict(status='complete',stage='complete',pid=os.getpid(),heartbeat=now()))
    except Exception as error:
        write_json(BASE/'progress.json',dict(status='failed',stage='pipeline',error=str(error),pid=os.getpid(),heartbeat=now()))
        traceback.print_exc();raise


if __name__=='__main__':main()
