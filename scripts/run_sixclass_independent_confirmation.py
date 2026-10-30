#!/usr/bin/env python3
"""Run/resume all four priority experiments in one logged background job."""
from __future__ import annotations
import argparse
import fcntl
import itertools
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from PIL import Image
from independent_confirmation_common import *
from run_dog_local_pipeline import validate_feature

def child(stage,arguments):
    log=BASE/'logs'/f'{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
    stream=log.open('a',encoding='utf-8')
    p=subprocess.Popen([str(PYTHON),'-u',*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                       env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','PYTHONUNBUFFERED':'1'})
    write_json(BASE/'processes'/f'{stage}.json',dict(pid=p.pid,arguments=arguments,log=str(log),started_at=now()))
    return p,stream

def run_stage(stage,arguments):
    marker=BASE/'stages'/f'{stage}.json'
    if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':return
    p,stream=child(stage,arguments)
    while p.poll() is None:
        note(stage,child_pid=p.pid,log=f'logs/{stage}.log');time.sleep(20)
    stream.close()
    if p.returncode:raise RuntimeError(f'{stage} exit {p.returncode}; see logs/{stage}.log')
    write_json(marker,dict(status='complete',finished_at=now(),child_pid=p.pid))

def inputs(rows):
    for domain in ('old','car'):
        output=[]
        for row in rows:
            if (row['class_name']=='car')!=(domain=='car'):continue
            with Image.open(row['image_path']) as image:
                from prepare_multiclass_object_candidate_pool import absolute_box
                box=absolute_box(json.loads(row['bbox_norm_xyxy']),image.width,image.height,0)
            output.append(dict(image_id=row['sample_id'],target_instance_id='target_1',target_class=row['class_name'],
                               target_class_id=row['target_class_id'],bbox=json.dumps(box),image_path=row['image_path'],
                               marked_image_path=row['marked_image_path'],crop_image_path=row['feature_crop_path'],
                               protocol_version='car_shared_state_v1.0' if domain=='car' else 'object_state_v1.0'))
        write_csv(BASE/'inputs'/f'{domain}.csv',output)

def missing(rows):
    failures=[]
    for row in rows:
        domain='car' if row['class_name']=='car' else 'old'
        schema=json.loads((BASE/'configs'/('car_annotation_schema.json' if domain=='car' else 'old_annotation_schema.json')).read_text())
        sample=dict(image_id=row['sample_id'],target_instance_id='target_1')
        for backend in ('local','online'):
            path=BASE/'annotations'/backend/domain/'parsed'/f"{row['sample_id']}.json"
            try:validate_feature(json.loads(path.read_text()),sample,schema)
            except (OSError,ValueError,TypeError,json.JSONDecodeError):failures.append(dict(sample_id=row['sample_id'],class_name=row['class_name'],backend=backend))
    return failures

def annotation_round(label):
    processes=[]
    for backend in ('local','online'):
        p,stream=child(label+'_'+backend,['scripts/annotate_independent_sixclass_pool.py','--backend',backend])
        processes.append((backend,p,stream))
    while any(p.poll() is None for _,p,_ in processes):
        counts={}
        for backend,p,_ in processes:
            counts[backend]=sum(len(list((BASE/'annotations'/backend/domain/'parsed').glob('*.json'))) for domain in ('old','car'))
        note('annotate',round=label,completed_strict_files=counts,children={b:p.pid for b,p,_ in processes});time.sleep(20)
    for backend,p,stream in processes:
        stream.close()
        if p.returncode:raise RuntimeError(f'{label}/{backend} exited {p.returncode}; see logs')

def annotations():
    marker=BASE/'stages/annotate.json'
    if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':return
    rows=read_csv(BASE/'manifests/candidate_pool.csv');inputs(rows)
    for backend in ('online','local'):
        run_stage('smoke_'+backend,['scripts/annotate_independent_sixclass_pool.py','--backend',backend,'--smoke'])
    annotation_round('initial')
    if missing(rows):annotation_round('bounded_retry')
    reserves=read_csv(BASE/'manifests/technical_reserves.csv');replacements=[]
    used={r['sample_id'] for r in rows}
    for attempt in range(1,61):
        errors=missing(rows)
        if not errors:break
        failed={r['sample_id'] for r in errors}
        changes=[]
        from run_multiclass_object_selection_benchmark import scale_state
        for i,row in enumerate(rows):
            if row['sample_id'] not in failed:continue
            options=[r for r in reserves if r['class_name']==row['class_name'] and r['sample_id'] not in used]
            same=[r for r in options if scale_state(float(r['bbox_area_ratio']))==scale_state(float(row['bbox_area_ratio']))]
            if not options:raise ValueError(f"Reserves exhausted for {row['class_name']}")
            chosen=min(same or options,key=lambda r:int(r['selection_rank']));used.add(chosen['sample_id'])
            chosen={**chosen,'role':'candidate','replaces':row['sample_id']}
            mapping=dict(old_sample_id=row['sample_id'],new_sample_id=chosen['sample_id'],class_name=row['class_name'],
                         same_scale=bool(same),reason='strict_annotation_unavailable_after_bounded_retry',round=attempt)
            replacements.append(mapping);changes.append(chosen);rows[i]=chosen
        write_csv(BASE/'manifests/replacement_mapping.csv',replacements)
        write_json(BASE/f'audit/annotation_replacements_round{attempt}.json',dict(failures=errors,replacements=replacements))
        # Annotate only newly chosen targets; preserve old raw and parsed evidence.
        inputs(changes);annotation_round('replacement_'+str(attempt));inputs(rows)
        write_csv(BASE/'manifests/candidate_pool.csv',rows)
    failures=missing(rows)
    if failures:raise ValueError(f'Paired annotation incomplete: {len(failures)}')
    write_json(BASE/'annotation_audit.json',dict(status='pass',targets=2400,strict_per_backend=2400,
               replacements=len(replacements),manual_review=False,original_pool_unchanged=not replacements))
    write_json(marker,dict(status='complete',finished_at=now()))

def registration():
    path=BASE/'registration.json'
    if path.exists():return
    write_json(path,dict(registered_at=now(),protocol_sha256=sha256(PROTOCOL),classes=list(CLASSES),models=list(MODELS),seeds=list(SEEDS),
                        targets_per_class=400,training_frozen=True,stages=list(STAGES),primary_confirmation_family=6,primary_selection_family=4,
                        strong_baseline_family=12,global_selection_sensitivity_family=20,random_replicates=1000,
                        independent_samples_not_dataset_source=True))
    files=[PROTOCOL,*[ROOT/'scripts'/name for name in ('run_sixclass_independent_confirmation.py','prepare_independent_sixclass_pool.py',
           'independent_confirmation_common.py','annotate_independent_sixclass_pool.py','freeze_independent_sixclass_selection.py','analyze_independent_sixclass.py')]]
    files+=[ROOT/'scripts'/name for name in ('annotate_dual_vlm.py','annotate_multiclass_online_sensenova.py','analyze_sixclass_difficulty.py',
            'run_dog_local_pipeline.py','analyze_dual_vlm_study.py','run_class_stratified_error_controls.py','run_coverage_incremental_value.py')]
    freeze(BASE/'execution_code_freeze.sha256.json',files)

def research_records():
    write_json(BASE/'process.json',dict(pid=os.getpid(),started_at=now(),log=str(BASE/'pipeline.log')))
    (BASE/'paper_brief.md').write_text('# 独立确认研究范围\n\n冻结原分类器，确认新样本上的风险响应、相同风险预算下的诊断收益及A到B状态迁移。原方法论文的范围不扩大为通用算法或因果证明。\n')
    (BASE/'claim_evidence.md').write_text('# 本轮待检验结论\n\n独立风险响应→confirmation/；同风险额外诊断→comparison/；薄弱状态跨池稳定性→transfer/；风险/来源/簇依赖→robustness/。执行完成前均为待检验，不写成功。\n')
    with (BASE/'research_log.md').open('a') as f:f.write(f'\n[{now()}] 用户授权按四项优先级启动完整流水线；原权重不训练，阶段日志在logs/。入口PID={os.getpid()}。\n')
    (BASE/'review.md').write_text('# 当前审阅\n\n执行中，不能声称完整性门禁通过或结果支持。旧研究引用沿用；结果待各阶段完成再核对。\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-pid',type=int,default=0);args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registration();research_records()
    try:
        if args.prepare_pid and not (BASE/'preparation_audit.json').exists():
            while not (BASE/'preparation_audit.json').exists():
                try:os.kill(args.prepare_pid,0)
                except ProcessLookupError:raise RuntimeError('Preparation failed; see prepare.log')
                note('prepare',child_pid=args.prepare_pid,log='prepare.log');time.sleep(20)
        if not (BASE/'preparation_audit.json').exists():run_stage('prepare',['scripts/prepare_independent_sixclass_pool.py'])
        verify(BASE/'preparation_freeze.sha256.json')
        annotations()
        run_stage('freeze_selection',['scripts/freeze_independent_sixclass_selection.py'])
        for stage in ('infer','confirm','compare','transfer','robustness','report'):
            verify(BASE/'execution_code_freeze.sha256.json')
            run_stage(stage,['scripts/analyze_independent_sixclass.py','--stage',stage])
        note('complete')
        write_json(BASE/'progress.json',dict(status='complete',stage='complete',heartbeat=now(),pid=os.getpid()))
        (BASE/'review.md').write_text('# 执行审阅\n\n全部优先级阶段完成。计数、配额、标签和源输入冻结见各阶段audit及final_audit.json。研究结论必须依据report_zh.md中全部正负结果；新池同属COCO，不宣称新数据来源确认或人工标签真值。\n')
        with (BASE/'research_log.md').open('a') as f:f.write(f'\n[{now()}] 全部阶段完成；原训练/权重和冻结输入无变化。结果见report_zh.md。\n')
    except Exception as exc:
        write_json(BASE/'progress.json',dict(status='failed',error=str(exc),heartbeat=now(),pid=os.getpid()))
        traceback.print_exc();raise

if __name__=='__main__':main()
