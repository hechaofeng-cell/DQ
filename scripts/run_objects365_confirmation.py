#!/usr/bin/env python3
"""Priority-only external-source confirmation, logged and resumable in background."""
import argparse
import fcntl
import itertools
import json
import os
import sys
import time
import traceback
from objects365_confirmation_common import *


def annotation_branch(backend,smoke):
    # Reuse audited transport accounting without executing or modifying the VOC study.
    import run_sixclass_voc_external_confirmation as cost_adapter
    cost_adapter.BASE=BASE
    cost_adapter.annotate(backend,smoke)


def registration():
    from independent_confirmation_common import freeze
    path=BASE/'registration.json'
    if not path.exists():
        write_json(path,dict(registered_at=now(),protocol_sha256=sha256(PROTOCOL),source='Objects365 v2 train package prefix',
            classes=list(CLASSES),models=list(MODELS),seeds=list(SEEDS),targets_per_class=400,technical_reserves_max_per_class=60,
            budgets_per_class=[20,40,60,80,100],random_replicates=1000,manual_review=False,training_frozen=True,
            primary_risk_family=6,primary_diagnostic_family=4,transfer_family=2,strong_baseline_family=12,
            global_selection_sensitivity_family=20,stages=list(STAGES),state_support_minimum=20,
            source_amendment='validation metadata sheep=227; before new classifier prediction changed to train package prefix',
            package_rule='SHA256 order first >=600/class prefix <=80GiB; no result dependent package selection',
            no_glmm_repair=True,no_new_training=True))
    paths=code_paths()+[path,ROOT/'scripts/run_sixclass_independent_confirmation.py',ROOT/'scripts/run_sixclass_voc_external_confirmation.py']
    paths += [WEIGHTS/m/'oi_gap_add0'/str(s)/'best.pt' for m,s in itertools.product(MODELS,SEEDS)]
    freeze(BASE/'execution_code_freeze.sha256.json',paths)
    for filename,content in {
        'paper_brief.md':'# 确认目标\n\n仅补独立数据集来源的风险层关联与冻结弱状态复现；六类、24份权重、双VLM不变，无人工复核。\n',
        'claim_evidence.md':'# 待确认\n\n风险响应→confirmation；冻结状态复现→transfer；方法比较→comparison。均待执行，来源数量通过不代表结果支持。\n',
        'review.md':'# 执行中\n\n元数据、数据独立性、双标签及结果未完成前不声明确认成功。\n',
    }.items():
        if not (BASE/filename).exists():(BASE/filename).write_text(content)
    with (BASE/'research_log.md').open('a') as stream:
        stream.write(f'\n[{now()}] 本轮只补新来源确认，PID={os.getpid()}；初始val sheep不足，元数据驱动修订为train分包前缀。旧数据、权重及失败统计不修改。\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('local','online'));parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--source-pid',type=int,default=0)
    args=parser.parse_args()
    if args.backend:
        annotation_branch(args.backend,args.smoke);return
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    common,_,_,_=activate()
    import run_sixclass_independent_confirmation as runner
    runner.BASE=BASE;runner.PROTOCOL=PROTOCOL;runner.note=note
    original_child=runner.child
    def child(stage,arguments):
        if arguments[0]=='scripts/annotate_independent_sixclass_pool.py':
            arguments=['scripts/run_objects365_confirmation.py',*arguments[1:]]
        return original_child(stage,arguments)
    runner.child=child
    registration()
    write_json(BASE/'process.json',dict(pid=os.getpid(),started_at=now(),log=str(BASE/'pipeline.log'),command=sys.argv))
    try:
        while args.source_pid:
            try:os.kill(args.source_pid,0)
            except ProcessLookupError:break
            note('source_audit',source_pid=args.source_pid,log='source_audit.log');time.sleep(20)
        audit=BASE/'metadata/source_audit.json'
        if not audit.exists() or json.loads(audit.read_text()).get('split')!='train':
            runner.run_stage('source_audit',['scripts/audit_objects365_confirmation_source.py','--split','train'])
        value=json.loads(audit.read_text())
        if value['status']!='pass':
            write_json(BASE/'progress.json',dict(status='source_unavailable',stage='source_audit',source_audit=str(audit),
                reason='class counts or bounded archive budget insufficient; no classifier prediction executed',heartbeat=now()))
            return
        common.verify(BASE/'execution_code_freeze.sha256.json')
        runner.run_stage('prepare',['scripts/prepare_objects365_confirmation.py'])
        common.verify(BASE/'preparation_freeze.sha256.json')
        runner.annotations()
        for stage in ('freeze_selection','support','infer','confirm','compare','transfer','report'):
            common.verify(BASE/'execution_code_freeze.sha256.json')
            runner.run_stage(stage,['scripts/analyze_objects365_confirmation.py','--stage',stage])
        write_json(BASE/'progress.json',dict(status='complete',stage='complete',finished_at=now(),pid=os.getpid(),
            targets=2400,runs=24,report=str(BASE/'report_zh.md'),manual_review=False))
        print(f'[{now()}] COMPLETE: {BASE / "report_zh.md"}',flush=True)
    except Exception as error:
        write_json(BASE/'progress.json',dict(status='failed',stage='pipeline',error=str(error),heartbeat=now(),pid=os.getpid()))
        traceback.print_exc();raise


if __name__=='__main__':main()
