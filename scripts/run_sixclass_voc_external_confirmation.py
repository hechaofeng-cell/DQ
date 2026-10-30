#!/usr/bin/env python3
"""Logged/resumable execution of external-source confirmation and evidence packaging."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time
import traceback
from voc_external_common import *

def annotate(backend,smoke):
    import annotate_independent_sixclass_pool as adapter
    adapter.BASE=BASE;adapter.configure()
    ledger=BASE/'cost/request_attempts.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock()
    def record(body,status,text,elapsed,started):
        try:response=json.loads(text)
        except (ValueError,TypeError):response={}
        messages=body.get('messages',[]);content=messages[0].get('content','') if messages else ''
        if isinstance(content,list):content='\n'.join(v.get('text','') for v in content if v.get('type')=='text')
        ids=re.findall(r'\nimage_id:\s*([^\s]+)',content)
        usage=response.get('usage') if backend=='online' else {'prompt_tokens':response.get('prompt_eval_count'),'completion_tokens':response.get('eval_count')}
        row=dict(backend=backend,model=body.get('model'),sample_id=ids[-1] if ids else None,
          started_at=started,finished_at=common.now(),elapsed_seconds=elapsed,http_status=status,
          usage=usage,usage_available=bool(usage) and any(v is not None for v in usage.values()),
          local_server_duration_seconds=response.get('total_duration',0)/1e9 if backend=='local' else None)
        with lock:
            with ledger.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    if backend=='online':
        original=adapter.adapter.online.post_json
        def request(url,body,headers,timeout):
            started=common.now();begin=time.monotonic()
            try:
                out=original(url,body,headers,timeout);record(body,out[0],out[1],out[2],started);return out
            except Exception:
                record(body,None,'{}',time.monotonic()-begin,started);raise
        adapter.adapter.online.post_json=request
    else:
        original=adapter.adapter.local.post
        def request(url,body,timeout):
            started=common.now();begin=time.monotonic()
            try:
                out=original(url,body,timeout);record(body,out[0],out[1],out[2],started);return out
            except Exception:
                record(body,None,'{}',time.monotonic()-begin,started);raise
        adapter.adapter.local.post=request
    for domain in ('old','car'):
        (adapter.adapter.run_local if backend=='local' else adapter.adapter.run_online)(domain,0,smoke)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-pid',type=int,default=0)
    parser.add_argument('--backend',choices=['local','online']);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    common.BASE=BASE;common.PROTOCOL=PROTOCOL
    if args.backend:annotate(args.backend,args.smoke);return
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import run_sixclass_independent_confirmation as runner
    runner.BASE=BASE;runner.PROTOCOL=PROTOCOL
    original_child=runner.child
    def child(stage,arguments):
        if arguments[0]=='scripts/annotate_independent_sixclass_pool.py':arguments=['scripts/run_sixclass_voc_external_confirmation.py',*arguments[1:]]
        return original_child(stage,arguments)
    runner.child=child
    adapt(runner,'annotations',[('targets=2400,strict_per_backend=2400','targets=TOTAL,strict_per_backend=TOTAL')])
    path=BASE/'registration.json'
    if not path.exists():
        common.write_json(path,dict(registered_at=common.now(),protocol_sha256=common.sha256(PROTOCOL),classes=list(common.CLASSES),models=list(common.MODELS),seeds=list(common.SEEDS),
            targets_per_class=N,tier_size=TIER_N,budgets=list(BUDGETS),training_frozen=True,
            primary_risk_family=6,primary_diagnostic_family=4,transfer_family=2,strong_baseline_family=12,global_sensitivity_family=20,
            random_replicates=1000,dataset_source='VOC2007_trainval_and_test',historical_exclusion_required=True))
    common.freeze(BASE/'execution_code_freeze.sha256.json',code_paths())
    common.write_json(BASE/'process.json',dict(pid=os.getpid(),started_at=common.now(),log=str(BASE/'pipeline.log')))
    if not (BASE/'paper_brief.md').exists():
        (BASE/'paper_brief.md').write_text('# 外部来源确认\n\n固定模型的可解释状态测试与诊断；VOC六类确认，原24权重不训练。禁止全面优势、因果、人工真值或普遍排名声明。\n')
        (BASE/'claim_evidence.md').write_text('# 待检验结论\n\n风险响应→confirmation；共同风险诊断→comparison；冻结状态→transfer；依赖性→robustness。执行完不代表假设通过。\n')
        (BASE/'review.md').write_text('# 当前审阅\n\n外部实验执行中，结果未判定。已有COCO证据保持原样。\n')
    with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{common.now()}] 启动/续跑，PID={os.getpid()}；VOC2012完整历史排除后dog仅47，停止；预测前改VOC2007，100/class。\n')
    try:
        while args.prepare_pid and not (BASE/'preparation_audit.json').exists():
            try:os.kill(args.prepare_pid,0)
            except ProcessLookupError:raise RuntimeError('Preparation exited before audit; see prepare.log')
            common.note('prepare',child_pid=args.prepare_pid);time.sleep(20)
        if not (BASE/'preparation_audit.json').exists():
            runner.run_stage('download',['scripts/download_sixclass_voc_sources.py'])
            runner.run_stage('prepare',['scripts/prepare_sixclass_voc_external.py'])
        common.verify(BASE/'preparation_freeze.sha256.json');runner.annotations()
        for stage in ('freeze_selection','infer','confirm','compare','transfer','robustness','report'):
            common.verify(BASE/'execution_code_freeze.sha256.json')
            runner.run_stage(stage,['scripts/analyze_sixclass_voc_external.py','--stage',stage])
        runner.run_stage('evidence',['scripts/strengthen_empirical_evidence.py','--include-external'])
        comparisons=common.read_csv(BASE/'comparison/tables/comparisons.csv')
        primary=[r for r in comparisons if r['baseline']=='random']
        count=sum(float(r['mean_delta_auc'])>0 and float(r['holm_primary4_p'])<.05 for r in primary)
        (BASE/'claim_evidence.md').write_text('# 外部确认结论—证据\n\n'
            '风险响应：confirmation/tables/difficulty_contrasts.csv，六项校正与源图区间并报。\n\n'
            f'同风险Random对照：comparison/tables/comparisons.csv，主四项正向且家族通过{count}/4；全20、强基线、删字段与统一ViT不省略。\n\n'
            '状态迁移：transfer/tables/state_validation.csv，原A冻结清单保留不可评价项。\n\n'
            '成本和案例：../empirical_evidence_strengthening_20261005；金额未知，案例为探索性关联。\n\n'
            '只允许具体固定模型、候选池和预算下的条件性结论；禁止全面优势、因果、人工标签准确和首创声明。\n')
        (BASE/'review.md').write_text('# 执行后审阅\n\n'
            '结果可追踪：通过执行完整性审计、输入/权重冻结及表格来源检查；执行通过不等于假设支持。\n\n'
            '范围：全部主次和负结果保存，100/class与预算10–50及COCO比例差明确；自动标签无人工真值，有限架构和可能预训练重叠仍限制泛化。\n\n'
            '引用：本轮VOC来源与校验值有官方/PyTorch原始来源，未引入新颖性断言；旧完整稿引用及投稿模板尚须合稿核查，不宣称整篇投稿前门禁通过。\n')
        with (BASE/'research_log.md').open('a') as stream:stream.write(f'\n[{common.now()}] 全部外部阶段与成本、案例、MD/Word增补完成；科学结果见全部表，不按方向重跑。\n')
        common.write_json(BASE/'progress.json',dict(status='complete',stage='complete',heartbeat=common.now(),pid=os.getpid()))
        common.note('complete')
        common.write_json(BASE/'progress.json',dict(status='complete',stage='complete',heartbeat=common.now(),pid=os.getpid()))
    except Exception as error:
        common.write_json(BASE/'progress.json',dict(status='failed',error=str(error),heartbeat=common.now(),pid=os.getpid()))
        traceback.print_exc();raise

if __name__=='__main__':main()
