"""Small cat pilot: real v5 annotation, provisional selection, and blind human tasks.

Human labels are never filled by this script. The human-validity gate stays pending.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import run_fresh_vlm_feature_scoring as fresh
import run_qe_full_fine_annotations_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/cat_human_vlm_v5_pilot_20261017"
SELECTION = ROOT / "artifacts/cat_latest_process_pilot_20261017/manifest.csv"
VERSION = "cat-v5-human-pilot-20261017"
DEPENDENCIES = """
依赖校验（优先于字段独立描述）：若mouth_visibility=not_visible，则mouth_state和tongue_visibility都必须为unknown，不可填closed或not_visible。
若tail_visibility=not_observable，tail_posture必须unknown。未知是合法结果，不能为填满字段而推测。
"""
PROMPT = v5.PROMPT_V5 + "\n" + v5.TAIL_RULES + "\n" + DEPENDENCIES


def dump(path, value):
    fresh.atomic_json(path, value)


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def immutable(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Frozen input changed: {path}")
    dump(path, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = fresh.pilot.read_csv(SELECTION)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 20
    assert all(r['class_name'] == 'cat' for r in rows)
    for row in rows:
        with Image.open(row['image_path']) as im:
            row.update(image_width=str(im.width), image_height=str(im.height), source_pool='A_OpenImages')
        for k in ('image_path', 'marked_image_path', 'feature_crop_path'):
            assert Path(row[k]).is_file()
    manifest_path = OUT / 'manifest.csv'
    if manifest_path.exists():
        assert fresh.pilot.read_csv(manifest_path) == rows
    else:
        write(manifest_path, rows)
    schema = fresh.build_schema('cat')
    immutable(OUT/'schemas/cat.json', schema)
    immutable(OUT/'scoring_protocol.json', fresh.scoring_protocol())
    runtime = dict(model='qwen3-vl:30b-a3b-instruct', url='http://localhost:11434/api/chat',
                   num_ctx=16384, num_predict=5000, temperature=0, seed=20261017,
                   timeout_seconds=900, max_retries=0, retry_wait_seconds=2)
    immutable(OUT/'runtime.json', runtime)
    prompt_path = OUT/'prompt.txt'
    if prompt_path.exists():
        assert prompt_path.read_text() == PROMPT
    prompt_path.write_text(PROMPT)
    immutable(OUT/'protocol.json', dict(version=VERSION, category='cat', targets=20,
        selection='seed20261017 random cats fixed before predictions; no replacement of failures',
        fields=35, preserved_cat_fields=23, prompt_base='v5_concise_tail_semantic_gate',
        dependency_clarification=DEPENDENCIES, max_semantic_attempts=6,
        support_requirement=2, selection_budget=10, tier_quotas=[2]*5,
        tier_rule='sort score midpoint, then sample_id; 4 per tier; relative within pilot',
        coverage_dictionary='all legal categorical states plus 4 geometric scale bins; unknown and not_observable excluded; noncar torso/limb not_in_view excluded per v5',
        greedy='sum log(min(count,2)+1), exact 2 per tier',
        random_baseline_repetitions=200,
        human_A_targets=20, human_B_targets=20, independent_double_review=True,
        review_scope_revision='r2_fine_labels_only',
        human_scope='fine-grained visual-state label correctness only; no machine defaults',
        fine_label_validity_claim_gate='real human fine-label review required only for human-validated label claims',
        downstream_scope='scoring, ranking, quintiles, quotas, encoding, selection, reports and model evaluation are automatic; no human approval',
        single_class_metric='cat recall only; no sixclass MacroF1 or cat precision claims',
        no_training=True, no_classifier_inference=True))
    paths = [manifest_path, OUT/'schemas/cat.json', OUT/'scoring_protocol.json', OUT/'runtime.json', prompt_path,
             OUT/'protocol.json', Path(__file__), Path(__file__).with_suffix('.html'), Path(fresh.__file__), Path(v5.__file__)]
    paths += [Path(row[k]) for row in rows for k in ('image_path','marked_image_path','feature_crop_path')]
    immutable(OUT/'input_freeze.sha256.json', {str(p.resolve()):fresh.pilot.sha256(p) for p in paths})
    return rows, schema, runtime


def annotate(rows, schema, runtime):
    failures = []
    for i, row in enumerate(rows, 1):
        dest = OUT/'annotations'/f"{row['sample_id']}.json"
        if dest.exists():
            value=json.loads(dest.read_text());v5.validate_v5(value,row,schema)
            assert value['pilot_version'] == VERSION
            assert value['prompt_sha256'] == fresh.pilot.sha256(OUT/'prompt.txt')
            continue
        interior, facts = v5.frame_facts(row)
        prompt = PROMPT + '\n冻结字典：' + json.dumps(dict(schema_version=schema['schema_version'], category='cat', fields=schema['universal_features']),ensure_ascii=False)
        prompt += '\n原图几何：' + json.dumps(dict(wholly_interior_to_original_frame=interior,**facts),ensure_ascii=False)
        errors=[]
        for attempt in range(1,7):
            try:
                correction = '' if not errors else '\n上轮校验未通过：'+errors[-1]['error']+'。重新观察并输出完整JSON。'+DEPENDENCIES+v5.TAIL_RULES
                value = fresh.ask(runtime,prompt+correction,dict(image_id=row['sample_id']),
                    [Path(row[k]) for k in ('image_path','marked_image_path','feature_crop_path')],
                    OUT/'raw'/f"{row['sample_id']}_semantic{attempt}.json", fresh.feature_schema(OUT/'schemas/cat.json'))
                v5.validate_v5(value,row,schema)
                value.update(pilot_version=VERSION,annotation_model=runtime['model'],annotated_at=fresh.now(),
                    schema_sha256=fresh.pilot.sha256(OUT/'schemas/cat.json'),prompt_sha256=fresh.pilot.sha256(OUT/'prompt.txt'),
                    input_view_sha256={k:fresh.pilot.sha256(Path(row[k])) for k in ('image_path','marked_image_path','feature_crop_path')})
                dump(dest,value)
                print(f"[v5 annotated] {i}/20 {row['sample_id']} attempts={attempt}",flush=True)
                break
            except Exception as exc:
                errors.append(dict(attempt=attempt,error=str(exc),time=fresh.now()))
                print(f"[v5 retry] {i}/20 attempt={attempt} {exc}",flush=True)
        else:
            failures.append(row['sample_id'])
        if errors:
            dump(OUT/'attempt_errors'/f"{row['sample_id']}.json",errors)
        dump(OUT/'progress.json',dict(status='running',processed=i,planned=20,
            successful=len(list((OUT/'annotations').glob('*.json'))),failed=failures,updated_at=fresh.now()))
    dump(OUT/'annotation_audit.json',dict(status='incomplete' if failures else 'pass',expected=20,
        valid=20-len(failures),failures=failures,semantic_human_verified=False,predictions_read=False))
    if failures:
        raise ValueError('Unresolved annotations; keep fixed sample IDs: '+str(failures))


def thumbnail(path):
    with Image.open(path) as im:
        im=im.convert('RGB'); im.thumbnail((1050,850))
        buf=io.BytesIO(); im.save(buf,format='JPEG',quality=87)
        return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()


def review_tasks(rows, schema):
    # Intentionally never open annotations/, scores.csv, tiers.csv, or model predictions here.
    out=OUT/'human_review';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(7120261017)
    for role in ('A','B'):
        order=rng.permutation(len(rows))
        tasks=[dict(sample_id=rows[i]['sample_id'],views=[thumbnail(rows[i][k]) for k in
              ('image_path','marked_image_path','feature_crop_path')]) for i in order]
        payload=dict(pilot_version=VERSION,role=role,schema_sha256=fresh.pilot.sha256(OUT/'schemas/cat.json'),
                     fields=schema['universal_features'],tasks=tasks)
        template=(Path(__file__).with_suffix('.html')).read_text()
        (out/f'blind_review_{role}.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False)))
        blank=[dict(sample_id=t['sample_id'],annotator_id='',review_status='',
             **{f['feature_id']:'' for f in schema['universal_features']},review_notes='') for t in tasks]
        write(out/f'annotator_{role}_blind.csv',blank)
    if not (out/'status.json').exists():
        dump(out/'status.json',dict(status='awaiting_fine_label_review',human_A_completed=0,human_B_completed=0,
            review_scope='fine-grained visual-state labels only',
            automatic_experiment_complete=(OUT/'model_tier_recall.csv').exists(),
            human_ground_truth_available=False))


def export(rows, schema):
    protocol=json.loads((OUT/'scoring_protocol.json').read_text())
    states=[(f['feature_id'],str(v)) for f in schema['universal_features'] for v in f['possible_values']
        if v not in ('unknown','not_observable') and not (f['feature_id'] in ('torso_visibility','limb_visibility') and v=='not_in_view')]
    states += [('target_scale',s) for s in ('lt1pct','1to3pct','3to10pct','ge10pct')]
    index={s:i for i,s in enumerate(states)}
    X=np.zeros((20,len(states)),dtype=int);scores=[];long=[]
    for i,row in enumerate(rows):
        value=json.loads((OUT/'annotations'/f"{row['sample_id']}.json").read_text())
        v5.validate_v5(value,row,schema)
        score=fresh.fresh_score(row,value,protocol)
        scores.append(dict(sample_id=row['sample_id'],class_name='cat',**score))
        for f in schema['universal_features']:
            key=f['feature_id']; val=value['features'][key]
            long.append(dict(sample_id=row['sample_id'],field=key,value=json.dumps(val,ensure_ascii=False),evidence=value['evidence'][key]))
            for v in (val if isinstance(val,list) else [val]):
                if (key,str(v)) in index:X[i,index[(key,str(v))]]=1
        a=float(row['bbox_area_ratio']);s='lt1pct' if a<.01 else '1to3pct' if a<.03 else '3to10pct' if a<.1 else 'ge10pct'
        X[i,index[('target_scale',s)]]=1
    write(OUT/'scores.csv',scores);write(OUT/'feature_annotations_with_evidence.csv',long)
    order=sorted(range(20),key=lambda i:(scores[i]['score_midpoint'],rows[i]['sample_id']))
    tier={i:pos//4+1 for pos,i in enumerate(order)}
    write(OUT/'tiers.csv',[dict(sample_id=rows[i]['sample_id'],tier=tier[i],score_midpoint=scores[i]['score_midpoint'],score_lower=scores[i]['score_lower'],score_upper=scores[i]['score_upper']) for i in order])
    write(OUT/'state_encoding.csv',[dict(sample_id=r['sample_id'],**{f'{f}={v}':int(X[i,j]) for j,(f,v) in enumerate(states)}) for i,r in enumerate(rows)])
    selected=[];trace=[];counts=np.zeros(len(states),int);used=Counter()
    for step in range(10):
        eligible=[i for i in range(20) if i not in selected and used[tier[i]]<2]
        gains={i:float((np.log1p(np.minimum(counts+X[i],2))-np.log1p(np.minimum(counts,2))).sum()) for i in eligible}
        best=min(eligible,key=lambda i:(-gains[i],rows[i]['sample_id']))
        selected.append(best);counts+=X[best];used[tier[best]]+=1
        trace.append(dict(step=step+1,sample_id=rows[best]['sample_id'],tier=tier[best],marginal_gain=gains[best],states_present=int((counts>0).sum()),states_supported=int((counts>=2).sum())))
    assert [used[q] for q in range(1,6)]==[2]*5
    write(OUT/'greedy_trace.csv',trace)
    write(OUT/'provisional_test_set.csv',[dict(rows[i],tier=tier[i],selection_status='automatic_provisional_not_human_validated') for i in selected])
    rng=np.random.default_rng(20261017)
    random_sets=[[int(i) for q in range(1,6) for i in rng.choice([i for i in range(20) if tier[i]==q],2,replace=False)] for _ in range(200)]
    write(OUT/'random_selections.csv',[dict(repetition=j,sample_id=rows[i]['sample_id'],tier=tier[i]) for j,sel in enumerate(random_sets) for i in sel])
    def coverage(sel):
        n=X[sel].sum(axis=0)
        return dict(n=len(sel),states_present=int((n>0).sum()),states_supported=int((n>=2).sum()),dictionary_size=len(states),occurrence_coverage=float((n>0).mean()),support_coverage=float((n>=2).mean()))
    cov=[dict(method='candidate_pool',**coverage(list(range(20)))),dict(method='risk_constrained_greedy',**coverage(selected))]
    rc=[coverage(sel) for sel in random_sets]
    cov.append(dict(method='quota_matched_random_mean',**{k:float(np.mean([c[k] for c in rc])) for k in rc[0]}))
    write(OUT/'coverage_summary.csv',cov)
    write(OUT/'state_support.csv',[dict(field=f,value=v,pool_count=int(X[:,j].sum()),selected_count=int(counts[j]),support_requirement=2) for j,(f,v) in enumerate(states)])
    frozen={str(OUT/name):fresh.pilot.sha256(OUT/name) for name in ['tiers.csv','state_encoding.csv','greedy_trace.csv','provisional_test_set.csv','random_selections.csv']}
    immutable(OUT/'selection_frozen_before_predictions.json',dict(frozen_at=fresh.now(),predictions_read=False,files=frozen)) if not (OUT/'selection_frozen_before_predictions.json').exists() else None
    for p,digest in json.loads((OUT/'selection_frozen_before_predictions.json').read_text())['files'].items():assert fresh.pilot.sha256(Path(p))==digest
    # Prediction access starts only after state definitions and selected target IDs are frozen.
    original=fresh.pilot.read_csv(fresh.POOL/'manifests/candidate_pool.csv')
    ids=[r['sample_id'] for r in original];lookup={sid:i for i,sid in enumerate(ids)}
    metrics=[];binding=[]
    for p in sorted((fresh.POOL/'benchmark/runs').glob('*/*/candidate_outputs.npz')):
        with np.load(p) as cache:
            assert cache['sample_ids'].tolist()==ids
            logits=cache['logits'];assert logits.shape==(len(ids),6)
            pred=logits.argmax(1)
        correct=np.array([pred[lookup[r['sample_id']]]==1 for r in rows],dtype=int)
        binding.append(dict(path=str(p),sha256=fresh.pilot.sha256(p),identity_verified=True))
        for method,sel in [('candidate_pool',list(range(20))),('risk_constrained_greedy',selected)]:
            for q in range(1,6):
                cell=[i for i in sel if tier[i]==q]
                metrics.append(dict(method=method,model=p.parent.parent.name,seed=p.parent.name,tier=q,n=len(cell),recall=float(correct[cell].mean())))
        for q in range(1,6):
            recalls=[float(correct[[i for i in sel if tier[i]==q]].mean()) for sel in random_sets]
            metrics.append(dict(method='quota_matched_random_mean',model=p.parent.parent.name,seed=p.parent.name,tier=q,n=2,recall=float(np.mean(recalls))))
    assert len(binding)==24
    write(OUT/'model_tier_recall.csv',metrics);dump(OUT/'classifier_cache_binding_audit.json',binding)
    dump(OUT/'audit.json',dict(automatic_pipeline='pass',targets=20,fields_per_target=35,field_evidence_records=len(long),
        all_annotations_v5_validated=True,tiers=dict(Counter(tier.values())),selected=10,exact_tier_quotas=[used[q] for q in range(1,6)],
        dictionary_states=len(states),prediction_runs=24,selection_read_predictions=False,
        fine_label_human_validation='pending',automatic_experiment_complete=True,
        downstream_human_approval_required=False))
    lines=['# 猫类人工—VLM协同流程：20目标试跑','',
        '自动实验已完成。人工仅复核细粒度视觉状态标签；评分、五等分、配额、覆盖选样、报告和模型评价由程序自动完成。当前标签尚未通过人工复核。','',
        '固定20个随机猫目标，原图/目标标记图/裁剪三视图，35字段（保留原猫23字段）。实际重新请求本地Qwen3-VL，使用v5语义约束和显式嘴部/尾部依赖校验，保留原始响应与重试。','',
        f'已输出{len(long)}条字段证据、五维评分、20×{len(states)}二值编码。按类内中点分+sample_id排序，五等分每层4例；按每层2例的精确配额贪心选择10例。覆盖支持门槛K=2仅用于小样本流程验证，不沿用到正式实验。','',
        '## 自动标签下的覆盖结果（暂定）','','|集合|目标数|出现状态|达到2例支持|字典状态数|','|---|---:|---:|---:|---:|']
    for c in cov:lines.append(f"|{c['method']}|{c['n']:.0f}|{c['states_present']:.2f}|{c['states_supported']:.2f}|{c['dictionary_size']:.0f}|")
    lines += ['', '字典分母包括未出现的全部合法状态；unknown/not_observable不算已覆盖，v5禁止的躯干/肢体not_in_view不在分母。200次相同配额随机抽样只作描述性对照。以覆盖为目标选样后覆盖升高本身不证明独立效用。','',
        '## 冻结模型评价','', '读取8架构×3种子缓存后验证全部目标ID和6类logits绑定。由于只有猫正例，只报告猫类Recall；不报告六类Macro-F1或猫类Precision。候选层每层4例，选后每层2例，误差很大，不作显著性和优越性判断。逐模型结果见model_tier_recall.csv。','',
        '## 人工协同现在怎么做','',
        '1. 两位真实标注者分别打开human_review/blind_review_A.html和blind_review_B.html，在不同浏览器配置或设备中独立操作。页面嵌入原图/标记图/裁剪和字典，不包含任何VLM答案、模型预测或Q层。',
        '2. 按现有字段定义观察图片，只复核细粒度状态是否准确、遗漏或无法判断，无需人工审批字典或算法。',
        '3. 每人对20个目标的35字段独立选择，无法判断明确填unknown。页面无机器预填，下载各自的JSON结果。',
        '4. 用本脚本--phase compare --human-a A.json --human-b B.json检查完成度并输出逐字段一致性、集合Jaccard、Kappa和分歧清单。没有真实输入时不生成可靠性数值或human_frozen。',
        '5. 人工分歧处理只针对标签。需要使用修正标签时，由程序自动重新编码、评分、分层、选样和评价，无需人工介入后续算法。现有自动结果可独立报告，不能称为人工已验证标签。','',
        '## 对既有工作的修正','',
        '不能把其他类别概括为没有自动细粒度标注：既有QE_FullFine_v5_20261013的历史审计记录共有7229例，其中猫1053例自动通过校验。但它没有完成多类别人工协同验证。本试跑主要补齐可操作的盲审入口与可追溯的小闭环，而非把旧自动标注当成人工证明。','',
        '复现：.venv/bin/python scripts/run_cat_human_vlm_pilot_20261017.py --phase all','']
    (OUT/'report.md').write_text('\n'.join(lines))


def compare(rows,schema,a_path,b_path):
    from sklearn.metrics import cohen_kappa_score
    people=[]
    for role,path in [('A',a_path),('B',b_path)]:
        if path is None:raise ValueError('Real human exports A and B are required')
        d=json.loads(path.read_text())
        assert d['pilot_version']==VERSION and d['role']==role
        assert d['schema_sha256']==fresh.pilot.sha256(OUT/'schemas/cat.json')
        assert d['reviewer_id'].strip()
        records=d['records'];assert len(records)==20 and len({r['sample_id'] for r in records})==20
        assert {r['sample_id'] for r in records}=={r['sample_id'] for r in rows}
        for r in records:
            assert r['review_status']=='complete'
            assert set(r['features'])=={f['feature_id'] for f in schema['universal_features']}
            for f in schema['universal_features']:
                v=r['features'][f['feature_id']]
                if f['value_type']=='set':
                    assert isinstance(v,list) and v and len(set(v))==len(v) and set(v)<=set(f['possible_values'])
                    assert not (set(v)&set(f.get('exclusive_values',[])) and len(v)>1)
                else:assert not isinstance(v,list) and v in f['possible_values']
            f=r['features']
            if f['mouth_visibility']=='not_visible':assert f['mouth_state']==f['tongue_visibility']=='unknown'
            if f['tail_visibility']=='not_observable':assert f['tail_posture']=='unknown'
        people.append(d)
    assert people[0]['reviewer_id'].strip()!=people[1]['reviewer_id'].strip(),'Require two different real reviewers'
    maps=[{r['sample_id']:r['features'] for r in p['records']} for p in people]
    stats=[];diff=[]
    for f in schema['universal_features']:
        key=f['feature_id'];left=[];right=[];jaccards=[]
        for row in rows:
            sid=row['sample_id'];a=maps[0][sid][key];b=maps[1][sid][key]
            norm=lambda v:json.dumps(sorted(v) if isinstance(v,list) else v,ensure_ascii=False)
            left.append(norm(a));right.append(norm(b))
            if isinstance(a,list):jaccards.append(len(set(a)&set(b))/len(set(a)|set(b)))
            if norm(a)!=norm(b):diff.append(dict(sample_id=sid,field=key,human_A=norm(a),human_B=norm(b),adjudicated_value='',adjudicator_id='',reason=''))
        kappa=cohen_kappa_score(left,right) if len(set(left+right))>1 else None
        stats.append(dict(field=key,n=20,exact_agreement=sum(a==b for a,b in zip(left,right))/20,
            cohen_kappa=None if kappa is None or not math.isfinite(kappa) else float(kappa),
            set_jaccard=None if not jaccards else float(np.mean(jaccards))))
    write(OUT/'human_review/agreement.csv',stats)
    write(OUT/'human_review/disagreements.csv',diff,fields=['sample_id','field','human_A','human_B','adjudicated_value','adjudicator_id','reason'])
    dump(OUT/'human_review/status.json',dict(status='awaiting_adjudication' if diff else 'double_review_complete_pending_freeze',
        human_A_completed=20,human_B_completed=20,review_scope='fine-grained visual-state labels only',
        downstream_human_approval_required=False,
        disagreements=len(diff),human_ground_truth_available=False,
        input_sha256={str(p):fresh.pilot.sha256(p) for p in (a_path,b_path)}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','annotate','export','compare'],default='all')
    p.add_argument('--human-a',type=Path);p.add_argument('--human-b',type=Path);args=p.parse_args()
    rows,schema,runtime=prepare()
    if args.phase in ('prepare','all'):review_tasks(rows,schema)
    if args.phase in ('annotate','all'):annotate(rows,schema,runtime)
    if args.phase in ('export','all'):export(rows,schema)
    if args.phase=='compare':compare(rows,schema,args.human_a,args.human_b)


if __name__=='__main__':main()
