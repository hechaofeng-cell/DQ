#!/usr/bin/env python3
"""Resumable full-pool local fine VLM annotation before frozen-rule scoring."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import argparse,json,os,time,urllib.request,traceback,shutil
from PIL import Image
import run_fresh_vlm_feature_scoring as fresh
from run_dog_local_pipeline import ask,feature_schema

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/QE_VisualState_CoverRisk_v1_20261013'
OUT=ROOT/'artifacts/QE_FullFine_v5_20261013'
OLD=ROOT/'artifacts/fresh_vlm_feature_scoring_20261012'
CLASSES=fresh.CLASSES
PROMPT_V5=fresh.PROMPT+"""
统一v5补充：以原图判断截断，官方框不接触原图边界则none/[none]。非车躯干肢体不可用not_in_view；原图外缺失为not_visible。模糊低清但轮廓存在为partial/unknown。灯光烟雾不属于不透明物体遮挡。"""
TAIL_RULES="""
最高优先级判定检查 / FINAL PRIORITY RULES:
1. ORIGINAL FRAME, NOT CROP: use the attached original-frame geometry. If wholly_interior_to_original_frame=true, truncation_level MUST be none and truncation_parts MUST [none]. Tight crop borders, blur and occlusion do not truncate the original image.
2. SILHOUETTE: if a head/torso/limb outline can be seen but details are blurred, visibility is partial or unknown. Never not_visible solely from missing fine detail. Noncar torso/limb never not_in_view; actual parts outside original frame are not_visible.
3. OPAQUE OCCLUDER ONLY: occlusion requires a clearly visible concrete opaque object blocking a named target part. Stage lighting, lasers, smoke, fog, blur, shadows, backlight and low contrast are illumination/blur conditions, NOT opaque-object occlusion. If you cannot identify the opaque object and location reliably, use occlusion_level unknown and occluder_type [unknown]. Do not invent a foreground object.
中文检查：模糊但有轮廓→partial或unknown；明确物体挡住才有遮挡；灯光/烟雾/模糊不算物体遮挡。无法确认不透明遮挡物时，用unknown，不编造heavy/other_object。
Valid ambiguity example (not a label for this image): features.occlusion_level="unknown", features.occluder_type=["unknown"], evidence.occlusion_level="低清光照不足，无法确认是否有物体遮挡". An unclear condition is a valid complete annotation; do not guess to fill fields.
Output only the complete schema-bound JSON features and corresponding evidence. Every value must be a listed enum.
"""

BLUR_WORDS=('模糊','低清','光照','灯光','舞台光','烟雾','烟','雾','逆光','阴影','过曝','blur','lighting','smoke','fog','backlight','shadow')
LOSS_WORDS=('原图外','原图边界','画面外','截断','遮挡','挡住','被挡','图像外','边界之外','边缘之外','outside','truncat','occlud','blocked')

def validate_v5(value,row,schema):
    fresh.validate_annotation(value,row,schema)
    if row['class_name']!='car':
        for key in ('torso_visibility','limb_visibility'):
            if value['features'][key]=='not_in_view':raise ValueError('v5noncar '+key+' cannot be not_in_view; frame/crop loss is not_visible or unknown')
    level=value['features']['occlusion_level']
    evidence=value['evidence']['occlusion_level'].lower()
    if level not in ('none','unknown') and any(word in evidence for word in BLUR_WORDS):
        raise ValueError('v5 positive occlusion evidence cites blur/light/smoke rather than a concrete opaque occluder and location. Do not invent foreground objects. Use unknown if no independently visible opaque blocker; never raw lighting/smoke as heavy occlusion.')
    if row['class_name']!='car':
        for key in ('head_visibility','torso_visibility','limb_visibility'):
            observation=value['evidence'][key].lower()
            if value['features'][key]=='not_visible' and any(word in observation for word in BLUR_WORDS):
                raise ValueError('v5 '+key+' not_visible supported by blur/light/smoke. Recognizable silhouette with insufficient detail is partial or unknown; actual full frame loss/opaque occlusion must be independently supported. Reobserve original image.')
    interior,_=frame_facts(row)
    if interior:
        if value['features']['truncation_level']!='none':raise ValueError('v5 original official bbox is wholly interior beyond2px: truncation_level must none; crop edge/blur is not original frame truncation')
        if 'truncation_parts' in value['features'] and value['features']['truncation_parts']!=['none']:raise ValueError('v5 interior original bbox: truncation_parts must [none]')
    return value

def frame_facts(row):
    box=json.loads(row['bbox_norm_xyxy']);w=int(row['image_width']);h=int(row['image_height'])
    distances={'left':box[0]*w,'top':box[1]*h,'right':(1-box[2])*w,'bottom':(1-box[3])*h}
    contact={k:v<=2 for k,v in distances.items()}
    return not any(contact.values()),dict(original_width=w,original_height=h,official_bbox_xyxy_pixels=[box[0]*w,box[1]*h,box[2]*w,box[3]*h],frame_contact_tolerance_pixels=2,contacts_original_frame=contact,original_frame_distance_pixels=distances)


def dump(path,obj):fresh.atomic_json(path,obj)

def event(stage,**extra):
    value=dict(stage=stage,updated_at=fresh.now(),pid=os.getpid(),**extra)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)

def immutable(path,value,json_file=True):
    if path.exists():
        old=json.loads(path.read_text()) if json_file else path.read_text()
        assert old==value,('frozen value changed',str(path))
    elif json_file:dump(path,value)
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=fresh.pilot.read_csv(DATA/'manifest.csv')
    assert len(manifest)==7800 and len({r['sample_id'] for r in manifest})==7800
    assert Counter(r['class_name'] for r in manifest)==dict.fromkeys(CLASSES,1300)
    byclass={c:[r for r in manifest if r['class_name']==c] for c in CLASSES}
    manifest=[dict(byclass[c][j]) for j in range(1300) for c in CLASSES]
    freeze={str(DATA/'manifest.csv'):fresh.pilot.sha256(DATA/'manifest.csv')}
    for row in manifest:
        assert row['annotation_target_instance_id']=='target_1'
        row['official_target_instance_id']=row['target_instance_id']
        row['target_instance_id']=row['annotation_target_instance_id']
        for field in ('image_path','marked_image_path','feature_crop_path'):
            p=(DATA/row[field]).resolve();assert p.is_file();row[field]=str(p);freeze[str(p)]=fresh.pilot.sha256(p)
        assert freeze[row['image_path']]==row['image_sha256']
        with Image.open(row['image_path']) as im:
            assert im.size==(int(row['image_width']),int(row['image_height']))
            bbox=json.loads(row['bbox_norm_xyxy']);assert len(bbox)==4 and 0<=bbox[0]<bbox[2]<=1+3e-6 and 0<=bbox[1]<bbox[3]<=1+3e-6
            row['annotation_bbox']=json.dumps([round(bbox[0]*im.width),round(bbox[1]*im.height),round(bbox[2]*im.width),round(bbox[3]*im.height)])
            area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]);assert abs(area-float(row['bbox_area_ratio']))<1e-6
    mpath=OUT/'manifest.csv'
    if mpath.exists():assert fresh.pilot.read_csv(mpath)==manifest
    else:fresh.pilot.write_csv(mpath,manifest)
    runtime=dict(model='qwen3-vl:30b-a3b-instruct',url='http://localhost:11434/api/chat',num_ctx=16384,num_predict=4000,temperature=0,seed=20261012,timeout_seconds=900,max_retries=0,retry_wait_seconds=2)
    immutable(OUT/'runtime.json',runtime)
    immutable(OUT/'prompt.txt',PROMPT_V5+'\n[TAIL AFTER DICTIONARY AND TARGET GEOMETRY]\n'+TAIL_RULES,False)
    protocol=fresh.scoring_protocol();immutable(OUT/'scoring_protocol.json',protocol)
    immutable(OUT/'protocol.json',dict(targets=7800,per_class=1300,local_inference_only=True,model=runtime['model'],schemas='original fresh26-35 fields; all23dog/cat fields preserved',
        views='packaged original, marked target, feature crop; bound to original official box and target_1alias',
        reuse='no old prompt annotations reused; only samev5 validated target checkpoints with exactschema/prompt/viewSHA are resumed',
        five_dimensional_scoring='original fresh_score: originalbbox scale; noncar head/torso/limb visibility; car8parts excludingnotinview; body+headorientation; occlusion/truncation20each',
        nominal='all described, no invented ordinal hair/color/texture risk weights',
        predictions_read=False,semantic_retries=3,resumable=True,prompt_version='v5_concise_tail_semantic_gate',frame_gate='official_bbox interior >2px -> truncation none and partsnone; original source bbox rounding tolerance3e-6',semantic_gate='conservative keyword evidence rule: positive occlusion may not cite blur/light/smoke; noncar not_visible may not be justified by these. Retry actualVLM, no label rewriting. Imperfect rule not semantic ground truth',scoring_scope='interim frozen fine-v1 scores; not allfinefields individually in score; parent may establish separately versioned extension',
        stopping='complete only when all7800 nativeJSONs validated; failures pending and explicitly reported; no budget sampling'))
    for c in CLASSES:immutable(OUT/'schemas'/f'{c}.json',fresh.build_schema(c))
    with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=30) as response:tags=json.load(response)
    model=next(m for m in tags['models'] if m['name']==runtime['model'])
    immutable(OUT/'model_provenance.json',model)
    for p in [Path(__file__),ROOT/'scripts/run_fresh_vlm_feature_scoring.py',ROOT/'scripts/run_dog_local_pipeline.py',ROOT/'scripts/run_category_five_dimension_pilot.py',mpath,OUT/'runtime.json',OUT/'prompt.txt',OUT/'scoring_protocol.json',OUT/'protocol.json',OUT/'model_provenance.json']+[OUT/'schemas'/f'{c}.json' for c in CLASSES]:freeze[str(p.resolve())]=fresh.pilot.sha256(p)
    immutable(OUT/'input_freeze.sha256.json',freeze)
    event('prepared',targets=7800,frozen_inputs=len(freeze),fields={c:len(fresh.build_schema(c)['universal_features']) for c in CLASSES})
    return manifest,runtime,protocol,freeze

def try_reuse(row,schema,freeze):
    # Older prompt versions are intentionally excluded after semantic spot review.
    return None


def score_one(row,value,protocol):
    score=fresh.fresh_score(row,value,protocol)
    dump(OUT/'scores'/f"{row['sample_id']}.json",dict(sample_id=row['sample_id'],class_name=row['class_name'],source_pool=row['source_pool'],scoring_protocol='fresh_five_dimensions_v1',**score))
    return score

def worker(row,runtime,protocol,freeze,allow_reuse):
    sid=row['sample_id'];schema_path=OUT/'schemas'/f"{row['class_name']}.json";schema=json.loads(schema_path.read_text());dest=OUT/'annotations'/f'{sid}.json'
    if dest.exists():
        value=validate_v5(json.loads(dest.read_text()),row,schema)
        assert value['annotation_prompt_version']=='v5_concise_tail_semantic_gate'
        assert value['schema_sha256']==fresh.pilot.sha256(schema_path)
        assert value['prompt_sha256']==fresh.pilot.sha256(OUT/'prompt.txt')
        assert value['input_view_sha256']=={k:freeze[row[k]] for k in ('image_path','marked_image_path','feature_crop_path')}
        score_one(row,value,protocol)
        return dict(sample_id=sid,class_name=row['class_name'],status='resumed',elapsed=0)
    if allow_reuse:
        try:
            value=try_reuse(row,schema,freeze)
            if value:
                dump(dest,value);score_one(row,value,protocol)
                return dict(sample_id=sid,class_name=row['class_name'],status='reused_verified',elapsed=0)
        except Exception as exc:
            dump(OUT/'reuse_rejections'/f'{sid}.json',dict(error=str(exc),sample_id=sid))
    run_id=fresh.now().replace(':','').replace('+','_').replace('.','_')
    prompt=PROMPT_V5+'\n冻结特征字典：\n'+json.dumps(dict(schema_version=schema['schema_version'],category=row['class_name'],fields=schema['universal_features']),ensure_ascii=False)
    interior,facts=frame_facts(row)
    prompt+='\n此目标的原图几何事实（来自官方目标框，非裁剪边界）：'+json.dumps(dict(target_id=sid,wholly_interior_to_original_frame=interior,**facts),ensure_ascii=False)
    prompt+=TAIL_RULES
    errors=[];started=time.monotonic()
    for attempt in range(1,4):
        try:
            correction='' if not errors else '\n上次校验失败：'+errors[-1]['error']+'。重新观察原图：灯光/烟雾/模糊不能证明heavy遮挡。看不清不透明遮挡物具体位置时用unknown，勿猜。轮廓存在但细节模糊用partial/unknown。完整输出全部JSON字段。'+TAIL_RULES
            value=ask(runtime,prompt+correction,dict(image_id=sid),[Path(row[k]) for k in ('image_path','marked_image_path','feature_crop_path')],OUT/'raw'/f'{sid}_{run_id}_semantic{attempt}.json',feature_schema(schema_path))
            validate_v5(value,row,schema)
            value.update(annotation_prompt_version='v5_concise_tail_semantic_gate',original_frame_geometry=facts,annotation_model=runtime['model'],annotated_at=fresh.now(),schema_sha256=fresh.pilot.sha256(schema_path),prompt_sha256=fresh.pilot.sha256(OUT/'prompt.txt'),input_view_sha256={k:freeze[row[k]] for k in ('image_path','marked_image_path','feature_crop_path')})
            score_one(row,value,protocol);dump(dest,value)
            if errors:dump(OUT/'attempt_errors'/f'{sid}_{run_id}.json',errors)
            failpath=OUT/'failures'/f'{sid}.json'
            if failpath.exists():failpath.unlink()
            return dict(sample_id=sid,class_name=row['class_name'],status='fresh',elapsed=time.monotonic()-started,attempts=attempt)
        except Exception as exc:
            errors.append(dict(attempt=attempt,error=str(exc),updated_at=fresh.now()));
    dump(OUT/'failures'/f'{sid}.json',dict(sample_id=sid,status='pending',attempt_errors=errors))
    return dict(sample_id=sid,class_name=row['class_name'],status='pending',elapsed=time.monotonic()-started,error=errors[-1]['error'])

def export(manifest,protocol):
    scores=[];long=[];missing=[]
    for row in manifest:
        path=OUT/'annotations'/f"{row['sample_id']}.json"
        if not path.exists():missing.append(row['sample_id']);continue
        schema=json.loads((OUT/'schemas'/f"{row['class_name']}.json").read_text());value=validate_v5(json.loads(path.read_text()),row,schema)
        assert value['annotation_prompt_version']=='v5_concise_tail_semantic_gate' and value['prompt_sha256']==fresh.pilot.sha256(OUT/'prompt.txt')
        score=fresh.fresh_score(row,value,protocol);scores.append(dict(sample_id=row['sample_id'],class_name=row['class_name'],source_pool=row['source_pool'],**score,**{'feature_'+k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in value['features'].items()}))
        for field in schema['universal_features']:
            k=field['feature_id'];long.append(dict(sample_id=row['sample_id'],class_name=row['class_name'],source_pool=row['source_pool'],field=k,feature_name=field.get('feature_name_zh',k),value=json.dumps(value['features'][k],ensure_ascii=False),evidence=value['evidence'][k]))
    if scores:fresh.pilot.write_csv(OUT/'scores.csv',scores);fresh.pilot.write_csv(OUT/'feature_annotations_with_evidence.csv',long)
    dump(OUT/'annotation_audit.json',dict(status='pass' if not missing else 'incomplete',expected=7800,valid=len(scores),feature_rows=len(long),missing=len(missing),missing_sample_ids=missing,by_class=dict(Counter(r['class_name'] for r in scores)),fully_known=sum(r['fully_known'] for r in scores),evaluable=sum(r['evaluable'] for r in scores),predictions_read=False))
    return len(scores)

def run(manifest,runtime,protocol,freeze,workers,limit,force_fresh):
    work=manifest[:limit] if limit else manifest
    done=Counter();started=time.monotonic();active={};iterator=iter(work)
    def progress():
        valid=len(list((OUT/'annotations').glob('*.json')));elapsed=time.monotonic()-started
        rate=done['fresh']/elapsed if elapsed else 0
        state=dict(status='running',stage='fine_annotation',planned=7800,current_batch=len(work),successful=valid,newly_processed=sum(done.values()),states=dict(done),workers=workers,active_samples=list(active.values()),elapsed_seconds=elapsed,fresh_targets_per_minute=60*rate,eta_seconds=(7800-valid)/rate if rate else None,updated_at=fresh.now(),pid=os.getpid())
        dump(OUT/'progress.json',state)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers):
            row=next(iterator,None)
            if row:active[executor.submit(worker,row,runtime,protocol,freeze,not force_fresh)]=row['sample_id']
        while active:
            complete,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            if not complete:progress();event('heartbeat',active=list(active.values()),valid=sum(done.values()));continue
            for future in complete:
                sid=active.pop(future)
                try:result=future.result()
                except Exception as exc:result=dict(sample_id=sid,status='pending',error=str(exc),trace=traceback.format_exc());dump(OUT/'failures'/f'{sid}.json',result)
                done[result['status']]+=1;event('target_done',**result)
                row=next(iterator,None)
                if row:active[executor.submit(worker,row,runtime,protocol,freeze,not force_fresh)]=row['sample_id']
            progress()
    valid=export(manifest,protocol)
    dump(OUT/'progress.json',dict(status='complete' if valid==7800 else 'partial_batch' if limit else 'pending_repairs',stage='fine_annotation',planned=7800,successful=valid,pending=7800-valid,workers=workers,states=dict(done),updated_at=fresh.now(),pid=os.getpid()))
    event('batch_finished',valid=valid,expected=7800,limit=limit,elapsed_seconds=time.monotonic()-started,states=dict(done))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','annotate','export','all'),default='all');parser.add_argument('--workers',type=int,default=2);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--force-fresh',action='store_true');args=parser.parse_args()
    assert 1<=args.workers<=4
    manifest,runtime,protocol,freeze=prepare()
    if args.phase in ('annotate','all'):run(manifest,runtime,protocol,freeze,args.workers,args.limit,args.force_fresh)
    elif args.phase=='export':export(manifest,protocol)
