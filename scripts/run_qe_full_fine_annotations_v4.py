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
OUT=ROOT/'artifacts/QE_FullFine_v4_20261013'
OLD=ROOT/'artifacts/fresh_vlm_feature_scoring_20261012'
CLASSES=fresh.CLASSES
PROMPT_V4=fresh.PROMPT+"""
补充判定规则（统一版本v4，必须遵守）：
原图只拍到头部等近景时，原图范围外的躯干和肢体属于取景截断造成不可见，torso_visibility或limb_visibility必须not_visible（确实无法判断则unknown），绝不能not_in_view。非车辆的躯干、肢体不能用not_in_view规避截断、遮挡、低清。truncation_parts必须列出原图边界切掉或原图外缺失的相关部分；身体大部分不在原图时，不仅标注头部。
blur、远距离小尺度、逆光、低对比、过曝导致部位细节难辨时，不能标为“被物体遮挡”。可辨部位轮廓但细节不足可用partial或unknown，证据说明细节不足；不可凭像素噪声推断部位消失。occlusion_level的light/partial/moderate/heavy只能由可见且确实覆盖目标的物体支持，证据明确物体与部位。没有可见遮挡物时不要编造other_object遮挡；无法判断是否被挡时用unknown，清楚没有遮挡物时用none。舞台灯光、模糊、阴影、物体自身颜色不属于遮挡物。
正常视角不呈现的not_in_view仅限字典允许且不由原图边界/遮挡/成像不清造成的背侧部件，例如正面车图车尾。非车torso_visibility和limb_visibility一律不使用not_in_view。
根据原图判断截断；仅目标裁剪紧贴边缘不构成原图截断。未知区域的朝向、动作和细节必须unknown，不能为了填满字段而猜测。
每目标将附原图宽高、官方像素框、原图四边接触事实。若官方目标框四边均位于原图内部（距离原图边界超过2像素），该目标没有原图边界截断：truncation_level必须none，truncation_parts（若存在）必须[none]。目标裁剪边缘、背景烟雾、低清、遮挡不属于原图边界。只有官方框接触原图边界，才能结合原图证据判断是否真正截断；接触本身并不证明截断。
不能从灯光、散射、低清或烟雾脑补不存在的不透明前景物体。有明确不透明遮挡物才填实际遮挡程度；模糊/光照让遮挡无法可靠区分时填unknown并说明无法确认，绝不能编造heavy和other_object。
以下轮廓原则优先于把细节当缺失的判断：可辨认头部、躯干、肢体轮廓但无法辨认细节时，部位可见性应partial或unknown，不能仅因模糊/灯光/远距离标not_visible。not_visible专指该部位适用但实际不在原图/被不透明物体完全挡住；说明缺失原因。
遮挡必须有能具体命名的不透明实体遮挡物和明确覆盖部位，例如栅栏遮腿、前景人挡躯干；没有明确实体或位置且轮廓低清时occlusion_level用unknown。舞台光、灯光、雾、烟雾、模糊、逆光、阴影不在本协议的物体遮挡定义中，不能用这些因素及猜测前景物体证明heavy/moderate/light/partial。不要套用“舞台灯光、烟雾和前景物体”的模板。
"""
BLUR_WORDS=('模糊','低清','光照','灯光','舞台光','烟雾','烟','雾','逆光','阴影','过曝','blur','lighting','smoke','fog','backlight','shadow')
LOSS_WORDS=('原图外','原图边界','画面外','截断','遮挡','挡住','被挡','图像外','边界之外','边缘之外','outside','truncat','occlud','blocked')

def validate_v4(value,row,schema):
    fresh.validate_annotation(value,row,schema)
    if row['class_name']!='car':
        for key in ('torso_visibility','limb_visibility'):
            if value['features'][key]=='not_in_view':raise ValueError('v4noncar '+key+' cannot be not_in_view; frame/crop loss is not_visible or unknown')
    level=value['features']['occlusion_level']
    evidence=value['evidence']['occlusion_level'].lower()
    if level not in ('none','unknown') and any(word in evidence for word in BLUR_WORDS):
        raise ValueError('v4 positive occlusion evidence cites blur/light/smoke rather than a concrete opaque occluder and location. Do not invent foreground objects. Use unknown if no independently visible opaque blocker; never raw lighting/smoke as heavy occlusion.')
    if row['class_name']!='car':
        for key in ('head_visibility','torso_visibility','limb_visibility'):
            observation=value['evidence'][key].lower()
            if value['features'][key]=='not_visible' and any(word in observation for word in BLUR_WORDS):
                raise ValueError('v4 '+key+' not_visible supported by blur/light/smoke. Recognizable silhouette with insufficient detail is partial or unknown; actual full frame loss/opaque occlusion must be independently supported. Reobserve original image.')
    interior,_=frame_facts(row)
    if interior:
        if value['features']['truncation_level']!='none':raise ValueError('v4 original official bbox is wholly interior beyond2px: truncation_level must none; crop edge/blur is not original frame truncation')
        if 'truncation_parts' in value['features'] and value['features']['truncation_parts']!=['none']:raise ValueError('v4 interior original bbox: truncation_parts must [none]')
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
    immutable(OUT/'prompt.txt',PROMPT_V4,False)
    protocol=fresh.scoring_protocol();immutable(OUT/'scoring_protocol.json',protocol)
    immutable(OUT/'protocol.json',dict(targets=7800,per_class=1300,local_inference_only=True,model=runtime['model'],schemas='original fresh26-35 fields; all23dog/cat fields preserved',
        views='packaged original, marked target, feature crop; bound to original official box and target_1alias',
        reuse='no old prompt annotations reused; only samev4 validated target checkpoints with exactschema/prompt/viewSHA are resumed',
        five_dimensional_scoring='original fresh_score: originalbbox scale; noncar head/torso/limb visibility; car8parts excludingnotinview; body+headorientation; occlusion/truncation20each',
        nominal='all described, no invented ordinal hair/color/texture risk weights',
        predictions_read=False,semantic_retries=3,resumable=True,prompt_version='v4_silhouette_opaque_evidence_gate',frame_gate='official_bbox interior >2px -> truncation none and partsnone; original source bbox rounding tolerance3e-6',semantic_gate='conservative keyword evidence rule: positive occlusion may not cite blur/light/smoke; noncar not_visible may not be justified by these. Retry actualVLM, no label rewriting. Imperfect rule not semantic ground truth',scoring_scope='interim frozen fine-v1 scores; not allfinefields individually in score; parent may establish separately versioned extension',
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
        value=validate_v4(json.loads(dest.read_text()),row,schema)
        assert value['annotation_prompt_version']=='v4_silhouette_opaque_evidence_gate'
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
    prompt=PROMPT_V4+'\n冻结特征字典：\n'+json.dumps(dict(schema_version=schema['schema_version'],category=row['class_name'],fields=schema['universal_features']),ensure_ascii=False)
    interior,facts=frame_facts(row)
    prompt+='\n此目标的原图几何事实（来自官方目标框，非裁剪边界）：'+json.dumps(dict(target_id=sid,wholly_interior_to_original_frame=interior,**facts),ensure_ascii=False)
    errors=[];started=time.monotonic()
    for attempt in range(1,4):
        try:
            correction='' if not errors else '\n上一份输出校验错误：'+errors[-1]['error']+'。重新输出全部字段和证据。'
            value=ask(runtime,prompt+correction,dict(image_id=sid),[Path(row[k]) for k in ('image_path','marked_image_path','feature_crop_path')],OUT/'raw'/f'{sid}_{run_id}_semantic{attempt}.json',feature_schema(schema_path))
            validate_v4(value,row,schema)
            value.update(annotation_prompt_version='v4_silhouette_opaque_evidence_gate',original_frame_geometry=facts,annotation_model=runtime['model'],annotated_at=fresh.now(),schema_sha256=fresh.pilot.sha256(schema_path),prompt_sha256=fresh.pilot.sha256(OUT/'prompt.txt'),input_view_sha256={k:freeze[row[k]] for k in ('image_path','marked_image_path','feature_crop_path')})
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
        schema=json.loads((OUT/'schemas'/f"{row['class_name']}.json").read_text());value=validate_v4(json.loads(path.read_text()),row,schema)
        assert value['annotation_prompt_version']=='v4_silhouette_opaque_evidence_gate' and value['prompt_sha256']==fresh.pilot.sha256(OUT/'prompt.txt')
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
