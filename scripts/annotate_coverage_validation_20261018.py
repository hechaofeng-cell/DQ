"""New external annotations and repeatability checks, original frozen schema/prompt."""
import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import requests
from PIL import Image, ImageDraw, ImageOps
from coverage_validation_common_20261018 import *


def original_functions():
    path=ROOT/'scripts/annotate_other_classes_specific_v2_20261018.py'
    tree=ast.parse(path.read_text())
    keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('specification','validate')]
    assert len(keep)==2
    namespace={'json':json}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['specification'],namespace['validate']


SPEC,VALIDATE=original_functions()


def prepare():
    contexts=load_contexts(); rows=read(EXTERNAL/'manifests/candidate_pool.csv')
    source_rows=manifest(); hashes={r['image_sha256'] for r in source_rows}
    assert not hashes & {r['image_sha256'] for r in rows}, 'Source/external image overlap'
    chosen=[]
    for c in CLASSES:
        values=[r for r in rows if r['class_name']==c]
        chosen.extend(sorted(values,key=lambda r:hashlib.sha256(('20261018|'+r['sample_id']).encode()).hexdigest())[:200])
    external_path=OUT/'external_manifest.csv'
    if external_path.exists(): assert read(external_path)==chosen
    else:write(external_path,chosen)
    audit=[];repeat=[]
    for c,ctx in contexts.items():
        for q in range(1,6):
            values=[r for i,r in enumerate(ctx['rows']) if ctx['tiers'][i]==q]
            values=sorted(values,key=lambda r:hashlib.sha256(('audit20261018|'+r['sample_id']).encode()).hexdigest())
            audit.extend(dict(r,Q=str(q)) for r in values[:16])
            repeat.extend(dict(r,Q=str(q)) for r in values[:4])
    for name,rr in [('human_review_manifest.csv',audit),('repeat_manifest.csv',repeat)]:
        path=OUT/name
        if path.exists():assert read(path)==rr
        else:write(path,rr)
    inputs=[OUT/'protocol.json',external_path,OUT/'human_review_manifest.csv',OUT/'repeat_manifest.csv',
            ROOT/'scripts/annotate_other_classes_specific_v2_20261018.py',BASE/'manifest.csv',EXTERNAL/'manifests/candidate_pool.csv']
    for c in CLASSES:
        src=SOURCE/'frozen_dictionaries'/f'{c}.json'; dst=OUT/'dictionaries'/f'{c}.json'
        freeze(dst,json.loads(src.read_text()));inputs.append(src)
    freeze(OUT/'annotation_input_freeze_v2.json',{str(p):sha(p) for p in inputs})
    # Verify overlap with historical train/eval manifests where hashes are available.
    overlaps=[]
    for p in [ROOT/'artifacts/sixclass_independent_confirmation_20261004/manifests'/n for n in ('train.csv','eval.csv')]:
        rr=read(p); present={x.get('image_sha256') for x in rr if x.get('image_sha256')}
        overlaps.append(dict(path=str(p),rows=len(rr),hashes_available=len(present),external_overlap=len(present & {x['image_sha256'] for x in chosen})))
    assert not any(x['external_overlap'] for x in overlaps)
    dump(OUT/'external_identity_audit.json',dict(source_external_exact_hash_overlap=0,external_targets=len(chosen),
        external_unique_images=len({r['image_sha256'] for r in chosen}),train_eval=overlaps,
        near_duplicates='Existing external-pool audit retained; this run checks exact hashes only'))
    return chosen,repeat


def evidence(row):
    assert sha(row['image_path'])==row['image_sha256'] and sha(row['crop_path'])==row['crop_sha256']
    im=Image.open(row['image_path']).convert('RGB');box=json.loads(row['bbox_norm_xyxy']);w,h=im.size
    ImageDraw.Draw(im).rectangle((box[0]*w,box[1]*h,box[2]*w,box[3]*h),outline='red',width=3)
    canvas=Image.new('RGB',(640,340),'white')
    canvas.paste(ImageOps.pad(im,(320,310),color='white'),(0,28))
    canvas.paste(ImageOps.pad(Image.open(row['crop_path']).convert('RGB'),(320,310),color='white'),(320,28))
    ImageDraw.Draw(canvas).text((8,7),f'0 | original + target crop | {row["sample_id"]}',fill='black')
    b=io.BytesIO();canvas.save(b,format='JPEG',quality=89);return b.getvalue()


def annotate(row,group):
    c=row['class_name'];sid=row['sample_id'];folder=OUT/'annotations'/group/c;folder.mkdir(parents=True,exist_ok=True)
    dictionary=OUT/'dictionaries'/f'{c}.json';fields=json.loads(dictionary.read_text())['features']
    prompt,schema=SPEC(c,fields);path=folder/f'{sid}_parsed.json'
    if path.exists():
        d=json.loads(path.read_text());VALIDATE(d['features'],fields)
        assert d['dictionary_sha256']==sha(dictionary) and d['original_sha256']==row['image_sha256']
        assert d['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
        return d
    ev=folder/f'{sid}_evidence.jpg'
    if group=='repeat':
        content=(SOURCE/'annotations_v2'/c/f'{sid}_evidence.jpg').read_bytes()
    else:content=evidence(row)
    ev.write_bytes(content)
    session=requests.Session();session.trust_env=False
    errors=[]
    for attempt in range(1,4):
        rawpath=folder/f'{sid}_attempt{attempt}_raw.json';start=time.monotonic()
        try:
            if rawpath.exists():raw=json.loads(rawpath.read_text())
            else:
                opts=dict(temperature=0,seed=(20261018 if group=='external' else 20261019)+attempt-1,num_ctx=16384,num_predict=2000)
                response=session.post('http://localhost:11434/api/chat',json=dict(model='qwen3-vl:30b-a3b-instruct',stream=False,keep_alive='30m',format=schema,
                    options=opts,messages=[dict(role='user',content=prompt,images=[base64.b64encode(content).decode()])]),timeout=300)
                response.raise_for_status();raw=response.json();dump(rawpath,raw)
            assert raw.get('done_reason')=='stop','generation truncated'
            features=json.loads(raw['message']['content']);VALIDATE(features,fields)
            result=dict(sample_id=sid,category=c,features=features,model='qwen3-vl:30b-a3b-instruct',attempt=attempt,
                group=group,elapsed_seconds=time.monotonic()-start,dictionary_sha256=sha(dictionary),evidence_sha256=sha(ev),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),original_sha256=row['image_sha256'],crop_sha256=row['crop_sha256'],
                human_verified=False)
            dump(path,result);return result
        except Exception as exc:
            errors.append(repr(exc));dump(folder/f'{sid}_attempt{attempt}_error.json',dict(error=repr(exc),time=now()))
    raise RuntimeError((sid,errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    external,repeat=prepare()
    for c in CLASSES:
        prompt,schema=SPEC(c,json.loads((OUT/'dictionaries'/f'{c}.json').read_text())['features'])
        (OUT/'dictionaries'/f'{c}_prompt.txt').write_text(prompt);freeze(OUT/'dictionaries'/f'{c}_response_schema.json',schema)
    jobs=[(r,'external') for r in external]+[(r,'repeat') for r in repeat]
    if args.smoke:jobs=[(next(r for r in external if r['class_name']==c),'external') for c in CLASSES]
    failures=[];completed=0;started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(annotate,r,g):(r['sample_id'],g) for r,g in jobs}
        for future in as_completed(futures):
            sid,group=futures[future]
            try:future.result();completed+=1
            except Exception as e:failures.append(dict(sample_id=sid,group=group,error=str(e)))
            if completed%20==0 or args.smoke or failures:
                d=dict(time=now(),completed=completed,total=len(jobs),failures=failures,elapsed_seconds=time.monotonic()-started)
                dump(OUT/('annotation_smoke_progress.json' if args.smoke else 'annotation_progress.json'),d)
                print(json.dumps({k:v for k,v in d.items() if k!='failures'},ensure_ascii=False),flush=True)
    dump(OUT/('annotation_smoke_audit.json' if args.smoke else 'annotation_audit.json'),dict(status='pass' if not failures else 'incomplete',completed=completed,total=len(jobs),failures=failures))
    if failures:raise RuntimeError(f'Unresolved annotation failures: {len(failures)}')


if __name__=='__main__':main()
