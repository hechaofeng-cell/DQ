"""Create honest, unfilled human review tasks; never substitute VLM labels as truth."""
import argparse
import base64
import json
from pathlib import Path
from collections import Counter
import numpy as np
from coverage_validation_common_20261018 import *

TEMPLATE=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>视觉状态盲审</title>
<style>body{font:16px system-ui;max-width:1150px;margin:25px auto;padding:15px;color:#203040}header,nav{position:sticky;top:0;background:white;padding:10px;border-bottom:1px solid #ccd}img{width:100%;max-width:1000px}section{padding:16px;border-bottom:1px solid #ddd}label{display:inline-block;margin:6px 15px 6px 0}button,input,select{font:inherit;padding:5px}small{display:block;color:#52606d;margin:6px 0}#status{color:#086}</style>
<h1>四类视觉状态独立盲审 <span id="role"></span></h1>
<p>只根据图像和字典判据填写。左侧原图红框指定目标，右侧为目标裁剪。看不清或无法判断时选择“未知”。本页不显示自动标签、模型预测或选样方法。人工结果保持为空，直到你实际填写。</p>
<header><label>标注者姓名/代号 <input id="reviewer" placeholder="必填"></label><button onclick="saveExport()">导出本人标注 JSON</button><span id="status"></span></header>
<nav><button onclick="go(-1)">上一目标</button><span id="pos"></span><button onclick="go(1)">下一目标</button><label>跳转 <input id="jump" type="number" min="1" style="width:75px"><button onclick="jumpTo()">转到</button></label></nav>
<h2 id="target"></h2><img id="evidence"><main id="fields"></main>
<script>
const DATA=__PAYLOAD__;const KEY='qe-review-'+DATA.version+'-'+DATA.role;
let store=JSON.parse(localStorage.getItem(KEY)||'{"reviewer":"","answers":{}}');let at=0;
document.getElementById('role').textContent=DATA.role;document.getElementById('reviewer').value=store.reviewer;
document.getElementById('reviewer').oninput=e=>{store.reviewer=e.target.value;persist()};
function persist(){localStorage.setItem(KEY,JSON.stringify(store));status()}
function status(){let n=0;for(const t of DATA.tasks){let a=store.answers[t.sample_id]||{};if(DATA.schemas[t.class_name].every(f=>a[f.feature_id]!==undefined))n++}document.getElementById('status').textContent=' 已完整填写 '+n+'/'+DATA.tasks.length+' 个目标'}
function render(){const t=DATA.tasks[at];document.getElementById('pos').textContent=' '+(at+1)+' / '+DATA.tasks.length+' ';document.getElementById('target').textContent=t.class_name+' · '+t.sample_id;document.getElementById('evidence').src=t.evidence;const box=document.getElementById('fields');box.replaceChildren();let a=store.answers[t.sample_id]||{};
for(const f of DATA.schemas[t.class_name]){const section=document.createElement('section'),h=document.createElement('b'),desc=document.createElement('p');h.textContent=f.name_zh||f.feature_id;desc.textContent=f.definition_zh;section.append(h,desc);const u=document.createElement('small');u.textContent=f.unknown_rule_zh;section.append(u);
if(f.value_type==='enum'){let select=document.createElement('select');let blank=document.createElement('option');blank.value='';blank.textContent='— 尚未标注 —';select.append(blank);for(const v of f.values){let option=document.createElement('option');option.value=v.value_id;option.textContent=v.label_zh+'：'+v.criterion_zh;select.append(option)}select.value=a[f.feature_id]||'';select.onchange=()=>{store.answers[t.sample_id]??={};if(select.value)store.answers[t.sample_id][f.feature_id]=select.value;else delete store.answers[t.sample_id][f.feature_id];persist()};section.append(select)}
else{for(const v of f.values){let label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=v.value_id;input.checked=(a[f.feature_id]||[]).includes(v.value_id);input.onchange=()=>{store.answers[t.sample_id]??={};let values=new Set(store.answers[t.sample_id][f.feature_id]||[]);if(input.checked){if((f.exclusive_values||['unknown']).includes(v.value_id))values.clear();else for(const x of (f.exclusive_values||['unknown']))values.delete(x);values.add(v.value_id)}else values.delete(v.value_id);if(values.size)store.answers[t.sample_id][f.feature_id]=[...values];else delete store.answers[t.sample_id][f.feature_id];persist();render()};label.append(input,document.createTextNode(v.label_zh));label.title=v.criterion_zh;section.append(label);let rule=document.createElement('small');rule.textContent=v.label_zh+'：'+v.criterion_zh;section.append(rule)}}box.append(section)}status()}
function go(d){at=Math.max(0,Math.min(DATA.tasks.length-1,at+d));render()}function jumpTo(){at=Math.max(0,Math.min(DATA.tasks.length-1,Number(document.getElementById('jump').value)-1));render()}
function saveExport(){if(!store.reviewer.trim()){alert('请填写本人姓名或代号');return}if(!confirm('确认：导出的内容由本人查看图像后独立填写，没有直接采用自动标注作为参考真值。'))return;let payload={version:DATA.version,role:DATA.role,reviewer:store.reviewer,human_attestation:true,exported_at:new Date().toISOString(),answers:store.answers};let url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));let link=document.createElement('a');link.href=url;link.download='coverage_human_'+DATA.role+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}render();
</script></html>'''


def prepare():
    folder=OUT/'human_review';folder.mkdir(exist_ok=True)
    rows=read(OUT/'human_review_manifest.csv');schemas={c:json.loads((OUT/'dictionaries'/f'{c}.json').read_text())['features'] for c in CLASSES}
    tasks=[]
    for r in rows:
        ev=SOURCE/'annotations_v2'/r['class_name']/f'{r["sample_id"]}_evidence.jpg'
        tasks.append(dict(sample_id=r['sample_id'],class_name=r['class_name'],evidence='data:image/jpeg;base64,'+base64.b64encode(ev.read_bytes()).decode()))
    rng=np.random.default_rng(20261018)
    for role in ['A','B']:
        payload=dict(version='coverage-diagnostic-validation-v1-20261018',role=role,schemas=schemas,tasks=[tasks[i] for i in rng.permutation(len(tasks))])
        (folder/f'blind_review_{role}.html').write_text(TEMPLATE.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
    blank=[dict(sample_id=r['sample_id'],class_name=r['class_name'],field=f['feature_id'],human_value='',reviewer='',notes='') for r in rows for f in schemas[r['class_name']]]
    for role in ['A','B']:write(folder/f'annotator_{role}_blank.csv',blank)
    dump(folder/'status.json',dict(status='pending_actual_human_annotations',targets=len(rows),fields_per_annotator=len(blank),reviewers=2,
        completed=0,automatic_defaults=False,human_accuracy_computed=False))
    (folder/'使用说明.md').write_text('''# 独立人工标注待完成

打开 blind_review_A.html 和 blind_review_B.html，由两位实际标注者分别完成。字段全部为空；可以分批填写，浏览器在本地保存进度。完成后导出 JSON。不要把自动结果或第二个模型的结果填成人工答案。

`python3 scripts/prepare_coverage_blind_review_20261018.py --a /绝对路径/A.json --b /绝对路径/B.json`

导入器仅对两人均完成且一致的字段计算自动标注与一致参考的对照，分歧字段单独导出，不能把一致子集准确率称为总体准确率。人工样本为每类每Q随机16个，共320个，属于分层样本，不按自动预测结果筛选。导入后需结合分歧处理进一步分析。
''')
    log('human_review_prepared',targets=len(rows),fields=len(blank),human_labels_filled=0)


def analyze(a_path,b_path):
    from annotate_coverage_validation_20261018 import VALIDATE
    aa=json.loads(Path(a_path).read_text());bb=json.loads(Path(b_path).read_text())
    assert aa['role']=='A' and bb['role']=='B' and aa['reviewer'].strip()!=bb['reviewer'].strip()
    assert aa['human_attestation'] is True and bb['human_attestation'] is True
    assert aa['version']==bb['version']=='coverage-diagnostic-validation-v1-20261018'
    schemas={c:json.loads((OUT/'dictionaries'/f'{c}.json').read_text())['features'] for c in CLASSES}
    rows=read(OUT/'human_review_manifest.csv');results=[];disagreements=[]
    for r in rows:
        sid=r['sample_id'];c=r['class_name'];a=aa['answers'].get(sid,{});b=bb['answers'].get(sid,{})
        # Partial exports remain partial; no value is silently filled.
        auto=json.loads((SOURCE/'annotations_v2'/c/f'{sid}_parsed.json').read_text())['features']
        for f in schemas[c]:
            key=f['feature_id']
            if key not in a or key not in b:continue
            VALIDATE({key:a[key]},[f]);VALIDATE({key:b[key]},[f])
            as_set=lambda v:set(v) if isinstance(v,list) else {v}
            av,bv,mv=as_set(a[key]),as_set(b[key]),as_set(auto[key])
            if av!=bv:
                disagreements.append(dict(sample_id=sid,class_name=c,field=key,A=json.dumps(a[key],ensure_ascii=False),B=json.dumps(b[key],ensure_ascii=False),adjudicated_value=''));continue
            record=dict(sample_id=sid,class_name=c,Q=r['Q'],field=key,value_type=f['value_type'],exact_match=int(mv==av),
                human_unknown=int(av=={'unknown'}),auto_unknown=int(mv=={'unknown'}),reference_value=json.dumps(a[key],ensure_ascii=False),
                model_value=json.dumps(auto[key],ensure_ascii=False),set_tp=len((av-{'unknown'})&(mv-{'unknown'})),
                set_fp=len((mv-{'unknown'})-(av-{'unknown'})),set_fn=len((av-{'unknown'})-(mv-{'unknown'})))
            results.append(record)
    folder=OUT/'human_review'
    if results:write(folder/'consensus_comparison.csv',results)
    if disagreements:write(folder/'disagreements_for_adjudication.csv',disagreements)
    dump(folder/'import_status.json',dict(consensus_fields=len(results),disagreement_fields=len(disagreements),
        consensus_subset_exact_match_pct=float(100*np.mean([r['exact_match'] for r in results])) if results else None,
        coverage='consensus-completed fields only; not full-population accuracy',inputs={str(a_path):sha(a_path),str(b_path):sha(b_path)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a');p.add_argument('--b');args=p.parse_args()
    if args.a and args.b:analyze(args.a,args.b)
    elif args.a or args.b:p.error('Both human exports required')
    else:prepare()
