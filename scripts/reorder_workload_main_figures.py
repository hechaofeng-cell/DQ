#!/usr/bin/env python3
"""Place model-drop figures beside the corresponding work, preserving assets."""
from pathlib import Path
from copy import deepcopy
import re
import json
import hashlib
import shutil
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/工作量_含模型降幅多图版_20261008.docx'
OUT=ROOT/'artifacts/workload_main_integrated_20261008'
OUT.mkdir(parents=True,exist_ok=True)
NAME='工作量_正文整合版_20261008'
doc=Document(SOURCE)
body=doc.element.body
portrait=deepcopy(doc.sections[0]._sectPr)
landscape=deepcopy(doc.sections[1]._sectPr)
def text(e):return ''.join(e.xpath('.//w:t/text()'))
def replace(e,value):
    ts=e.xpath('.//w:t')
    if not ts:return
    ts[0].text=value
    for t in ts[1:]:t.text=''

# Divide the document at its existing page and section breaks.
pages=[[]]
for e in list(body):
    if e.tag==qn('w:sectPr'):continue
    if e.xpath('.//w:br[@w:type="page"]') or e.xpath('./w:pPr/w:sectPr'):
        pages.append([])
    else:pages[-1].append(e)
assert len(pages)==20
main=pages[:12];charts=pages[12:]
for e in main[0]:
    if text(e).startswith('新增：'):
        replace(e,'阅读顺序：分层结果与各模型降幅（第3—7页）→换来源核验（第8—10页）→随机与去项检查（第11—12页）→固定数量实验（第13—15页）→后续对照与汇报稿（第16—20页）。')
        break
else:raise AssertionError('Missing old locator')
for e in main[2]:
    if text(e).startswith('2  分组后'):
        replace(e,'2  分层后，各模型的表现降低多少？')
    elif text(e)=='分组完成后，下一步是计算每一组的表现。':
        replace(e,'本节先看五组平均表现，接着直接比较8种模型的层间降幅。主口径：第1组F1－第5组F1，单位是百分点。')
for e in main[3]:
    if text(e)=='原样本有差别，还要检查换图片、换来源能否再看到。':
        replace(e,'换图片、换来源检查差别能否重复；本节后两页展示各模型降幅总览与两套分组的对照。')
for e in main[6]:
    if text(e)=='避免把“图片更多”和“图片更难”混在一起。':
        replace(e,'固定数量后，先看平均变化，再看8种模型相对0%困难组起点的降幅曲线。此处起点来自第1—4组混合。')

chart_titles=[
    '2.1  各模型降低多少：本地分组',
    '2.2  各模型降低多少：在线分组',
    '2.3  五组中的降幅变化：本地分组',
    '2.4  五组中的降幅变化：在线分组',
    '3.1  各模型在三个来源的降幅总览',
    '3.2  本地与在线分组的降幅对照',
    '6.1  固定120个目标时的各模型降幅：本地分组',
    '6.2  固定120个目标时的各模型降幅：在线分组',
]
for elements,title in zip(charts,chart_titles):replace(elements[0],title)
order=[]
for i,elements in enumerate(main):
    order.append((elements,'portrait'))
    insert={2:[0,1,2,3],3:[4,5],6:[6,7]}.get(i,[])
    for j in insert:order.append((charts[j],'landscape'))
assert len(order)==20

# Number every figure in reading order. Each figure remains with its caption.
figure=0
for elements,orientation in order:
    after_image=False
    for e in elements:
        if e.xpath('.//w:drawing'):
            figure+=1;after_image=True
        elif after_image and e.tag==qn('w:p') and text(e).strip():
            old=text(e)
            stripped=re.sub(r'^图\s*\d+\s*[:：]?\s*','',old)
            replace(e,f'图{figure}  {stripped}')
            after_image=False
assert figure==19

for e in list(body):body.remove(e)
for i,(elements,orientation) in enumerate(order):
    for e in elements:body.append(e)
    if i==len(order)-1:break
    boundary=OxmlElement('w:p')
    if order[i+1][1]!=orientation:
        pr=OxmlElement('w:pPr')
        props=deepcopy(portrait if orientation=='portrait' else landscape)
        # A section's properties govern the content preceding this boundary.
        st=props.find(qn('w:type'))
        if st is None:st=OxmlElement('w:type');props.insert(0,st)
        st.set(qn('w:val'),'nextPage')
        pr.append(props);boundary.append(pr)
    else:
        run=OxmlElement('w:r');br=OxmlElement('w:br');br.set(qn('w:type'),'page')
        run.append(br);boundary.append(run)
    body.append(boundary)
body.append(deepcopy(portrait))
doc.core_properties.title='工作量：模型降幅图融入正文'
doc.core_properties.subject='模型降幅随分层结果、来源核验和固定数量实验展开'
doc.save(OUT/f'{NAME}.docx')
shutil.copy2(OUT/f'{NAME}.docx',ROOT/'docs'/f'{NAME}.docx')
assert len(doc.inline_shapes)==19
assert len(doc.sections)==7
md=[]
for page,(elements,orientation) in enumerate(order,1):
    md.append(f'\n<!-- 第{page}页：{orientation} -->\n')
    for e in elements:
        if e.tag==qn('w:tbl'):
            for row in e.xpath('./w:tr'):
                md.append('| '+' | '.join(text(c) for c in row.xpath('./w:tc'))+' |\n')
        elif text(e).strip():md.append(text(e)+'\n\n')
(ROOT/'docs'/f'{NAME}.md').write_text(''.join(md),encoding='utf-8')
audit={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       'expected_pages':20,'figures':19,'sections':7,'figure_numbers':'1–19',
       'model_endpoint_and_tier_pages':[4,5,6,7],'confirmation_comparison_pages':[9,10],
       'fixed120_model_pages':[14,15],'render_verified':False}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
shutil.copy2(Path(__file__),OUT/Path(__file__).name)
print(json.dumps({'word':str(ROOT/'docs'/f'{NAME}.docx'),'figures':19,'sections':7,'pages':20},ensure_ascii=False))
