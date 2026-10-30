#!/usr/bin/env python3
"""Export the new paper, keeping its coverage equation editable in Word."""
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from export_empirical_paper_docx import export

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/visual_state_coverage_difficulty_stratified_paper_20261004.md'

def run(text):
    r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r

def sub(letter,index):
    node=OxmlElement('m:sSub');body=OxmlElement('m:e');body.append(run(letter))
    tail=OxmlElement('m:sub');tail.append(run(index));node.extend([body,tail]);return node

def frac(top,bottom):
    node=OxmlElement('m:f');a=OxmlElement('m:num');b=OxmlElement('m:den')
    a.extend(top);b.extend(bottom);node.extend([a,b]);return node

def sum_term(index,body):
    node=OxmlElement('m:nary');props=OxmlElement('m:naryPr')
    char=OxmlElement('m:chr');char.set(qn('m:val'),'∑')
    hide=OxmlElement('m:supHide');hide.set(qn('m:val'),'1');props.extend([char,hide])
    tail=OxmlElement('m:sub');tail.extend(index);upper=OxmlElement('m:sup')
    val=OxmlElement('m:e');val.extend(body);node.extend([props,tail,upper,val]);return node

def main():
    target=PAPER.with_suffix('.docx');export(PAPER,target);document=Document(target)
    equations=[p for p in document.paragraphs if p.text.startswith('$$ C_c(S)')]
    assert len(equations)==1
    p=equations[0];p.clear()
    equation=OxmlElement('m:oMathPara');math=OxmlElement('m:oMath')
    math.extend([sub('C','c'),run('(S) = '),frac([run('1')],[run('|'),sub('U','c'),run('|')]),
                 sum_term([run('s∈'),sub('U','c')],[run('min('),frac([sub('n','s'),run('(S)')],[run('20')]),run(', 1)')]),
                 run('， C(S) = '),frac([run('1')],[run('6')]),sum_term([run('c')],[sub('C','c'),run('(S)')])])
    equation.append(math);p._p.append(equation);document.save(target)
    print(target)

if __name__=='__main__':main()
