#!/usr/bin/env python3
"""Export the integrated manuscript with editable Word display equations."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from export_empirical_paper_docx import export
from export_sixclass_stratified_paper import run, sub, frac, sum_term

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/visual_state_testing_empirical_paper_integrated_20261005.md'
OUT = ROOT / 'artifacts/label_disagreement_sensitivity_20261005'


def bounded_sum(lower, upper, body):
    node = sum_term(lower, body)
    node.find(qn('m:naryPr')).remove(node.find('.//' + qn('m:supHide')))
    node.find(qn('m:sup')).extend(upper)
    return node


def binomial(top, bottom):
    quotient = frac(top, bottom)
    props = OxmlElement('m:fPr')
    kind = OxmlElement('m:type')
    kind.set(qn('m:val'), 'noBar')
    props.append(kind)
    quotient.insert(0, props)
    delimiter = OxmlElement('m:d')
    expr = OxmlElement('m:e')
    expr.append(quotient)
    delimiter.append(expr)
    return delimiter


def replace_equation(paragraph, nodes):
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Pt(0)
    outer = OxmlElement('m:oMathPara')
    inner = OxmlElement('m:oMath')
    inner.extend(nodes)
    outer.append(inner)
    paragraph._p.append(outer)


def main():
    target = PAPER.with_suffix('.docx')
    export(PAPER, target)
    document = Document(target)
    equations = [p for p in document.paragraphs if p.text.startswith('$$ ')]
    assert len(equations) == 3, 'Unexpected manuscript display equations'
    assert equations[0].text.startswith('$$ C_c(S)')
    assert equations[1].text.startswith('$$ g(x')
    assert equations[2].text.startswith('$$ 1-')
    replace_equation(equations[0], [
        sub('C', 'c'), run('(S) = '),
        frac([run('1')], [run('|'), sub('U', 'c'), run('|')]),
        sum_term([run('s∈'), sub('U', 'c')], [
            run('min('), frac([sub('n', 's'), run('(S)')], [run('τ')]), run(', 1)')]),
        run('， C(S) = '), frac([run('1')], [run('6')]),
        bounded_sum([run('c=1')], [run('6')], [sub('C', 'c'), run('(S)')]),
        run('， τ = 20')])
    replace_equation(equations[1], [
        run('g(x | S) = '),
        sum_term([run('s∈A(x), '), sub('n', 's'), run('(S)<20')], [
            run('[log('), sub('n', 's'), run('(S)+2) − log('),
            sub('n', 's'), run('(S)+1)]')])])
    replace_equation(equations[2], [run('1 − '), frac(
        [binomial([run('N−r')], [run('q')])],
        [binomial([run('N')], [run('q')])])])
    document.save(target)

    source = PAPER.read_text(encoding='utf-8')
    expected_figures = len(re.findall(r'^!\[', source, flags=re.M))
    expected_tables = len(re.findall(r'^\|[^\n]+\n\|[ :|\-]+\|\s*$', source, flags=re.M))
    with ZipFile(target) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
              'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
              'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
        counts = {'editable_display_equations': len(root.findall('.//m:oMath', ns)),
                  'embedded_figures': len(root.findall('.//a:blip', ns)),
                  'tables': len(root.findall('.//w:tbl', ns))}
        assert counts['editable_display_equations'] == 3
        assert counts['embedded_figures'] == expected_figures
        assert counts['tables'] == expected_tables
        assert not any('$$' in (node.text or '') for node in root.findall('.//w:t', ns))
    audit = {'status': 'pass', 'markdown': str(PAPER), 'word': str(target), **counts,
             'scope': 'structural export audit; no claim of rendered page-layout validation'}
    (OUT / 'word_export_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
