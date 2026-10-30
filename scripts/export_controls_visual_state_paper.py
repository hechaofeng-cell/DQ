#!/usr/bin/env python3
"""Export the new controls manuscript without touching earlier Word files."""
from pathlib import Path
import json
import export_standard_visual_state_paper as exporter

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/random_stratification_ablation_20261006'
PAPER=ROOT/'docs/visual_state_testing_paper_controls_20261006.md'

if __name__=='__main__':
    exporter.SOURCE=PAPER
    exporter.OUTPUT=PAPER.with_suffix('.docx')
    exporter.build()
    (BASE/'word_export.json').write_text(json.dumps(dict(markdown=str(PAPER),word=str(PAPER.with_suffix('.docx'))),ensure_ascii=False,indent=2)+'\n')
    print(str(PAPER.with_suffix('.docx')))
