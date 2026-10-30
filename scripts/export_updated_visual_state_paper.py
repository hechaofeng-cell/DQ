#!/usr/bin/env python3
"""Export only the new integrated manuscript, preserving all preceding exports."""
from pathlib import Path
import export_integrated_visual_state_paper as exporter

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    exporter.PAPER=ROOT/'docs/visual_state_testing_empirical_paper_updated_20261006.md'
    exporter.OUT=ROOT/'artifacts/visual_state_paper_updated_20261006'
    exporter.OUT.mkdir(parents=True,exist_ok=True)
    exporter.main()
