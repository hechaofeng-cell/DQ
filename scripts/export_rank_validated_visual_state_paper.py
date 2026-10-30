#!/usr/bin/env python3
"""Reuse the manuscript exporter, keeping the preceding Word export intact."""
import export_integrated_visual_state_paper as exporter
from analyze_ranking_attribute_validation import OUT, ROOT


if __name__ == '__main__':
    exporter.PAPER = ROOT / 'docs/visual_state_testing_empirical_paper_rank_validated_20261005.md'
    exporter.OUT = OUT
    exporter.main()
