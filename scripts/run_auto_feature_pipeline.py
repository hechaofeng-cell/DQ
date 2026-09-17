#!/usr/bin/env python3
"""Run automatic category discovery, schema generation, and feature annotation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qe_quality.auto_feature.pipeline import main


if __name__ == "__main__":
    main()
