#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hcf/project/QE
PYTHON="$ROOT/.venv/bin/python"
BASE="$ROOT/artifacts/openimages_v7_dog_base_features_v1"
CANDIDATE="$ROOT/artifacts/openimages_v7_dog_candidate_features_v1"

while ! "$PYTHON" -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
s = json.loads(p.read_text(encoding="utf-8"))
sys.exit(0 if s.get("status") == "complete" and s.get("completed_features") == 3000 else 1)
' "$CANDIDATE/pipeline_status.json"; do
  sleep 30
done

"$PYTHON" -c '
import json, sys
from pathlib import Path
for name, count in (("base", 1500), ("candidate", 3000)):
    root = Path(sys.argv[1] if name == "base" else sys.argv[2])
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["planned"] != count or report["feature_ok"] != count or report["errors"]:
        raise SystemExit(f"{name} feature annotation incomplete: {report}")
' "$BASE" "$CANDIDATE"

# Feature inference has finished; release the resident VLM before GPU training.
ollama stop qwen3-vl:30b-a3b-instruct

"$PYTHON" -u "$ROOT/scripts/select_openimages_dog_supplements.py" \
  --base-features "$BASE/results.csv" \
  --candidate-features "$CANDIDATE/results.csv"

"$PYTHON" -u "$ROOT/scripts/run_openimages_v7_four_model_experiment.py"
"$PYTHON" -u "$ROOT/scripts/report_openimages_v7_four_model_experiment.py"
