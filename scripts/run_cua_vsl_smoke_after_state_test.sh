#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hcf/project/QE
PYTHON="$ROOT/.venv/bin/python"
OUTPUT="$ROOT/artifacts/openimages_v7_cua_vsl_v1"
LOG="$OUTPUT/background/smoke_after_state_test.log"

exec >"$LOG" 2>&1

echo "[$(date -Is)] waiting for cua_vsl_state_test"
while tmux has-session -t cua_vsl_state_test 2>/dev/null; do
  sleep 10
done

REPORT="$OUTPUT/state_test_features_auto/report.json"
if [[ ! -f "$REPORT" ]]; then
  echo "[$(date -Is)] state-test VLM ended without report.json"
  exit 1
fi

"$PYTHON" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("planned") != 200 or report.get("feature_ok") != 200 or report.get("errors") != 0:
    raise SystemExit(f"state-test VLM did not pass strict 200/200 gate: {report}")
print("state-test VLM strict gate passed: 200/200")
PY

cd "$ROOT"
echo "[$(date -Is)] running ResNet-18 smoke E0/E5/E8"
"$PYTHON" -u scripts/run_openimages_v7_cua_vsl.py --smoke --workers 12

echo "[$(date -Is)] aggregating smoke diagnostics"
"$PYTHON" -u scripts/report_openimages_cua_vsl.py --run-group smoke_runs

echo "[$(date -Is)] smoke complete; formal pilot remains blocked pending discussion"
