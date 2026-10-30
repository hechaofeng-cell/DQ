#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT"
PYTHON="$SOURCE/.venv/bin/python"
OUTPUT="$ROOT/artifacts/multiclass_object_pool_v1_20260928"

echo "[$(date --iso-8601=seconds)] object_state_v1.0 annotation started"
"$PYTHON" -u "$ROOT/scripts/run_dog_local_pipeline.py" \
  --input "$OUTPUT/inputs/object_state_feature_manifest.csv" \
  --schema "$ROOT/configs/multiclass_object_state_schema_v1_0.json" \
  --feature-prompt "$ROOT/prompts/feature_annotation_multiclass_object_v1_0.txt" \
  --output "$OUTPUT/annotations" \
  --stages features \
  --model qwen3-vl:30b-a3b-instruct \
  --num-predict 2600 \
  --semantic-retries 2 \
  --resume

"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["feature_ok"] == p["planned"] and p["errors"] == 0, p' \
  "$OUTPUT/annotations/report.json"

echo "[$(date --iso-8601=seconds)] object_state_v1.0 annotation finished"
