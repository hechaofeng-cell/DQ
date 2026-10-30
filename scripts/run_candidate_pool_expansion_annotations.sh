#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT"
PYTHON="$SOURCE/.venv/bin/python"
OUTPUT="$ROOT/artifacts/candidate_pool_expansion_v2_20260928"
mkdir -p "$OUTPUT"

echo "[$(date --iso-8601=seconds)] cat_v1.0 annotation started"
"$PYTHON" -u "$ROOT/scripts/run_dog_local_pipeline.py" \
  --input "$OUTPUT/cat/inputs/cat_feature_expansion_manifest.csv" \
  --schema "$SOURCE/configs/cat_feature_schema_v1_0.json" \
  --feature-prompt "$SOURCE/prompts/cat_feature_annotation_v1_0.txt" \
  --output "$OUTPUT/cat/features_auto" \
  --stages features \
  --model qwen3-vl:30b-a3b-instruct \
  --num-predict 2200 \
  --semantic-retries 2 \
  --resume

"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["feature_ok"] == p["planned"] and p["errors"] == 0, p' \
  "$OUTPUT/cat/features_auto/report.json"

echo "[$(date --iso-8601=seconds)] head_face_v1.0 positive annotation started"
"$PYTHON" -u "$ROOT/scripts/annotate_head_face_v1.py" \
  --experiment "$OUTPUT/face_binary" \
  --model qwen3-vl:30b-a3b-instruct \
  --annotation-status automatic_vlm_structurally_validated \
  --resume

echo "[$(date --iso-8601=seconds)] all annotations finished"
