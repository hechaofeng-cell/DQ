#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hcf/project/QE
PYTHON="$ROOT/.venv/bin/python"
OUTPUT="$ROOT/artifacts/openimages_v7_cua_vsl_v1"
LOG="$OUTPUT/background/stage1_followup.log"

exec >"$LOG" 2>&1

echo "[$(date -Is)] waiting for cua_vsl_oof"
while tmux has-session -t cua_vsl_oof 2>/dev/null; do
  sleep 10
done

if [[ ! -f "$OUTPUT/state_priority/state_priority.csv" ]]; then
  echo "[$(date -Is)] OOF task ended without state_priority.csv"
  exit 1
fi

echo "[$(date -Is)] selecting random, gap, and CUA supplements"
cd "$ROOT"
"$PYTHON" -u scripts/select_openimages_cua_supplement.py

echo "[$(date -Is)] running automatic dog_v3.1 on 200 frozen test dogs"
"$PYTHON" -u scripts/run_dog_local_pipeline.py \
  --input "$OUTPUT/state_test_inputs/dog_feature_test_manifest.csv" \
  --schema "$ROOT/configs/dog_feature_schema_v3_1.json" \
  --feature-prompt "$ROOT/prompts/dog_feature_annotation_v3_1.txt" \
  --output "$OUTPUT/state_test_features_auto" \
  --stages features \
  --model qwen3-vl:30b-a3b-instruct \
  --num-predict 2200 \
  --resume

echo "[$(date -Is)] automatic state annotation complete"
echo "[$(date -Is)] PAUSED: CUA300 currently duplicates gap300; smoke and formal pilots require discussion"
