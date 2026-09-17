#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/hcf/project/QE
PYTHON="$ROOT/.venv/bin/python"
SCHEMA="$ROOT/configs/dog_feature_schema_v3_1.json"
PROMPT="$ROOT/prompts/dog_feature_annotation_v3_1.txt"
MODEL=qwen3-vl:30b-a3b-instruct

mkdir -p "$ROOT/artifacts/dog500_features_v3_1_20260913" "$ROOT/artifacts/dog_candidate400_features_v3_1_20260913"

"$PYTHON" -u "$ROOT/scripts/run_dog_local_pipeline.py" \
  --input "$ROOT/artifacts/dog_feature_pipeline_v2_1_inputs_20260912/target_manifest.csv" \
  --schema "$SCHEMA" --feature-prompt "$PROMPT" \
  --output "$ROOT/artifacts/dog500_features_v3_1_20260913" \
  --stages features --model "$MODEL" --num-predict 2200 --resume

"$PYTHON" -u "$ROOT/scripts/run_dog_local_pipeline.py" \
  --input "$ROOT/artifacts/dog_candidate400_v2_1_inputs_20260912/target_manifest.csv" \
  --schema "$SCHEMA" --feature-prompt "$PROMPT" \
  --output "$ROOT/artifacts/dog_candidate400_features_v3_1_20260913" \
  --stages features --model "$MODEL" --num-predict 2200 --resume
