#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hcf/project/QE
PYTHON="$ROOT/.venv/bin/python"
PIPELINE="$ROOT/scripts/run_dog_local_pipeline.py"
SCHEMA="$ROOT/configs/dog_feature_schema_v3_1.json"
PROMPT="$ROOT/prompts/dog_feature_annotation_v3_1.txt"
DATA="$ROOT/data/openimages_v7_dog_gap_v1"
MODEL=qwen3-vl:30b-a3b-instruct

"$PYTHON" -u "$PIPELINE" \
  --input "$DATA/dog_feature_base_manifest.csv" \
  --schema "$SCHEMA" \
  --feature-prompt "$PROMPT" \
  --output "$ROOT/artifacts/openimages_v7_dog_base_features_v1" \
  --stages features \
  --model "$MODEL" \
  --num-predict 2200 \
  --resume

"$PYTHON" -u "$PIPELINE" \
  --input "$DATA/dog_feature_candidate_manifest.csv" \
  --schema "$SCHEMA" \
  --feature-prompt "$PROMPT" \
  --output "$ROOT/artifacts/openimages_v7_dog_candidate_features_v1" \
  --stages features \
  --model "$MODEL" \
  --num-predict 2200 \
  --resume
