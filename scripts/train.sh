#!/bin/bash
# Train DSRL on one DexJoCo task on one GPU.
#
# Usage: GPU=0 bash scripts/train.sh <task> [rand_obj|rand_full] [--arg value ...]
#   GPU       GPU index (default 0); MuJoCo renders on the same GPU through EGL.
#   EXP_NAME  run name (default dsrl_<suite>_<task>_<time>); output in logs/<EXP_NAME>/
# Example:
#   GPU=3 bash scripts/train.sh pinch_tongs rand_obj --noise_rows 20
# Extra arguments go to opendexrl/train.py (see `python -m opendexrl.train -h`).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TASK="${1:-water_plant}"
SUITE="${2:-rand_obj}"
shift $(( $# < 2 ? $# : 2 ))
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
# EGL enumerates all GPUs regardless of CUDA_VISIBLE_DEVICES; render on ours.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU}"

EXP_NAME="${EXP_NAME:-dsrl_${SUITE}_${TASK}_$(date +'%Y%m%d-%H%M%S')}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/logs/$EXP_NAME}"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
python -m opendexrl.train \
    --task "$TASK" --suite "$SUITE" \
    --output_dir "$OUT_DIR" --exp_name "$EXP_NAME" \
    "$@" 2>&1 | tee -a "$OUT_DIR/train.log"
