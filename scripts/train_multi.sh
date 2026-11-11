#!/bin/bash
# Several DSRL runs at once, tasks assigned round-robin to GPUs.
#
# Usage: bash scripts/train_multi.sh [rand_obj|rand_full] [task ...] [-- train args ...]
#   no task list -> the 8 tasks below, one per GPU of an 8-GPU node.
#   GPU_IDS   space-separated GPU indices (default: all GPUs from nvidia-smi); task k runs
#             on GPU_IDS[k % len]. Runs may share a GPU (~25 GB GPU, ~32 GB host RAM each
#             with the default 8 train + 16 eval envs).
#   DRY_RUN=1 print the task -> GPU plan without launching.
# Examples:
#   bash scripts/train_multi.sh rand_obj
#   GPU_IDS="0 1" bash scripts/train_multi.sh rand_obj click_mouse pinch_tongs -- --noise_rows 20
# Waits for all runs; exits non-zero if any failed. Logs: logs/<run>/ and logs/<run>.out
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SUITE="${1:-rand_obj}"
shift || true
TASKS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TASKS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
if [ "${#TASKS[@]}" -eq 0 ]; then
    TASKS=(click_mouse pinch_tongs pick_bucket fold_glasses
           bimanual_hanoi bimanual_assembly bimanual_microwave_cook bimanual_photograph)
fi
read -r -a GPUS <<< "${GPU_IDS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
STAMP="$(date +'%Y%m%d-%H%M%S')"
mkdir -p "$REPO_ROOT/logs"

PIDS=()
RUNS=()
for k in "${!TASKS[@]}"; do
    task="${TASKS[$k]}"
    gpu="${GPUS[$(( k % ${#GPUS[@]} ))]}"
    run="dsrl_${SUITE}_${task}_${STAMP}"
    echo "[train_multi] $task -> GPU $gpu ($run)"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  GPU=$gpu EXP_NAME=$run bash $SCRIPT_DIR/train.sh $task $SUITE $*"
        continue
    fi
    GPU=$gpu EXP_NAME="$run" bash "$SCRIPT_DIR/train.sh" "$task" "$SUITE" "$@" \
        > "$REPO_ROOT/logs/$run.out" 2>&1 &
    PIDS+=($!)
    RUNS+=("$run")
done

status=0
for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
        echo "[train_multi] ${RUNS[$k]} FAILED (see logs/${RUNS[$k]}.out)"
        status=1
    fi
done
exit $status
