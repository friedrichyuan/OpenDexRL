#!/bin/bash
# Queue DSRL runs on one GPU: start the next task only while the GPU has spare
# capacity, counting every DSRL run on that GPU (also runs started by other
# launchers, e.g. tune_stability.sh).
#
# Usage: bash scripts/queue.sh <suite> <task> [task ...] [-- train args ...]
# Environment:
#   GPU                 GPU index (default 0)
#   MAX_RUNS            max concurrent DSRL runs on that GPU (default 3; ~25 GB GPU, ~32 GB RAM each)
#   MIN_FREE_GPU_MB     free GPU memory required to start a run (default 30000; 0 disables)
#   MIN_FREE_RAM_GB     available host RAM required to start a run (default 36; 0 disables)
#   SETTLE              seconds the capacity must stay free before a launch (default 90), so a
#                       launcher that refills its own slot after a run ends goes first
#   LAUNCH_GAP          seconds after a launch before checking again (default 180; pi0.5 load time)
#   DRY_RUN=1           print the plan only
# Example:
#   GPU=1 bash scripts/queue.sh rand_obj pick_bucket fold_glasses hammer_nail
# Waits for all queued runs; exits non-zero if any failed. Logs: logs/<run>/ and logs/<run>.out
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SUITE="${1:?usage: queue.sh <suite> <task> [task ...] [-- train args ...]}"
shift
TASKS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TASKS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ "${#TASKS[@]}" -gt 0 ] || { echo "no tasks given" >&2; exit 1; }
GPU="${GPU:-0}"
MAX_RUNS="${MAX_RUNS:-3}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-30000}"
MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-36}"
SETTLE="${SETTLE:-90}"
LAUNCH_GAP="${LAUNCH_GAP:-180}"
PROC_PATTERN="${PROC_PATTERN:-opendexrl.train}"

runs_on_gpu() {  # DSRL main processes whose CUDA_VISIBLE_DEVICES is $GPU
    local n=0 pid
    for pid in $(pgrep -f "$PROC_PATTERN"); do
        tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && n=$((n + 1))
    done
    echo "$n"
}
has_capacity() {
    [ "$(runs_on_gpu)" -lt "$MAX_RUNS" ] || return 1
    if [ "$MIN_FREE_GPU_MB" -gt 0 ]; then
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU") || return 1
        [ "$free_mb" -ge "$MIN_FREE_GPU_MB" ] || return 1
    fi
    if [ "$MIN_FREE_RAM_GB" -gt 0 ]; then
        local avail_gb
        avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
        [ "$avail_gb" -ge "$MIN_FREE_RAM_GB" ] || return 1
    fi
    return 0
}

STAMP="$(date +'%Y%m%d-%H%M%S')"
echo "[queue] GPU $GPU suite=$SUITE tasks=(${TASKS[*]}) max_runs=$MAX_RUNS extra: $*"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  runs now on GPU $GPU: $(runs_on_gpu); capacity: $(has_capacity && echo yes || echo no)"
    exit 0
fi
mkdir -p "$REPO_ROOT/logs"

PIDS=()
NAMES=()
for task in "${TASKS[@]}"; do
    free_since=""
    while true; do
        if has_capacity; then
            [ -z "$free_since" ] && free_since=$(date +%s)
            [ $(( $(date +%s) - free_since )) -ge "$SETTLE" ] && break
        else
            free_since=""
        fi
        sleep 15
    done
    run="dsrl_${SUITE}_${task}_${STAMP}"
    echo "[queue] $(date +'%F %T') start $task on GPU $GPU ($run)"
    GPU="$GPU" EXP_NAME="$run" bash "$SCRIPT_DIR/train.sh" "$task" "$SUITE" "$@" \
        > "$REPO_ROOT/logs/$run.out" 2>&1 &
    PIDS+=($!)
    NAMES+=("$task")
    sleep "$LAUNCH_GAP"
done

status=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[queue] $(date +'%F %T') ${NAMES[$i]} done"
    else
        echo "[queue] $(date +'%F %T') ${NAMES[$i]} FAILED (see logs/dsrl_${SUITE}_${NAMES[$i]}_${STAMP}.out)"
        status=1
    fi
done
exit $status
