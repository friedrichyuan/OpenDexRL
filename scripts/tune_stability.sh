#!/bin/bash
# Stability tuning for DSRL: run each (task, arm) pair through a job queue.
#
# Arms (all keep --critic_reduction min, the trainer default):
#   utd5        5 SAC updates per transition instead of 20. Same env-data budget as a
#               500k-step UTD-20 run (125k updates), evaluated every 1250 transitions
#               like the UTD-20 runs, so the curves line up on the data axis.
#   utd5_rows20 utd5 + --noise_rows 20: one noise vector per executed action instead
#               of one vector repeated over the whole chunk.
# The step-0 eval measures each arm's own start point (noise laid out like its action).
#
# Usage: bash scripts/tune_stability.sh [rand_obj|rand_full] [-- extra train args ...]
# Environment:
#   TASKS          tasks (default: "pinch_tongs click_mouse"; jobs run arm by arm, in this order)
#   ARMS           arms to run (default: "utd5 utd5_rows20")
#   GPU_IDS        GPUs to use (default: all visible)
#   SLOTS_PER_GPU  concurrent runs per GPU (default: floor(GPU memory / 28 GB); ~25 GB per run)
#   LAUNCH_GAP     seconds between launches so pi0.5 loads one at a time (default 120)
#   DRY_RUN=1      print the job list and slot plan only
# Logs: logs/dsrl_<suite>_<task>_<arm>_<time>/ and logs/<run>.out
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

SUITE="${1:-rand_obj}"
shift || true
[ "${1:-}" = "--" ] && shift
read -r -a TASK_LIST <<< "${TASKS:-pinch_tongs click_mouse}"
read -r -a ARM_LIST <<< "${ARMS:-utd5 utd5_rows20}"

arm_args() {
    local common="--multi_grad_step 5 --max_steps 125000 --eval_interval 6250 --checkpoint_interval 6250"
    case "$1" in
        utd5) echo "$common --base_noise_rows 1" ;;
        utd5_rows20) echo "$common --noise_rows 20 --base_noise_rows 20" ;;
        *) echo "unknown arm '$1' (expected utd5, utd5_rows20)" >&2; return 1 ;;
    esac
}

read -r -a GPUS <<< "${GPU_IDS:-$(seq -s ' ' 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
if [ -z "${SLOTS_PER_GPU:-}" ]; then
    mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPUS[0]}")
    SLOTS_PER_GPU=$(( mem_mb / 28000 ))
    [ "$SLOTS_PER_GPU" -lt 1 ] && SLOTS_PER_GPU=1
fi
SLOT_GPU=()
for gpu in "${GPUS[@]}"; do
    for _ in $(seq "$SLOTS_PER_GPU"); do SLOT_GPU+=("$gpu"); done
done

JOBS=()
for arm in "${ARM_LIST[@]}"; do
    arm_args "$arm" >/dev/null || exit 1
    for task in "${TASK_LIST[@]}"; do JOBS+=("$task:$arm"); done
done
STAMP="$(date +'%Y%m%d-%H%M%S')"
echo "[tune] suite=$SUITE jobs=(${JOBS[*]}) slots=${#SLOT_GPU[@]} (GPUs ${GPUS[*]} x $SLOTS_PER_GPU) extra: $*"
if [ "${DRY_RUN:-0}" = "1" ]; then
    for job in "${JOBS[@]}"; do echo "  ${job%%:*} [${job##*:}]: $(arm_args "${job##*:}") $*"; done
    exit 0
fi
mkdir -p "$REPO_ROOT/logs"

SLOT_PID=()
SLOT_JOB=()
status=0
reap() {  # free slots whose run has exited; record failures
    for s in "${!SLOT_GPU[@]}"; do
        pid="${SLOT_PID[$s]:-}"
        [ -z "$pid" ] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; rc=$?
            [ "$rc" -ne 0 ] && { echo "[tune] ${SLOT_JOB[$s]} FAILED (exit $rc)"; status=1; }
            [ "$rc" -eq 0 ] && echo "[tune] ${SLOT_JOB[$s]} done"
            SLOT_PID[$s]=""
        fi
    done
}
for job in "${JOBS[@]}"; do
    task="${job%%:*}"; arm="${job##*:}"
    while true; do
        reap
        free=""
        for s in "${!SLOT_GPU[@]}"; do [ -z "${SLOT_PID[$s]:-}" ] && { free=$s; break; }; done
        [ -n "$free" ] && break
        sleep 30
    done
    run="dsrl_${SUITE}_${task}_${arm}_${STAMP}"
    gpu="${SLOT_GPU[$free]}"
    echo "[tune] start $task [$arm] on GPU $gpu (slot $free): $run"
    # shellcheck disable=SC2046
    GPU=$gpu EXP_NAME="$run" bash "$SCRIPT_DIR/train.sh" "$task" "$SUITE" $(arm_args "$arm") "$@" \
        > "$REPO_ROOT/logs/$run.out" 2>&1 &
    SLOT_PID[$free]=$!
    SLOT_JOB[$free]="$task[$arm]"
    sleep "${LAUNCH_GAP:-120}"  # let the run load pi0.5 before the next one starts
done
while true; do
    reap
    busy=0
    for s in "${!SLOT_GPU[@]}"; do [ -n "${SLOT_PID[$s]:-}" ] && busy=1; done
    [ "$busy" = 0 ] && break
    sleep 30
done
exit $status
