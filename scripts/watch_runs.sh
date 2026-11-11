#!/bin/bash
# Periodic status of DSRL runs: one summary line per run every INTERVAL seconds,
# plus an immediate line when a run errors out or finishes.
# Usage: bash scripts/watch_runs.sh logs/<run> [logs/<run> ...]   (INTERVAL seconds, default 1800)
INTERVAL="${INTERVAL:-1800}"
declare -A reported
summary() {
    for d in "$@"; do
        local name evals last
        name=$(basename "$d")
        evals=$(grep -oE "^\[step [0-9]+\] eval_(base|dsrl)/success_rate=[0-9.]+" "$d/train.log" 2>/dev/null \
            | sed -E 's/^\[step ([0-9]+)\] eval_(base|dsrl)\/success_rate=([0-9.]+)/\1:\3/' | tr '\n' ' ')
        last=$(grep -oE "^\[step [0-9]+\] train" "$d/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
        echo "[status] $name | step ${last:-0} | eval success (step:rate) $evals"
    done
}
next=$(( $(date +%s) + INTERVAL ))
while true; do
    for d in "$@"; do
        if [ -z "${reported[$d]:-}" ] && grep -qE "Traceback|Killed|\[dsrl\] done" "$d/train.log" 2>/dev/null; then
            reported[$d]=1
            echo "[event] $(basename "$d"): $(grep -m1 -E 'Traceback|Killed|\[dsrl\] done' "$d/train.log")"
            summary "$d"
        fi
    done
    if [ "$(date +%s)" -ge "$next" ]; then
        summary "$@"
        next=$(( next + INTERVAL ))
    fi
    all_done=1
    for d in "$@"; do [ -z "${reported[$d]:-}" ] && all_done=0; done
    [ "$all_done" = 1 ] && exit 0
    sleep 20
done
