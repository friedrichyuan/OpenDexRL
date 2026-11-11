#!/bin/bash
# End-to-end smoke test on one GPU: 2 train envs, 60-step episodes, 120 SAC
# steps, base-policy eval + two DSRL evals + checkpoints (~3-5 min). Checks that
# the whole loop runs; says nothing about learning.
#
# Usage: GPU=0 bash scripts/smoke_test.sh [task] [suite] [extra train args ...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-water_plant}"
SUITE="${2:-rand_obj}"
shift $(( $# < 2 ? $# : 2 ))
EXP_NAME="smoke_${SUITE}_${TASK}_$(date +'%Y%m%d-%H%M%S')" \
bash "$SCRIPT_DIR/train.sh" "$TASK" "$SUITE" \
    --num_envs 2 --num_eval_envs 2 --eval_episodes 2 --num_videos 1 \
    --max_episode_steps 60 \
    --start_online_updates 8 --multi_grad_step 5 --batch_size 32 \
    --max_steps 120 --log_interval 20 --eval_interval 100 --checkpoint_interval 100 \
    "$@"
