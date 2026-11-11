#!/bin/bash
# Common environment for the OpenDexRL scripts. Source it from bash or zsh:
#   source scripts/env.sh
# Every path can be overridden by exporting it before sourcing.

# zsh sets $0 to the sourced file; bash exposes it through BASH_SOURCE.
_ODR_ENV_SH="${BASH_SOURCE[0]:-$0}"
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$_ODR_ENV_SH")/.." && pwd)}"
export DEXJOCO_ROOT="${DEXJOCO_ROOT:-$REPO_ROOT/third_party/dexjoco}"
export DSRL_ROOT="${DSRL_ROOT:-$REPO_ROOT/third_party/dsrl_pi0}"
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
# Local copy of the DexJoCo/DexJoCo-Pi05 release (JAX/LoRA SFT checkpoints, used as-is).
export DEXJOCO_MODELS_ROOT="${DEXJOCO_MODELS_ROOT:-$REPO_ROOT/checkpoints/DexJoCo-Pi05}"
# PaliGemma tokenizer: <OPENPI_DATA_HOME>/big_vision/paligemma_tokenizer.model (fetched by install.sh).
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/assets/openpi}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
