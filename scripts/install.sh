#!/bin/bash
# Build the OpenDexRL environment: one Python 3.11 uv venv with DexJoCo's OpenPI
# fork (JAX pi0.5, loads the LoRA SFT checkpoints as-is), the DexJoCo simulator,
# DSRL's jaxrl2 (pixel SAC) and opendexrl itself. The submodules are installed
# editable from third_party/ and are not modified.
#
# Usage: bash scripts/install.sh          (VENV_DIR overrides the venv path, default .venv)
# Requires: uv, git submodules checked out (git submodule update --init).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

for path in "$DEXJOCO_ROOT/openpi/pyproject.toml" "$DSRL_ROOT/setup.py"; do
    if [ ! -f "$path" ]; then
        echo "Missing $path; run: git submodule update --init" >&2
        exit 1
    fi
done
command -v uv >/dev/null || { echo "uv not found; see https://docs.astral.sh/uv/" >&2; exit 1; }

[ -x "$VENV_DIR/bin/python" ] || uv venv "$VENV_DIR" --python 3.11
PY="$VENV_DIR/bin/python"

# Version pins, applied as overrides to every resolve:
# * OpenPI pulls gym-aloha -> dm-control -> mujoco 2.x, never imported here;
#   DexJoCo needs mujoco 3.4.0 and gymnasium 1.0.0.
# * DexJoCo calls int()/float() on 1-element arrays, an error from numpy 2.4 on;
#   OpenPI needs numpy>2, so pin the last 2.2 release.
OVERRIDES="$(mktemp)"
trap 'rm -f "$OVERRIDES"' EXIT
printf 'mujoco==3.4.0\ngymnasium==1.0.0\nnumpy==2.2.6\n' > "$OVERRIDES"

# 1. DexJoCo's OpenPI fork and its client. openpi-client pins numpy==1.26.4 while
#    openpi needs numpy>2 (DexJoCo keeps them in separate conda envs); the client
#    is pure Python, so it goes in --no-deps with its runtime deps listed here.
uv pip install --python "$PY" --override "$OVERRIDES" -e "$DEXJOCO_ROOT/openpi" \
    "dm-tree>=0.1.8" "msgpack>=1.0.5" "websockets>=11.0" tree
uv pip install --python "$PY" --no-deps -e "$DEXJOCO_ROOT/openpi/packages/openpi-client"
# OpenPI only imports lerobot's dataset module; DexJoCo installs it --no-deps too.
uv pip install --python "$PY" --no-deps lerobot

# 2. DexJoCo simulator. Its numpy==1.26.4 pin conflicts with OpenPI, so the
#    package goes in --no-deps with its runtime deps listed here.
uv pip install --python "$PY" --override "$OVERRIDES" \
    "mujoco==3.4.0" "gymnasium==1.0.0" dm-robotics-transformations absl-py av glfw \
    numcodecs "zarr<3" opencv-python pyyaml
uv pip install --python "$PY" --no-deps -e "$DEXJOCO_ROOT/dexjoco"

# 3. DSRL's jaxrl2 and the extra deps it imports, then opendexrl itself.
uv pip install --python "$PY" --override "$OVERRIDES" \
    "distrax==0.1.5" "gym==0.26.2" "optax==0.2.4" "ml_collections==1.0.0" \
    matplotlib python-dateutil tqdm
uv pip install --python "$PY" --no-deps -e "$DSRL_ROOT"
uv pip install --python "$PY" --no-deps -e "$REPO_ROOT"
uv pip install --python "$PY" "numpy==2.2.6"

# 4. PaliGemma tokenizer for OpenPI: from Google's public bucket (OpenPI's own
#    source), falling back to a Hugging Face mirror (honors HF_ENDPOINT).
TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [ ! -f "$TOKENIZER" ]; then
    timeout 300 "$PY" -c "import openpi.shared.download as d; d.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})" \
        || timeout 300 "$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('RLinf/openpi_tokenizer', 'big_vision/paligemma_tokenizer.model', local_dir='$OPENPI_DATA_HOME')" \
        || echo "WARNING: could not download the PaliGemma tokenizer; place it at $TOKENIZER" >&2
fi

"$PY" - <<'EOF'
import jax, flax, mujoco, numpy, gymnasium
import openpi, openpi_client, dexjoco.tasks, jaxrl2, distrax, opendexrl  # noqa: F401
try:  # the install itself does not need a GPU
    backend = jax.default_backend()
except RuntimeError:
    backend = "no GPU visible"
print(f"OpenDexRL env OK: jax {jax.__version__} ({backend}), flax {flax.__version__}, "
      f"mujoco {mujoco.__version__}, numpy {numpy.__version__}, gymnasium {gymnasium.__version__}")
EOF
[ -f "$TOKENIZER" ] && echo "tokenizer: $TOKENIZER"
