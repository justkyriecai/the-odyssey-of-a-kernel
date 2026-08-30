#!/usr/bin/env bash
# Create the Python environment. Run once per machine, including a rented box.
#
# There is no package to install: the framework is prompts, skills and rules,
# and a workspace's `verify.py` runs straight from its directory. This only has
# to produce a venv whose torch matches the machine.
#
# Two knobs, both there because a rented GPU box is not a laptop:
#
#   VENV_DIR      where the venv goes. Default ./.venv, and the checkout keeps a
#                 symlink when it goes anywhere else. A checkout on a network
#                 volume must not hold torch: the wheel is 4+ GB across ~12k
#                 files, and every interpreter start reads it over the network.
#   SYSTEM_TORCH  auto (default) / 1 / 0. A CUDA image already ships a torch
#                 built against its own nvcc and Nsight Compute; a newer wheel
#                 from PyPI silently mismatches both, so inherit the image's
#                 copy when there is one rather than downloading a second.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-}"   # e.g. https://download.pytorch.org/whl/cu124
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SYSTEM_PY="${SYSTEM_PY:-/usr/bin/python3}"
SYSTEM_TORCH="${SYSTEM_TORCH:-auto}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# A real .venv directory where a symlink belongs would be silently written
# through, leaving the old tree on the volume. Say so instead of guessing.
if [[ "$VENV_DIR" != "$ROOT_DIR/.venv" && -d "$ROOT_DIR/.venv" && ! -L "$ROOT_DIR/.venv" ]]; then
  echo "$ROOT_DIR/.venv is a real directory but VENV_DIR points elsewhere." >&2
  echo "Remove it first: rm -rf $ROOT_DIR/.venv" >&2
  exit 1
fi

use_system_torch=0
case "$SYSTEM_TORCH" in
  1) use_system_torch=1 ;;
  0) ;;
  auto)
    if [[ -x "$SYSTEM_PY" ]] && "$SYSTEM_PY" -c 'import torch' >/dev/null 2>&1; then
      use_system_torch=1
    fi
    ;;
  *) echo "SYSTEM_TORCH must be auto, 1 or 0 (got $SYSTEM_TORCH)" >&2; exit 1 ;;
esac

if (( use_system_torch )); then
  echo "using the interpreter's own torch ($SYSTEM_PY)"
  uv venv --python "$SYSTEM_PY" --system-site-packages "$VENV_DIR"
  # uv resolves against the venv alone and does not count an inherited package
  # as installed, so an unfiltered install pulls a second torch and its whole
  # CUDA stack in behind it. Hand it everything except torch.
  req="$(mktemp)"
  trap 'rm -f "$req"' EXIT
  grep -vE '^[[:space:]]*torch([[:space:]<>=!~]|$)' requirements.txt > "$req"
  uv pip install --python "$VENV_DIR/bin/python" -r "$req"
  rm -f "$req"
  trap - EXIT
else
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
  if [[ -n "$TORCH_INDEX" ]]; then
    uv pip install --python "$VENV_DIR/bin/python" --index-url "$TORCH_INDEX" torch
  fi
  uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
fi

if [[ "$VENV_DIR" != "$ROOT_DIR/.venv" ]]; then
  ln -sfn "$VENV_DIR" "$ROOT_DIR/.venv"
  echo "$ROOT_DIR/.venv -> $VENV_DIR"
fi

echo
"$VENV_DIR/bin/python" - <<'PY'
import os, sys, torch
print(f"python {sys.version.split()[0]}   torch {torch.__version__}   cuda {torch.cuda.is_available()}")
print(f"  torch from {os.path.dirname(torch.__file__)}")
if torch.cuda.is_available():
    print(f"  cuda runtime {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
PY
