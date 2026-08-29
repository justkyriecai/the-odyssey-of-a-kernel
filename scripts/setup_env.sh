#!/usr/bin/env bash
# Create the environment. Run once per machine, including a rented GPU box.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-}"   # e.g. https://download.pytorch.org/whl/cu124

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv venv --python "$PYTHON_VERSION" .venv

if [[ -n "$TORCH_INDEX" ]]; then
  uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" torch
fi
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .

echo
.venv/bin/python -m odyssey doctor
