#!/usr/bin/env bash
# Create the Python environment. Run once per machine, including a rented box.
#
# There is no package to install: the framework is prompts, skills and rules,
# and a workspace's `verify.py` runs straight from its directory. This only
# has to produce a venv with a torch that matches the machine's CUDA runtime.
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

echo
.venv/bin/python - <<'PY'
import sys, torch
print(f"python {sys.version.split()[0]}   torch {torch.__version__}   cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB")
PY
