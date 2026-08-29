#!/usr/bin/env bash
# The first measurement of the project: how fast is the opponent?
#
# Do this before writing any kernel. `torch.compile max-autotune` is one flag
# away in the official script, so someone will ask. Knowing the answer first
# turns the hardest question of the Q&A into a slide.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-.venv/bin/python}"
SHAPES="${SHAPES:-dev}"

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(fused-safe fused-sdpa graph-safe graph-sdpa)
fi

"$PY" -m odyssey ladder "${CANDIDATES[@]}" \
  --shapes "$SHAPES" --rungs L0 L1 L2 --notes "baseline ladder"
