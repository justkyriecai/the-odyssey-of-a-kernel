#!/usr/bin/env bash
# The judge-picks-a-shape moment, as one short command.
#
# The pitch hands the panel the shape list and says "pick one". This script is
# what runs next: the organizer's own script, unmodified, with the shipping
# candidate patched in, at the strict tolerance -- PASS per trial, max_abs and
# max_rel on screen, both medians side by side.
#
#   ./scripts/demo.sh              list the shapes for the panel to choose from
#   ./scripts/demo.sh center       run one
#
# SHAPES overrides the shape set (default: official). CANDIDATE overrides the
# implementation under test (default: dispatch, the shipping layer).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-.venv/bin/python}"
SHAPES="${SHAPES:-official}"
CANDIDATE="${CANDIDATE:-dispatch}"

if [[ $# -eq 0 ]]; then
  echo "Pick a shape (set: $SHAPES), then: ./scripts/demo.sh <name>"
  echo
  "$PY" - "$SHAPES" <<'PYEOF'
import sys
from odyssey.shapes import load_set

for case in load_set(sys.argv[1]):
    shape = (
        f"B={case.batch_size} S={case.seq_len} d={case.d_model} "
        f"H={case.num_heads} ffn={case.ffn_dim} L={case.num_layers}"
    )
    flags = "causal" if case.causal else "full"
    print(f"  {case.name:<14} {shape:<44} {flags} {case.dtype}")
PYEOF
  exit 0
fi

CASE="$1"; shift
exec "$PY" -m odyssey official "$CANDIDATE" --shapes "$SHAPES" --case "$CASE" -- \
  --atol 0.001 --rtol 0.01 "$@"
