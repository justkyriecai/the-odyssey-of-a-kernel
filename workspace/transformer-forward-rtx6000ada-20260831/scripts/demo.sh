#!/usr/bin/env bash
# The judge-picks-a-shape moment, as one short command.
#
# The pitch hands the panel the shape list and says "pick one". This script is
# what runs next: the organizer's own script, unmodified, with the shipping
# candidate patched in, at the problem statement's tolerance -- PASS per trial, max_abs and
# max_rel on screen, both medians side by side.
#
#   ./scripts/demo.sh              list the shapes for the panel to choose from
#   ./scripts/demo.sh center       run one
#
# SHAPES overrides the shape set (default: official). CANDIDATE overrides the
# implementation under test (default: dispatch, the shipping layer).
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$WS/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
SHAPES="${SHAPES:-official}"
CANDIDATE="${CANDIDATE:-dispatch}"

if [[ $# -eq 0 ]]; then
  echo "Pick a shape (set: $SHAPES), then: ./scripts/demo.sh <name>"
  echo
  "$PY" - "$WS/bench/shapes/$SHAPES.json" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
for case in raw["cases"]:
    c = {**raw.get("defaults", {}), **case}
    shape = (f"B={c['batch_size']} S={c['seq_len']} d={c['d_model']} "
             f"H={c['num_heads']} ffn={c['ffn_dim']} L={c['num_layers']}")
    print(f"  {c['name']:<14} {shape:<44} {'causal' if c.get('causal') else 'full'} {c['dtype']}")
PY
  exit 0
fi

CASE="$1"; shift
exec "$PY" "$WS/verify.py" "$CANDIDATE" --shapes "$SHAPES" --case "$CASE" -- \
  --atol 0.002 --rtol 0.02 "$@"
