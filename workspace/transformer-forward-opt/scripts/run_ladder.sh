#!/usr/bin/env bash
# The first measurement of the project: how fast is the opponent?
#
# Do this before writing any kernel. The official script compares its baseline
# against the candidate, and `--compile-baseline --compile-mode max-autotune`
# turns that baseline into `torch.compile` at its strongest -- one flag away
# for anyone in the room. Three runs give the whole ladder:
#
#   L0   candidates vs eager                the script's default denominator
#   L2   candidates vs max-autotune         the real opponent; this is the number to quote
#   --   passthrough vs max-autotune        what compile alone buys: 1 / the speedup printed
#
# Every row is recorded with its `script_args`, so calibration can tell the
# two denominators apart.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$WS/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
SHAPES="${SHAPES:-dev}"
STRICT=(--atol 0.001 --rtol 0.01)

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(fused-safe fused-sdpa graph-safe graph-sdpa)
fi

"$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes "$SHAPES" --quiet --record --notes "ladder L0" -- \
  "${STRICT[@]}"
"$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes "$SHAPES" --quiet --record --notes "ladder L2" -- \
  "${STRICT[@]}" --compile-baseline --compile-mode max-autotune
"$PY" "$WS/verify.py" passthrough --shapes "$SHAPES" --quiet --record --notes "ladder compile-only" -- \
  "${STRICT[@]}" --compile-baseline --compile-mode max-autotune
