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
# The problem statement's numbers: "relative error < 0.02, abs error <
# 0.002". They are also the script's argparse defaults; passed explicitly so
# every recorded row carries the tolerance it was judged at.
TOLERANCE=(--atol 0.002 --rtol 0.02)

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(fused-safe fused-sdpa graph-safe graph-sdpa)
fi

# `|| true` on every rung: a candidate that fails a case reports exit 2, which
# is a recorded verdict, not a reason to abandon the other rungs. fused-sdpa
# failing bf16 is the documented finding that motivates dispatch; the first L0
# run of this ladder aborted on exactly that and never measured the opponent.
"$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes "$SHAPES" --quiet --record --notes "ladder L0" -- \
  "${TOLERANCE[@]}" || true
# --benchmark-on-failure, on the compiled rungs only. Compiling the baseline
# moves the accuracy reference itself: under TF32 the max-autotune build drifts
# ~5e-3 from its own eager version, so even a numerically identical candidate
# (passthrough) fails the official tolerance against it -- measured on the RTX
# 6000 Ada, 2026-08-30. Correctness is judged on the uncompiled rung above,
# where the reference is the real one; these rungs exist for the timing, which
# the script would otherwise skip on the spurious FAIL. Exit code 2 and
# passed=False are still recorded -- the flag skips nothing, it only stops the
# script from withholding the number this rung is for.
"$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes "$SHAPES" --quiet --record --notes "ladder L2" -- \
  "${TOLERANCE[@]}" --compile-baseline --compile-mode max-autotune --benchmark-on-failure || true
"$PY" "$WS/verify.py" passthrough --shapes "$SHAPES" --quiet --record --notes "ladder compile-only" -- \
  "${TOLERANCE[@]}" --compile-baseline --compile-mode max-autotune --benchmark-on-failure || true
