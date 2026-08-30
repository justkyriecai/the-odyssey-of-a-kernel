#!/usr/bin/env bash
# Correctness only, CPU only, seconds. Safe to run anywhere, including CI.
# A candidate that fails here is wrong, not slow.
#
# The default list is the candidates that must always pass. `fused-sdpa` is not
# among them: on backends that do not accumulate softmax in fp32 it genuinely
# fails bf16, which is the finding, not a regression. Run it explicitly:
#   ./scripts/smoke.sh fused-sdpa
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$WS/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(fused-safe passthrough dispatch)
fi

exec "$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes smoke --quiet -- \
  --device cpu --atol 0.001 --rtol 0.01 \
  --accuracy-trials 3 --warmup 2 --repeats 8 --benchmark-rounds 1
