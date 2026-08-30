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

# These shapes are tiny and this box has many cores: torch otherwise starts one
# OpenMP thread per core (192 on the rented pod) and the barrier between them
# costs orders of magnitude more than the work. Measured on `plain`: 1794 ms at
# 192 threads, 0.47 ms at one, which is the difference between a ten-minute gate
# and a three-second one. Nothing here is a timing result, so pinning the count
# costs nothing and the caller can still override it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(fused-safe passthrough dispatch)
fi

exec "$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes smoke --quiet -- \
  --device cpu --atol 0.001 --rtol 0.01 \
  --accuracy-trials 3 --warmup 2 --repeats 8 --benchmark-rounds 1
