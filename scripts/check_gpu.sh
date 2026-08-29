#!/usr/bin/env bash
# D0 gate. If any of this fails, fix it before writing a single kernel --
# discovering a broken profiler on day two costs a day.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-.venv/bin/python}"
status=0

step() { printf '\n=== %s ===\n' "$1"; }

step "nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
else
  echo "nvidia-smi not found -- no GPU visible"; status=1
fi

step "torch"
"$PY" -m odyssey doctor || status=1

step "Nsight Compute"
if command -v ncu >/dev/null 2>&1; then
  ncu --version | head -3
  echo "-- permission probe (needs CAP_SYS_ADMIN or nvidia NVreg_RestrictProfilingToAdminUsers=0)"
  ncu --metrics sm__cycles_elapsed.avg --target-processes all \
      "$PY" -c "import torch; torch.zeros(8, device='cuda').sum().item()" >/dev/null 2>&1 \
    && echo "   profiling permitted" \
    || { echo "   PROFILING DENIED -- fix this on D0, not D2"; status=1; }
else
  echo "ncu not found. Install the CUDA toolkit or Nsight Compute."; status=1
fi

step "clock locking"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "To reduce timing variance (needs root):"
  echo "  sudo nvidia-smi -pm 1"
  echo "  sudo nvidia-smi --lock-gpu-clocks=<min>,<max>"
  echo "  sudo nvidia-smi --lock-memory-clocks=<min>,<max>"
fi

step "correctness smoke on CPU"
"$PY" -m odyssey bench fused-safe passthrough --shapes smoke --device cpu \
  --trials 2 --warmup 2 --repeats 4 --rounds 1 --no-record || status=1

printf '\n=== %s ===\n' "$([[ $status -eq 0 ]] && echo READY || echo NOT READY)"
exit $status
