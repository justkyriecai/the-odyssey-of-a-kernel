#!/usr/bin/env bash
# D0 gate. If any of this fails, fix it before writing a single kernel --
# discovering a broken profiler on day two costs a day.
#
#   ./scripts/check_gpu.sh                      driver, torch, ncu permission, then the
#                                               workspace's own smoke test if there is one workspace
#   ./scripts/check_gpu.sh <workspace-name>     the same, naming the workspace
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
"$PY" - <<'PY' || status=1
import sys, torch
print(f"python {sys.version.split()[0]}   torch {torch.__version__}   cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda runtime {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GiB  {p.multi_processor_count} SMs")
else:
    print("  no CUDA device. Correctness runs on CPU; every latency number needs a GPU.")
    raise SystemExit(1)
PY

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

step "workspace smoke"
WS="${1:-${WORKSPACE:-}}"
if [[ -z "$WS" ]]; then
  candidates=(workspace/*/)
  if [[ ${#candidates[@]} -eq 1 && -d "${candidates[0]}" ]]; then
    WS="$(basename "${candidates[0]}")"
  fi
fi
if [[ -n "$WS" && -x "workspace/$WS/scripts/smoke.sh" ]]; then
  echo "workspace/$WS"
  "workspace/$WS/scripts/smoke.sh" || status=1
else
  echo "no single workspace with scripts/smoke.sh -- pass the workspace name to run its smoke test"
fi

printf '\n=== %s ===\n' "$([[ $status -eq 0 ]] && echo READY || echo NOT READY)"
exit $status
