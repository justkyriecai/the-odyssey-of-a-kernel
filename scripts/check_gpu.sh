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

step "profilers"
# Two independent capabilities, and a container commonly has the second without
# the first. ncu reads hardware performance counters, which the driver gates
# behind CAP_SYS_ADMIN (or NVreg_RestrictProfilingToAdminUsers=0 on the host).
# nsys traces through CUPTI instead, so it is usually permitted where ncu is
# refused, and it still answers the question Phase 2 opens with: per-kernel
# time, launch counts, and the gaps between kernels. Only the measured-intensity
# roofline and odyssey-ncu-report actually need the counters. Denied counters
# are therefore a downgrade, not a stop -- see that skill's
# reference/10-no-counter-fallback.md, and docs/runpod.md, "The NCU question",
# in a workspace that has one.
WORKLOAD="import torch; torch.zeros(8, device='cuda').sum().item()"

counters="not installed"
if command -v ncu >/dev/null 2>&1; then
  ncu --version | sed -n '3p'
  if ncu --metrics sm__cycles_elapsed.avg --target-processes all \
      "$PY" -c "$WORKLOAD" >/dev/null 2>&1; then
    counters="permitted"
  else
    counters="DENIED (ERR_NVGPUCTRPERM)"
  fi
fi

# Which gate shut, when they did. Both are set outside the container -- the
# module parameter when the host loaded the driver, the capability when the
# container was started -- so this is the platform's answer for every card it
# rents, not this pod's. Print it once here rather than rediscovering it from
# inside a phase.
gates=""
if [[ "$counters" == DENIED* ]]; then
  gates="$(grep RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -d ' ')"
  gates="${gates:-RmProfilingAdminOnly:unreadable}"
  capeff="$(awk '/^CapEff:/ {print $2}' /proc/self/status 2>/dev/null)"
  if [[ -n "$capeff" ]] && (( (16#$capeff >> 21) & 1 )); then
    gates="$gates, CAP_SYS_ADMIN held"   # bit 21 = CAP_SYS_ADMIN
  else
    gates="$gates, no CAP_SYS_ADMIN"
  fi
fi

timeline="not installed"
if command -v nsys >/dev/null 2>&1; then
  nsys --version | tail -1
  probe_dir="$(mktemp -d)"
  if nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
      -o "$probe_dir/probe" "$PY" -c "$WORKLOAD" >/dev/null 2>&1 \
     && [[ -s "$probe_dir/probe.nsys-rep" ]]; then
    timeline="working"
  else
    timeline="DENIED"
  fi
  rm -rf "$probe_dir"
fi

echo "-- hardware counters (ncu): $counters"
echo "-- kernel timeline (nsys):  $timeline"
if [[ "$counters" == "permitted" ]]; then
  echo "   full profiling evidence available"
elif [[ "$timeline" == "working" ]]; then
  echo "   counters unavailable; nsys covers launch-bound vs memory-bound vs compute-bound."
  echo "   gates: $gates -- both are set outside this container, so retrying and"
  echo "   swapping cards will not change it. The measured-intensity roofline needs"
  echo "   a box that grants them."
else
  echo "   NO PROFILER AT ALL -- fix this on D0, not D2. See docs/runpod.md."
  status=1
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
