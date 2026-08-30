# When Hardware Counters Are Denied

`ncu` needs GPU performance counters. On a rented container you very often
cannot have them, and no amount of retrying inside the container changes that.
This page is what to do instead. It is not a lesser version of the workflow --
it answers a different, and for most optimization work the *first*, question.

---

## First: is it actually denied, and whose fault is it?

```bash
ncu --version >/dev/null 2>&1 && echo "ncu present" || echo "ncu not installed"
ncu --target-processes all --metrics sm__cycles_elapsed.avg python -c "import torch; torch.zeros(8, device='cuda')" 2>&1 | tail -3
```

`ERR_NVGPUCTRPERM` means counters are refused. Two independent gates cause it,
and both live **outside** the container:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params   # 1 = the host driver refuses non-root profiling
grep CapEff /proc/self/status                          # needs CAP_SYS_ADMIN (bit 0x200000)
```

- `RmProfilingAdminOnly: 1` is a module parameter set when the **host** loaded
  the driver. `/proc/driver/nvidia/params` is read-only; writing it returns
  `Permission denied` even as container root. Only the host can change it, and
  only by reloading the driver.
- A missing `CAP_SYS_ADMIN` can only be granted when the container is
  **started**. Nothing inside it can add the capability.

So: if both gates are shut, this is the **platform's configuration**, not the
machine and not the card. A different GPU or a different pod from the same
provider behaves identically. Say that plainly and move on -- do not spend
iterations looking for a workaround that does not exist. Getting counters means
either a container launched with `CAP_SYS_ADMIN` (ask the provider whether any
template or machine type does that; you cannot find out from inside), or a host
you have real root on.

---

## What is still available, and what it answers

Counters are one of two profiling paths. The other -- CUPTI's *activity* /
timeline API -- is usually **not** blocked by either gate, because it does not
read performance counters.

| Path | Blocked by `ERR_NVGPUCTRPERM`? | What it gives you |
|---|---|---|
| `ncu` hardware counters | Yes | Stall reasons, DRAM bytes, achieved occupancy, tensor-core utilization, per-line attribution |
| `nsys` timeline (CUPTI activity) | Usually no | Per-kernel duration/count/variance, the **gaps between kernels**, CUDA API call counts and time |
| `torch.profiler` | Usually no | Per-op CPU/CUDA time, kernel names, call counts |

That is enough to answer the question Phase 2 opens with -- **launch-bound,
memory-bound or compute-bound?** -- and enough to back the single most common
optimization narrative, "there are too many kernels". It is not enough for a
measured roofline.

**Verify the fallback rather than assuming it.** `torch.profiler` is already
installed and takes ten seconds; run it first as the probe. If it reports CUDA
times and kernel names, CUPTI works, and `nsys` -- which uses the same API --
will almost certainly work too.

---

## Probe: does CUPTI work at all?

```python
import torch
from torch.profiler import profile, ProfilerActivity

a = torch.randn(1024, 1024, device="cuda")
for _ in range(3):
    a @ a
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
```

Real kernel names (`ampere_sgemm_32x128_tn` and the like) with non-zero CUDA
time means the activity API is open. An empty CUDA column means it is not, and
`nsys` will not help either.

---

## Installing nsys when the image has only ncu

Many CUDA images ship `ncu` and not `nsys`. The NVIDIA apt source is usually
already configured but the package lists were cleared when the image was built:

```bash
apt-get update
apt-cache search nsight-systems | head
apt-get install -y cuda-nsight-systems-12-8   # match the image's CUDA version
nsys --version
```

Match the CUDA version of the toolchain in the image. Installing a mismatched
Nsight Systems is a slower way of arriving at the same place.

**This lands on the container's writable layer and is gone when the pod is
recycled.** Put it in whatever bootstrap script the workspace's `infra/` holds,
or you will install it again every session.

---

## Collecting a timeline

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=profile/<run_name>/reports/timeline \
  --force-overwrite=true \
  python your_workload.py
```

Then export the summaries as text -- `.nsys-rep` is a binary dump and this
repository does not commit those, only the text export next to where it is
cited:

```bash
nsys stats --report cuda_gpu_kern_sum   profile/<run_name>/reports/timeline.nsys-rep
nsys stats --report cuda_api_sum        profile/<run_name>/reports/timeline.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum profile/<run_name>/reports/timeline.nsys-rep
```

`--report` names vary between Nsight Systems versions; `nsys stats --help-reports`
lists what yours has.

---

## Reading a timeline instead of a counter report

Four questions, each answerable from `cuda_gpu_kern_sum` and `cuda_api_sum`:

**1. Launch-bound?** Compare total kernel time against wall time, and look at
`cudaLaunchKernel` in the API summary. A large share of API time in
`cudaLaunchKernel`, many launches, and short kernels means the GPU is idle
waiting for work. The fix is fusion or CUDA graphs, and this is the one
diagnosis the fallback gives you *more* directly than `ncu` does -- `ncu`
profiles kernels one at a time and hides the gaps between them.

**2. Where does the time actually go?** The per-kernel duration sum, ranked.
Optimizing anything outside the top two or three entries is wasted effort, and
this ranking needs no counters.

**3. Is the work uniform?** The per-kernel duration *variance* across
instances. High variance on variable-length inputs is load imbalance -- the
same conclusion `05-analysis-dimensions.md` reaches from per-SM active cycles,
reached from the outside.

**4. Memory traffic?** `cuda_gpu_mem_time_sum` gives host/device copy time.
This catches copies that should not be happening at all. It does **not** give
you DRAM bandwidth utilization -- that is a counter.

---

## What you genuinely cannot get, and what to do about it

Counter-only, no substitute: stall reasons, achieved occupancy, DRAM bytes and
bandwidth as a fraction of peak, tensor-core utilization, per-line (SASS or
source) attribution, PM-sampling time series and tail effects.

Anything in `05-analysis-dimensions.md`, `06-diagnosis-playbook.md` or
`08-b200-metric-names.md` that names a metric requires counters. A measured
roofline requires counters.

When they are unavailable:

- **Say so in the report, in the same place a counter number would have gone.**
  "Not measured: counters unavailable on this pod (`RmProfilingAdminOnly: 1`)"
  is evidence. A confident guess dressed as a measurement is not.
- **Substitute an A/B measurement for the counter.** You cannot see the stall
  reason, but you can change one thing and measure the evaluator's own number.
  A direction that was going to be justified by a counter can instead be
  justified by a controlled experiment -- slower to run, equally admissible.
- **Do not silently lower the bar.** The workspace's target is still the
  target. Record the missing capability as a constraint on the round, in
  `runs/` and in the workspace `CLAUDE.md`, on the day it is discovered.

---

## Report note

A run made without counters is still a run: same
`profile/<run_name>/` layout, same `REPORT.md`, with a line at the top of the
report saying which path produced it:

```markdown
**Instrumentation:** nsys 2024.6.2 timeline (CUPTI activity) + torch.profiler.
Hardware counters unavailable: `ERR_NVGPUCTRPERM`, host has
`RmProfilingAdminOnly: 1` and the container lacks `CAP_SYS_ADMIN`.
Occupancy, stall reasons and DRAM utilization are not measured in this report.
```

A reader who knows which instrument produced a number can judge it. One who
does not, cannot.
