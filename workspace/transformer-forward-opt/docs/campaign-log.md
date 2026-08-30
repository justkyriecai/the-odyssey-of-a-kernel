# Campaign log

The optimization campaign, one entry per session, newest last. Each entry names
the evidence it produced -- a `runs/benchmark.csv` note tag, a `solutions.jsonl`
node, a profile directory, a commit -- so the tech report can cite rather than
reconstruct. Numbers in prose are medians; the CSV rows carry p90.

## 2026-08-30 -- D0: environment, gates, and the opponent

**Machine.** RTX 6000 Ada (sm_89, 142 SMs, 48 GB), driver 580.126.20, RunPod
Secure Cloud. The pod image's own torch (2.8.0+cu128) is used rather than a
downloaded wheel -- it matches the image's nvcc and Nsight, and moving the venv
off the NFS network volume onto local disk cut `import torch` from 5.0 s to
1.4 s (commit `a8af3f1`). The CPU smoke gate was unusable at 192 OpenMP threads
(1794 ms for a case that takes 0.47 ms single-threaded, a 3800x penalty on
tiny tensors); pinned in `smoke.sh` (commit `86faf75`). Full D0 gate:
10+ minutes before, 7.6 s after.

**Profiler permissions, settled by probe.** `ncu` is denied on this pod
(`ERR_NVGPUCTRPERM`: host driver `RmProfilingAdminOnly=1`, container without
`CAP_SYS_ADMIN`; no in-container fix exists) but `nsys` traces through CUPTI
and works. Verdict for the whole campaign: kernel timelines, launch counts and
inter-kernel gaps are available; hardware counters (stall reasons, DRAM bytes,
occupancy) are not on this box. `check_gpu.sh` now probes both paths separately
(commit `3ae7ecc`).

**Tolerance corrected to the problem statement.** The organizer's statement
(ref/track3-problem-statement.md, updated 27 Aug) sets `abs < 0.002 OR
rel < 0.02` -- the script's argparse defaults; its docstring's stricter
`0.001/0.01` is a leftover. The whole workspace had been targeting the stricter
pair as insurance; that margin was optimization headroom given away, and is now
reclaimed (commit `2ef7a73`). Also settled by the statement: participants run
the benchmark on their own machine and report; per-shape dispatch is explicitly
endorsed; no judging criterion scores raw speed.

**The opponent, measured.** `torch.compile` on the baseline, center shape:
default mode 0.615 ms, max-autotune 0.572 ms, vs eager 1.40 ms -- compile alone
is worth 2.4x. Two facts found while measuring it (probe logs, not recorded as
benchmark rows): under TF32, the max-autotune baseline drifts `max_abs 5.3e-3`
from its own eager weights -- past the official tolerance, so even a
numerically identical candidate FAILs against a compiled reference
(`--benchmark-on-failure` is therefore required on the compiled ladder rungs,
with correctness taken from the uncompiled rung); disabling TF32 collapses the
drift to 1.4e-6, isolating the cause to TF32 rounding differences between
cuBLAS and Triton matmul templates, not accumulation-order tricks.

**Ladder L0 (candidates vs eager, dev set, official tolerance).** 36 rows,
notes=`ladder L0`; DAG nodes n001-n005 in `runs/solutions.jsonl`.

| candidate | center | batch-1 | seq-1024 | heads-16 | bf16/fp16 |
|---|---|---|---|---|---|
| fused-safe | 1.34x | 1.39x | 1.33x | 1.22x | PASS, max_abs 0 |
| fused-sdpa | 1.69x | 1.66x | 4.08x | 2.77x | FAIL 0.0625 / 0.0059 |
| graph-safe | 2.24x | 8.30x | 1.33x | 1.26x | PASS |
| graph-sdpa | 2.53x | 7.47x | 4.03x | 2.83x | FAIL (inherited) |

Readings: CUDA Graphs and SDPA are independent, composable gains -- graphs own
the launch-bound end (batch-1: 0.172 ms, 8.3x), SDPA owns the attention-heavy
end (seq-1024: 4.1x). `graph-sdpa` on fp32 center hits 0.568 ms -- parity with
the max-autotune baseline -- before any new work. SDPA's softmax does not
accumulate in fp32 on every backend: bf16 fails by 31x, fp16 by 2.9x, so
reduced-precision shapes must dispatch to a safe path. The first L0 run also
exposed a ladder bug: a recorded FAIL aborted the remaining rungs via `set -e`;
fixed so every rung runs and records.

**Ladder L2 (vs `--compile-baseline --compile-mode max-autotune`), 45 rows,
notes=`ladder L2`/`ladder compile-only`.** The earlier 0.572 ms opponent number
was measured with TF32 off; with the evaluator's defaults the opponent is far
stronger on the launch-bound end and absent elsewhere: center 0.33 ms (4.3x over
eager), center-bf16 0.23 ms, batch-1 0.107 ms (13.3x) -- but seq-1024 78.0 ms,
batch-128 1.63 ms, wide-1024 9.68 ms, heads-16 2.94 ms, all within noise of
eager. Compile owns exactly the tiny-shape group and nothing else. Our best,
graph-sdpa, is 1.7x behind it on fp32 center (0.589 vs 0.345 ms) and 4.0x ahead
at seq-1024 (19.4 vs 78.0 ms). Also recorded: the compiled bf16 baseline drifts
0.0625 from its own eager numerics (fused-safe, exact at L0, "fails" by that
margin against it) -- the compiled reference is not a usable correctness
reference, which is why the compiled rungs run `--benchmark-on-failure` and
correctness is judged at L0 only.

**nsys anatomy (runs/profile/00-regime-anatomy/, REPORT.md).** Eager baseline:
~116 kernels per forward at 4-11 us median -- launch-bound confirmed by count,
not intuition. The compiled baseline runs ~5 fused Triton kernels per forward
and zero cuBLAS calls: the opponent's whole mechanism is 116 launches -> 5.
Our graph replay collapses launches but replays 116 unfused kernels, hence its
0.64 ms floor vs the opponent's 0.33. At seq-1024 the win is algorithmic, not
fusion: SDPA's mem-efficient kernel (3.3 ms/layer) vs materialized
scores+softmax+masking (~6.8 ms/layer); Inductor does not perform that rewrite.

Open at end of session: draft.md with ranked directions (workflow synthesis
done: compile-inside-candidate is the spine, flash-fp32 for the stress shape),
then /humanize:gen-plan and the RLCR loop.
