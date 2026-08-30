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

## 2026-08-30 -- the admissible opponent, and an artifact caught

**Plan converged (one review round, adversarial subagents standing in for the
unavailable Codex CLI -- user-approved substitution).** Decisions recorded in
docs/plan.md: the opponent is the fastest *numerically admissible* compiled
config at official default flags; stress evidence is an official-script sweep
to the memory ceiling plus a labeled off-script chunked comparison; the manual
graph insurance stays gated; targets are direction, not gates.

**M0.2: the admissible ceiling, measured per-case in fresh processes** (27
rows, notes=`M0.2 admissible ceiling`). `reduce-overhead` is the opponent on
every fp32 lane: center 0.368 ms (drift 6.99e-4, PASS), batch-1 0.133 ms,
batch-128 0.683 ms, heads-16 0.777 ms, seq-1024 24.77 ms, wide-1024 7.64 ms.
In bf16/fp16 *no* compiled mode is admissible -- 0.0625/0.0098 drift even at
mode=default, i.e. Inductor's reduced-precision softmax fusion, not Triton GEMM
templates -- so the admissible reduced-precision opponent is eager itself, and
graph-safe (max_abs=0, 2.57x) already leads that lane legally.
`max-autotune --no-allow-tf32` is near-exact (1.4e-6) but never the fastest
admissible option, and craters wide-1024 to 0.44x (IEEE Triton GEMMs vs cuBLAS
TF32).

**A prior conclusion corrected.** The L2 reading "compile does not touch
seq-1024/batch-128" was an artifact: that sweep ran 36 `main()` calls in one
process, tripping dynamo's recompile limit (8 per code object), after which
"compiled" baselines silently ran eager. Fresh-process truth: RO gets 3.15x on
seq-1024 (24.8 ms) and 2.3x on batch-128. Our sdpa lead at seq-1024 is
therefore 1.28x over the admissible opponent, not 4x. Recorded as bitlesson
BL-1 and in the goal tracker's evolution log; the official-grid rows
(52/52 PASS fp32, batch-10000 first numbers: fused-sdpa 2.30x) are unaffected
-- that sweep was uncompiled.

Scoreboard vs the admissible opponent, fp32: behind 1.60x on center
(0.589 vs 0.368), behind ~1.4x batch-1 (0.192 vs 0.133); ahead 1.28x seq-1024,
ahead at batch-10000 (opponent unmeasured there yet). bf16/fp16: legally ahead
everywhere measured. Next: the compile-fused candidate.

## 2026-08-30 -- round 1: the compiled candidates, the flash trade, the table

**compile-fused (kernels/v3_compiled.py).** The fused body as a pure function
of (x, valid_token_mask, packed weights), compiled inside the candidate,
artifact cached in `__dict__` so the strict weight copy never sees it; GEMMs
stay on cuBLAS by mode choice. Two iterations mattered: the first sweep hit
parity on center and lost batch-1 by 1.4x because masks and packing ran
eagerly outside the compiled region -- every uncaptured kernel is paid per
call -- and moving mask construction in-graph (decomposed broadcast form, not
a materialized [B,1,S,S]) flipped it: center 0.337 vs the opponent's 0.368,
batch-1 0.119 vs 0.133, both ahead for the first time. bf16/fp16 fail at
0.0625/0.0098 for every compiled variant including the exact-fp32-softmax
body: the reference quantizes scores to bf16 before its fp32 softmax and
Inductor's fusion skips that rounding -- measured confirmation of the
rounding-point mechanism, and why those lanes stay on graph-safe.

**flash (kernels/v4_flash.py).** Online-softmax Triton attention, padding as
per-row lengths, O(S*d) memory. The IEEE-fp32 variant is correct everywhere it
runs (drift 7-8e-4 at the stress geometry through the unmodified script, to
S=2816) but slow at d_model=1024 -- CUDA-core fp32 is ~4x under TF32 tensor
cores, and the baseline's TF32 GEMMs outrun the saved traffic. The named trade
flash-tf32 (tensor-core dots, fp32 recurrence) survives the official tolerance
on every measured geometry (drift 1.3-1.9e-3) and owns the attention axis:
seq-1024 6.50 ms (12.0x eager, 3.8x the admissible opponent), stress-s2816
80.3 ms (7.6x). Its worst margin: 2.1e-3 abs at batch-10000, rescued by the
rel branch -- recorded, and it does not serve that geometry anyway.

**Shape #14 evidence, per the agreed framing.** The chunked reference
orchestrator reproduces the unmodified script EXACTLY at S=2048 (max diff 0),
then extrapolates: at S=100000 the batch-sliced flash-tf32 forward runs in
25.48 s median (p90 25.66, n=3, peak 36.8 GiB) -- a latency, labeled
OFF-SCRIPT, for a shape whose reference OOMs in the script's own accuracy
phase; the off-script element-wise comparison runs on host memory. Verdict:
max_abs 0.00110, **zero bad elements out of 3,276,800,000**, judged by the
script's own compare_outputs at the official tolerance -- stated as an
off-script agreement, never as an official pass.

**The table (runs/dispatch_table.json).** Calibration over 200+ rows at the
official tolerance admits per geometry on worst-case speedup: compiled-safe-ro
takes the launch-bound group (worst cases 2.1-12.6x), compiled-sdpa takes
batch-10000 (3.10x) and wide-1024 (1.24x), flash-tf32 takes seq-1024 (12.0x)
and the stress axis, graph-safe keeps bf16/fp16 (2.4-2.6x) with every
sdpa/compiled failure named in the rejection column. flash-tf32 at wide-1024
is rejected at 1.037x -- below the 1.05 margin -- which is the admission rule
doing its job. The insurance direction (manual multi-slot graphs) never
activated: cudagraph trees held across in-process official sweeps.

## 2026-08-30 -- round 2: the review lands, the table closes every gap

**Round-1 adversarial review: CONTINUE, one P0.** The stale-table fallback
added in round 1 cached the dispatcher as its own delegate and recursed through
forward -- reproduced live by the reviewer. Degradation now happens at
resolution time (`_choose` validates against the live registry) and
`scripts/test_dispatch_fallback.py` holds the regression: a bogus table must
produce baseline output byte-for-byte. The review also caught an unpinned SDPA
call (now pinned to mem-efficient inside the compiled region; re-measured
byte-identical numerics, 3.12x at batch-10000), a cherry-picked center
headline, and a missing fidelity-table artifact (now filled into
docs/precision-budget.md from the CSV, compile-baseline rows excluded exactly
as calibration excludes them).

**The opponent as a candidate.** Three geometries still ran faster under the
compiled baseline than under any restructured body -- so `compiled-base-ro`
compiles the baseline's own forward inside the candidate, and dispatch now
serves the opponent's exact program where the opponent wins: batch-128 0.662 ms
(2.37x worst-case), heads-16 0.808 (3.67x), wide-1024 (1.245x worst-case).
"Never slower than the fastest admissible compiled configuration" is now a
table property, not a search outcome.

**Variant lanes measured, not argued.** A padded case for every runnable
official geometry (distinct case names, so calibration's latest-row-wins adds
them to the group instead of overwriting the dense row) plus bf16/fp16 lanes on
a representative spread: 31 rows, all PASS. The padded-and-dense admission
guarantee is a measurement now. One honest casualty: graph-safe's heads-16-bf16
worst case is 1.049x -- below the 1.05 margin -- so that lane falls back to the
baseline path rather than shipping a 4.9% win the rule cannot vouch for.

**The margins, restated at reproducible values.** center: 0.362 ms across three
fresh processes (spread 0.0012) vs the opponent's 0.368 -- parity to a +1.7%
edge, at the plan's own noise line; the decisive edges are batch-1 (0.113 vs
0.133, +18% and 12.3x over eager), seq-1024 (6.50 vs 24.77, 3.8x), batch-10000
(104.7 vs eager 326.7; the opponent was never measured there), and the stress
axis, where the official script now verifies the flash kernel to S=3072
(7.96x, 90.9 ms vs 723 ms -- the "marginal" estimate measured solid; S=4096 at
~68 GB of reference scores stays arithmetic, not an attempt). Final dispatch
validation: 13/13 dense + 3 padded spot-checks PASS, worst case 1.232x.
