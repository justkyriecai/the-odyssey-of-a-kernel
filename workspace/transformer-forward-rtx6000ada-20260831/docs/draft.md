# Phase 2 draft -- beat the compiled baseline where it is strong, keep the 4x where it is absent

Round target, set against measurement (no goalpost movement): **beat
`torch.compile max-autotune` on the launch-bound group** -- center below
0.33 ms fp32 / 0.23 ms bf16, batch-1 below 0.107 ms -- **while keeping the
>=4x at seq-1024 and completing the official-grid map** (batch-4/16/10000,
narrow-32, heads-1/2, seq-32 are unmeasured). Correctness bar: official
tolerance 0.002/0.02, zero bad elements, judged against the uncompiled eager
reference at L0.

## What is known, with evidence

- Eager baseline center: ~116 kernels/forward at 4-11 us median
  (`runs/profile/00-regime-anatomy/`). Launch-bound by count.
- The opponent rewrites the stack into ~5 fused Triton kernels/forward, zero
  cuBLAS (same profile). Center 0.33 ms, batch-1 0.107 ms, bf16 0.23 ms
  (`ladder L2` rows). It does not touch the attention regime: seq-1024 78 ms
  == eager.
- Our graph replay floors at fused-safe's kernel time: 0.64/0.59 ms center
  (`ladder L0`), because it replays 116 unfused kernels.
- SDPA owns seq-1024 (19.2 ms, 4.1x) via `fmha_cutlassF_f32` 3.3 ms/layer vs
  ~6.8 ms/layer materialized; fp32 lanes drift ~1e-3 (passes 0.002), bf16/fp16
  fail (0.0625/0.0098) -- dispatch must route reduced precision to safe paths.
- Numerics of compiling the candidate (probe, D0): compile-default keeps cuBLAS
  GEMMs, drift 6.99e-4 vs eager reference -> PASSES official tolerance;
  max-autotune's Triton GEMMs drift 5.3e-3 under TF32 -> FAILS fp32. The
  compiled bf16 baseline drifts 0.0625 from its own eager numerics, so
  whole-graph Triton attention in bf16 is a known failure shape.
- Noise floor: passthrough reads 0.988-1.004x across 18 rows; treat <1.5%
  as noise. Median with p90, full dev set, always.

## Ranked directions

### 1. compile-fused -- torch.compile inside the candidate, numerics-first
The organizer's own example list includes torch.compile in the candidate.
Compile the fused-safe body (1 QKV GEMM/layer, hoisted masks, exact fp32
softmax); the compiled program is strictly cheaper than what Inductor turned
into 0.33 ms, so parity is the floor.
- Benefit: closes the 1.7x center gap; applies to every launch-bound shape.
- Risk: numerics by mode (measured above); Inductor may pattern-match our
  explicit softmax into SDPA and reintroduce the bf16 failure -- check max_abs
  per dtype, disable the pattern if it fires. Graph breaks from lazy fused-QKV
  build -- build eagerly before the compiled region. Cold-compile time must
  land in warmup, not timed rounds.
- Iterations (cap 5): (1) `mode=default` correct at all dtypes on dev;
  (2) `reduce-overhead` (cudagraph trees; the principled graph-safe);
  (3) `max-autotune` with `max_autotune_gemm_backends` restricted to
  CUBLAS/ATEN -- Triton templates only where calibration passes; (4) dynamic
  shape guards: one compiled artifact per official shape, `dynamic=False`;
  (5) inductor cache priming so verify.py warmup absorbs compilation.
- Evidence: kern_sum before/after (target: <=10 kernels/forward), dev sweep
  rows, per-dtype max_abs table.

### 2. compile-sdpa -- the same compiled body with SDPA attention, fp32 lanes
Compile fuses the LN/FFN/residual glue, SDPA skips the S x S materialization;
the two compose and neither alone covers seq-1024 + center.
- Benefit: seq-1024 has ~6 ms/forward of fusable glue around 13.2 ms of
  attention; center gets SDPA on top of fusion. Also first candidate for
  batch-10000 (2.6 GB of scores per layer never materialized).
- Risk: fp32-lane-only by construction (bf16/fp16 route to direction 1 via
  dispatch); backend choice must be pinned (`sdpa_kernel`), not trusted.
- Iterations: (1) fp32 dev sweep correct; (2) pin backend per geometry;
  (3) seq-1024 + batch-128 wins confirmed vs direction 1; (4) official-grid
  fp32 lanes; (5) merge into dispatch.
- Evidence: seq-1024 kern_sum (glue kernels gone), benchmark rows.

### 3. official-grid completion -- measure before building anything else
batch-4/16/10000, narrow-32, heads-1/2, seq-32 have never been run. The
dispatch table cannot exist without them, and batch-10000 may already be won
by existing SDPA (scores materialization is the plausible bottleneck).
- Benefit: pure evidence; may hand us shapes for free. Zero code.
- Risk: batch-10000 memory (audit peak; 48 GB card); stress-100k excluded
  (script's own baseline cannot run it -- separate direction).
- One iteration: run the four passing candidates + the two compile candidates
  when born, `--shapes official --record` (minus stress-100k), calibrate.

### 4. flash-fp32-stress -- Triton online-softmax causal attention
The stress shape (B=32, S=100000, d=1024, H=16, L=2) cannot run on the
baseline at all (12.8 TB of scores); memory-efficient attention is entry, not
optimization. Requires the off-script chunked reference for correctness.
- Benefit: the only route to any result on shape #14; uncontested. Also a
  candidate at seq-1024 and batch-10000.
- Risk: highest implementation risk (hand kernel, online-softmax rescale,
  3 masking sites); fp32 accumulation is the correctness spine (it is exactly
  why fused-safe passes where SDPA fails).
- Iterations: (1) Triton causal flash, fp32 accumulate, no padding; (2) key
  padding mask; (3) chunked reference harness (off-script, documented as
  such); (4) stress shape end-to-end + memory audit; (5) tune block sizes for
  sm_89 (99 KB smem/SM; flashinfer PR-814 is the Ada reference point).
- Evidence: nsys timeline on a reduced S; measured GB moved vs roofline copy
  ceiling; the stress row itself.

### 5. graph-multislot -- insurance only
Fix v2's single-slot re-capture and hoist mask construction to capture time.
Killed the moment direction 1's cudagraph trees hold on the launch-bound
group. Max 2 of its 5 iterations unless direction 1 fails.

### Parked
- sdpa bf16/fp16 lanes: measured out (0.0625/0.0098). Recorded, not retried.
- Reduced-precision probes (bf16-internal GEMMs etc.): separate named
  candidates, run once for the precision-budget table; prior is rejection.
- Manual Triton whole-layer fusion: only if compile-fused plateaus above the
  opponent; five fresh iterations, new draft.

## How a result is admitted

Every candidate: smoke.sh, then dev sweep recorded at official tolerance, then
official grid before any headline number. Dispatch admits per geometry only on
worst-case speedup >= margin with correctness on every case in the group
(calibration rule as shipped). A speed/accuracy trade is a separate named
candidate. Median with p90; deltas under 1.5% are noise.
