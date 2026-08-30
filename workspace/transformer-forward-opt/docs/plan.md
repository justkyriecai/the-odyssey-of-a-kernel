# Phase 2 Plan -- Beat the Admissible Compiled Baseline, Keep the 4x, Complete the Grid

## Goal Description

Beat the fastest *numerically admissible* `torch.compile` configuration of the
baseline (official default flags: TF32 on, reference untouched -- DEC-1) on the
launch-bound shape group; keep the >=4x lead on the attention group; complete
official-grid evidence including the stress geometry to its memory ceiling; ship
a calibrated dispatch. Targets are optimization direction, not hard gates
(DEC-4): reach them or produce benchmark+profile evidence why this round cannot.
Correctness bar everywhere: official tolerance (abs<=0.002 OR rel<=0.02), zero
bad elements, unmodified organizer script, uncompiled eager reference.

Context that motivated the correction: the draft's numeric targets
(0.33/0.23/0.107 ms) all came from `max-autotune` runs that FAIL the official
tolerance (max_abs 0.0053/0.0625/0.0038). They remain reported as context; the
real bar is measured by M0.2.

## Acceptance Criteria

- AC-1: The admissible-compile ceiling is measured and recorded (M0.2).
  - Positive: `runs/benchmark.csv` holds fresh-process rows for
    {default, reduce-overhead, max-autotune --no-allow-tf32} x 9 dev cases with
    verdicts; the fastest admissible config per group is named in
    `docs/campaign-log.md`.
  - Negative: any M0.2 row produced by a multi-case in-process sweep (dynamo
    recompile limit) is absent or superseded; no target anywhere cites an
    inadmissible run as the bar.
- AC-2: A `compile-fused` candidate exists and is admissible.
  - Positive: passes `smoke.sh` (strict state-dict load intact, no
    OptimizedModule submodule); dev sweep recorded at official tolerance for
    fp32/bf16/fp16 with a per-dtype max_abs table in its solutions.jsonl node;
    global torch.backends flags restored after every forward (declared
    exemption: `torch._dynamo.config.cache_size_limit`); benchmark medians
    round-1 vs round-3 within noise (compile landed in accuracy phase).
  - Negative: a run where the compiled artifact mutates `allow_tf32`/matmul
    precision visible to the reference fails the smoke assertion; a 14-case
    in-process sweep does not silently record eager numbers (recompile limit
    raised in the factory).
- AC-3: `compile-fused` beats `graph-safe` on the launch-bound group, measured.
  - Positive: center/batch-1..16/seq-32/narrow-32/heads-1,2 medians below
    graph-safe's with p90 recorded; nsys kern_sum shows kernels/forward at or
    under the admissible opponent's count (anchored by M0.2's kern_sum, not the
    draft's guess).
  - Negative: a win only on center with regressions elsewhere in the group is
    not admitted (worst-case rule); deltas <1.5% are noise, not wins.
- AC-4: `compile-sdpa` serves fp32 lanes it wins, through dispatch.
  - Positive: fp32 dev + official rows recorded with pinned backend
    (`sdpa_kernel`, mem-efficient on this card for fp32+mask); wins at
    seq-1024/heads-16/batch-10000 vs both graph-sdpa and compile-fused
    measured head-to-head; fidelity table includes its lanes.
  - Negative: bf16/fp16 lanes never route to any SDPA path (measured out:
    0.0625/0.0098); an unpinned backend choice is not shipped.
- AC-5: Stress geometry evidence per DEC-2.
  - Positive: official-script rows at B=32,H=16,d=1024,L=2 with S swept to the
    memory ceiling (expected ~S=2048 solid, ~3000 marginal, from ~4096*S^2
    bytes peak on 48GB); at S=100000 a latency measurement plus an off-script
    chunked comparison built on the vendored script's weights and
    `compare_outputs`, cross-validated against the full script at S<=2048,
    explicitly labeled off-script; a peak-memory audit.
  - Negative: no "pass" claim for shape #14 anywhere; the flash kernel never
    inherits `_keep_mask` (10 GB triangle at S=100k).
- AC-6: Dispatch is calibrated and safe.
  - Positive: `dispatch.py` stale-name lookup falls back to the baseline path
    (no KeyError); `calibrate()` filters rows by tolerance pair; calibration
    covers padded + dtype variants for every geometry a candidate serves;
    `verify.py dispatch --shapes official` records >=1.0x everywhere.
  - Negative: a candidate with a FAIL row anywhere in a geometry group is not
    admitted for that group.
- AC-7: The record is complete per harness discipline.
  - Positive: every iteration lands a solutions.jsonl node (keep/reject/park,
    parent-linked); every direction keeps a profile under `runs/profile/`
    naming the tool; campaign-log entries cite rows/commits; n005's 0.572 ms
    context is corrected by a follow-up node.
  - Negative: no direction exceeds 5 iterations (D3 capped at 3, gated on a
    demonstrated D1 cudagraph failure -- DEC-3).

## Path Boundaries

### Upper Bound
All five directions executed to their iteration caps with per-iteration
recorded evidence; compile-fused and compile-sdpa both admitted into dispatch;
flash kernel tuned for sm_89; dispatch validated on the official grid; campaign
log and profile directories complete for the tech report.

### Lower Bound
M0.2 recorded and targets re-anchored; compile-fused correct at all dtypes and
faster than graph-safe on the launch-bound group; dispatch fixed (KeyError,
tolerance filter) and recalibrated over existing + new rows; official-grid
validation run; every attempted direction has its nodes, including rejections.

### Allowed Choices
- Can use: torch.compile in any mode inside candidates; manual CUDA graph
  capture; Triton kernels; SDPA with pinned backends; per-shape dispatch;
  fresh-process-per-case measurement.
- Cannot use: modifications to the vendored evaluator; global state visible to
  the reference left altered; SDPA on reduced-precision lanes; a speed/accuracy
  trade hidden as a flag (separate named candidate only); goalpost edits
  without a recorded decision.

## Feasibility Hints and Suggestions

- D1-I3 mechanism (verified in torch 2.8 on this box): Inductor bakes
  `ALLOW_TF32` into Triton GEMM templates at codegen; dynamo's GlobalStateGuard
  invalidates on an `allow_tf32` flip. So: compile with the flag off, toggle
  off/on around each candidate call (flag sandwich), restore before reference
  code runs. Expectation pre-registered: the TF32-off artifact may drift
  ~1e-3-class vs the TF32 reference or lose speed -- mode=default with cuBLAS
  GEMMs (drift 6.99e-4 in the fresh-process probe) is the favored survivor,
  and I3 confirming that null is a completed iteration, not a burned one.
- `sdpa_kernel` inside a compiled region does not graph-break
  (`SDPAKernelVariable`, torch/_dynamo). fp32+mask on this card selects
  mem-efficient (`can_use_flash_attention`=False, efficient=True; mask is
  always a tensor under the harness: all-ones at padding_ratio 0).
- Compiled artifact storage: plain-dict escape hatch (kernels/dispatch.py
  pattern) so `load_state_dict(strict=True)` never sees it.
- Cold compile lands in the accuracy phase (5 untimed trials precede warmup);
  do not spend an iteration on cache priming.
- Relevant: kernels/v1_fused_attention.py (fused body to restructure),
  kernels/v2_cuda_graph.py (capture protocol for D3),
  runs/profile/00-regime-anatomy/REPORT.md (the 116->5 anatomy).

## Dependencies and Sequence

1. M0 (measure first): M0.1 official grid [DONE, 52/52 PASS fp32; batch-10000:
   fused-sdpa 2.30x], M0.2 per-case ceiling [RUNNING], M0.3 n005 correction.
2. Bookkeeping fixes (dispatch KeyError, tolerance filter, padded/dtype
   calibration variants, smoke global-state assertion) -- before any
   calibration that admits new candidates.
3. D1 compile-fused I1..I5 (restructure -> mode ladder -> TF32-legalization
   experiment -> cudagraph hygiene -> fidelity table). I4 informs whether D3
   activates.
4. D2 compile-sdpa I1..I5 (after D1-I2 gives the compiled body).
5. D4 flash-fp32-stress I1..I5 (independent; can interleave).
6. D5 bf16 contest only after M0.2 verdicts. D3 only on D1-I4 failure.
7. Final: recalibrate dispatch, official-grid validation, campaign log.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | M0.2 per-case ceiling runs + verdict summary | AC-1 | coding | - |
| task2 | M0.3 solutions.jsonl correction node | AC-7 | coding | - |
| task3 | Bookkeeping: dispatch KeyError fallback, calibrate tolerance filter, padded/dtype grid, smoke assertion | AC-6 | coding | - |
| task4 | D1-I1 restructured compile-fused candidate | AC-2 | coding | task1 |
| task5 | D1-I2 mode ladder + recompile-limit hygiene, dev sweeps | AC-2, AC-3 | coding | task4 |
| task6 | D1-I3 TF32-legalization experiment (flag sandwich) | AC-3 | coding | task5 |
| task7 | D1-I4 cudagraph pool/guard hygiene; D3 gate decision | AC-3 | coding | task5 |
| task8 | D1-I5 per-dtype fidelity table (incl. bf16 rounding-point check) | AC-2 | coding | task5 |
| task9 | D2 compile-sdpa fp32 lanes I1..I5 | AC-4 | coding | task5 |
| task10 | D4 flash kernel I1..I2 (S<=1024 correct via script) | AC-5 | coding | - |
| task11 | D4 I3..I4 stress sweep + off-script chunked comparison | AC-5 | coding | task10 |
| task12 | D4 I5 sm_89 block tuning | AC-5 | coding | task11 |
| task13 | Recalibrate dispatch; official-grid validation run | AC-6 | coding | task5, task9 |
| task14 | Campaign log + profile reports per direction | AC-7 | analyze | task13 |

## Claude-Codex Deliberation

Codex CLI unavailable on this machine; per explicit user decision, independent
review was performed by two fresh-context Claude subagents (first-pass
adversarial, then convergence round 1), each verifying claims against the
repository and the installed torch. Reduced external-model diversity is noted.

### Agreements
- M0-before-building; passthrough-vs-compiled as the ceiling protocol; target
  correction away from inadmissible runs; D5 gating; bookkeeping scope; D1-I3
  technical mechanism (verified against torch internals); D2 premises verified
  on-card; stress-shape official rows impossible at S=100k.

### Resolved Disagreements
- Round target anchored to illegal 0.33ms -> re-anchored to M0.2 (DEC-1).
- "Stress row itself" as evidence -> impossible; rescoped to sweep + labeled
  off-script comparison (DEC-2).
- Global-state assertion "never touched" -> "restored after forward", with
  cache_size_limit exempted.
- M0.2 in-process sweep -> per-case fresh processes (the in-flight corrupted
  run was stopped and its rows purged before ever being committed).
- S-ceiling estimate for the stress sweep corrected from ~4000 to ~2048-3000.
- Cold-compile/cache-priming iteration dropped (lands in accuracy phase).

### Convergence Status
- Final Status: `converged` (one round; reviewer's five REQUIRED_CHANGES all
  folded; per user directive iteration count is prioritized over further
  review rounds).

## Pending User Decisions

- DEC-1 opponent definition -- Decision: fastest admissible compiled config
  under official default flags (TF32 on, reference untouched).
- DEC-2 stress evidence -- Decision: official sweep to memory ceiling +
  off-script chunked comparison, labeled, no pass claim.
- DEC-3 D3 insurance -- Decision: keep, capped at 3 iterations, gated on a
  demonstrated D1 cudagraph failure.
- DEC-4 target nature -- Decision: optimization direction, not hard gate;
  "reach it or produce the evidence why not".

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology
  such as "AC-", "Milestone", "Step", "Phase", "D1-I3", or similar workflow
  markers. Candidate names, comments and commit messages use domain language
  (what the code does and why), matching the existing kernels/ style.

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
