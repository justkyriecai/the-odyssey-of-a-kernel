# Round 1 Summary

Round semantics: all plan tasks believed complete (or formally closed). This
summary is the review-gate artifact; the reviewer verdict decides round 2.

## What was implemented

- kernels/v3_compiled.py: torch.compile inside the candidate over a pure-function
  fused body; artifact in __dict__ (never a submodule); masks built in-graph in
  decomposed broadcast form; cuBLAS GEMMs by mode choice; dynamo cache limit
  raised at import. Candidates: compiled-safe(-ro), compiled-sdpa(-ro).
- kernels/v4_flash.py: Triton online-softmax attention; padding as per-row
  lengths; O(S*d) memory; block-size heuristic for sm_89 smem; batch-slicing for
  oversized working sets. Candidates: flash-fp32 (IEEE, correctness spine),
  flash-tf32 (named trade, tensor-core dots + fp32 recurrence).
- scripts/stress_100k.py: off-script latency + chunked-reference comparison for
  shape #14, gated by an exact self-check vs the unmodified script at S=2048.
- Harness: dispatch stale-name fallback, calibrate tolerance filter, verify.py
  global-state tripwire, ladder verdict-not-abort, bench/shapes/stress-sweep.json.

## Measured outcomes (medians, official tolerance, eager denominator unless said)

- Admissible opponent (M0.2, fresh-process): reduce-overhead baseline; fp32
  center 0.368ms, batch-1 0.133, seq-1024 24.77. No admissible compiled config
  exists at bf16/fp16 (0.0625/0.0098 at every mode).
- vs opponent: center 0.337 (1.09x ahead), batch-1 0.119 (1.12x), seq-1024 6.50
  (3.8x), batch-10000 105.5 (opponent unmeasured there; 3.10x vs eager).
- Official grid: 39/39 PASS for new candidates; dispatch table serves every
  geometry (worst-case rule; flash-tf32 rejected at wide-1024 by margin).
- Shape #14: OFF-SCRIPT latency 25.52s median (peak 36.8 GiB); comparison vs
  exact-at-S2048 chunked reference: max_abs 0.00110, 0 bad of 3.28e9.
- bf16/fp16 lanes: graph-safe (max_abs=0), rivals' failures named in the table.
- Rounding-point mechanism CONFIRMED: every compiled variant fails bf16 at
  0.0625 even with fp32-softmax source (Inductor skips the reference's bf16
  score quantization).
- Dispatch validation on the official grid: 13/13 PASS; worst case wide-1024
  1.243x, best batch-1 12.07x; routing overhead unmeasurable (center via
  dispatch 0.3618 vs direct 0.362). "Never slower than the baseline" holds on
  every runnable official shape.

## Deviations from plan (all logged in goal-tracker Plan Evolution)

- D1-I3 re-scoped: TF32-legalization experiment answered by M0.2 before it ran;
  iteration spent on compile-boundary repair instead (decisive).
- D3 insurance: never activated, formally rejected (n012), zero iterations.
- "keep the 4x at seq-1024" re-read as "keep the measured lead" after the L2
  artifact correction; final lead is 3.8x over the admissible opponent.
- D4-I5 (block tuning) held in reserve: flash-tf32 already wins its lanes.

## Evidence index

- runs/benchmark.csv: ~240 rows, every one at a stated tolerance with flags.
- runs/solutions.jsonl: n001-n014, parent-linked, rejections included.
- runs/profile/00-regime-anatomy/ + 01-round1/ (kern_sum text exports, nsys).
- runs/dispatch_table.json; docs/campaign-log.md (5 entries); commits
  a8af3f1..40aba48 on main.

## BitLesson Delta

Action: add -- BL-1 (in-process compile sweeps degrade silently), BL-2
(compiled references are not correctness references), BL-3 (OMP thread
explosion on many-core CPU gates) were added this round; BL-4 candidate:
"everything outside the compiled region is paid per call" (the I2->I3 lesson).

## Known gaps for the reviewer

- Padded/dtype variant coverage per served geometry is partial (center-padded +
  stress-padded measured; other geometries' padded lanes rely on the shared
  code path argument, not rows).
- flash-tf32 batch-10000 margin is thin (2.1e-3 abs, rescued by rel branch);
  it does not serve that geometry, but the row exists.
- input_scale axis unmeasured beyond the smoke case.
