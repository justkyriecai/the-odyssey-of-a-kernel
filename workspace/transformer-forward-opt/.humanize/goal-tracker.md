# Goal Tracker -- transformer-forward-opt Phase 2

## IMMUTABLE SECTION (set Round 0, never edited)

### Ultimate Goal
Beat the fastest numerically admissible torch.compile configuration of the
baseline (official default flags) on the launch-bound group; keep the measured
lead on the attention group; complete official-grid evidence incl. the stress
geometry to its memory ceiling; ship a calibrated dispatch. Targets are
optimization direction (DEC-4). Correctness: official tolerance, zero bad
elements, unmodified organizer script, uncompiled eager reference.

### Acceptance Criteria (from docs/plan.md)
- AC-1 admissible-compile ceiling measured & recorded (fresh-process per case)
- AC-2 compile-fused candidate exists, admissible (smoke, per-dtype table, state hygiene, no timed-region compile)
- AC-3 compile-fused beats graph-safe on launch-bound group (worst-case rule, kern_sum anchored)
- AC-4 compile-sdpa serves fp32 lanes it wins, pinned backend, via dispatch
- AC-5 stress evidence per DEC-2 (official sweep to ceiling; off-script chunked comparison labeled; no pass claim)
- AC-6 dispatch calibrated & safe (KeyError fallback, tolerance filter, padded/dtype variants)
- AC-7 record complete (nodes per iteration, profiles per direction, campaign log, n005 corrected)

### Loop Substitution Note
Codex CLI unavailable; per explicit user decision the reviewer seat is filled
by fresh-context adversarial Claude subagents. Round semantics unchanged.
User's framing principle (quiz, 2026-08-30): official requirements govern;
where unspecified, choose the framing that best shows the approach's strengths.

## MUTABLE SECTION

### Active Tasks (from plan Task Breakdown)
- [x] task1 M0.2 per-case ceiling + verdicts  (DONE pre-round: 27 rows recorded)
- [x] task2 M0.3 solutions.jsonl correction node (n006, n007)
- [x] task3 bookkeeping: dispatch KeyError fallback + tolerance filter + verify.py global-state tripwire (padded/dtype grid measurement deferred to task13)
- [x] task4 D1-I1: kernels/v3_compiled.py -- pure-function body, plain-__dict__ artifact cache, cache_size_limit=64 at import, CPU eager fallback; smoke PASS
- [x] task5 D1-I2: dev sweep done. fp32 PASS everywhere; center parity with opponent (0.374 vs 0.368), seq-1024 1.41x ahead (compiled-sdpa 17.60ms), batch-1/batch-128 behind (eager pre-step tax). bf16/fp16 FAIL all variants (rounding-point mechanism confirmed).
- [x] task6->I3 done (masks in-graph; ahead of opponent on center+batch-1). Original TF32 experiment: TF32 experiment answered by M0.2 (max-autotune-notf32 never fastest); iteration spent instead on moving masks into the compiled region + decomposed mask form. Re-sweep RUNNING.
- [ ] task6 D1-I3 TF32-legalization experiment
- [x] task7: cudagraph trees held across 39-case in-process official sweep; D3 gate = NOT ACTIVATED, rejected (n012)
- [ ] task8 D1-I5 per-dtype fidelity table
- [x] task9: compiled-sdpa official grid 13/13 PASS; wins batch-10000 (3.10x) + wide-1024 (1.24x)
- [x] task10 D4 flash I1+I2: kernel with per-row lengths handles causal+padding, IEEE fp32 dots; CPU fallback smoke 10/10; first GPU run RUNNING
- [~] task11: stress sweep done (correct to S=2816); 100k latency 25.5s recorded; chunked comparison rerun in flight
- [ ] task12 D4 I5 sm_89 tuning
- [~] task13: table calibrated (every geometry served); dispatch official validation pending GPU
- [ ] task14 campaign log + profile reports  (analyze tag -> adversarial subagent review)

### Completed Items
- task1 (M0.2b): opponent = reduce-overhead. fp32: center 0.368ms, batch-1 0.133,
  batch-128 0.683, heads-16 0.777, seq-1024 24.77, wide-1024 7.64. bf16/fp16:
  NO admissible compiled config (0.0625/0.0098 at every mode) -> admissible
  reduced-precision opponent is eager itself; graph-safe already leads legally.
  max-autotune-notf32: near-exact (1.4e-6) but never fastest; wide-1024 0.44x.

### Plan Evolution Log
- R1-1: D1-I3 re-scoped from the TF32-legalization experiment to compile-boundary
  repair. Justification: M0.2 already measured max-autotune-notf32 (near-exact but
  never fastest, 0.44x on wide-1024) -- the experiment's question is answered; the
  measured deficit (batch-1 0.69x) points at the eager pre-step instead.
- R0-1: L2's "compile==eager on seq-1024/batch-128" was a measurement artifact
  (dynamo recompile limit in the 36-main in-process sweep silently degraded
  later cases to eager). Corrected by M0.2b: RO baseline seq-1024 24.8ms
  (3.15x), batch-128 0.683ms. Consequence: our sdpa lead at seq-1024 is 1.28x
  vs the admissible opponent, not 4x; plan wording "keep the 4x" now reads
  "keep the measured lead" -- targets unchanged in kind (DEC-4 direction).

### Deferred Items
- D3 graph-multislot: gated on demonstrated D1-I4 cudagraph failure (DEC-3).
- D5 bf16 contest: resolved by M0.2b -- no admissible compiled bf16 exists;
  graph-safe keeps the lane; only a rounding-point-faithful fused kernel could
  contest, parked unless D1's bf16 numbers surprise.
