# transformer-forward-opt: session report (2026-08-30, RTX 6000 Ada, sm_89)

Session budget ~30 min. All numbers below are the organizer's script's own
output, run through `verify.py <candidate> --shapes dev --record -- --atol
0.002 --rtol 0.02` (script md5 21584e5923680ce0455554bd0b45bda2, torch
2.8.0+cu128, driver 580.126.20). Everything is recorded in
`workspace/transformer-forward-opt/runs/benchmark.csv` (53 rows) and
`runs/solutions.jsonl` (8 nodes). Medians; p90 quoted for the shipped
candidate.

## 1. What was measured (verify.py verdicts, dev shape set)

Control: `passthrough` on `center`: PASS, 1.001x, max_abs 0 — the harness
measures cleanly.

Full sweep of the four prior candidates vs the **eager** baseline (verdict /
speedup / max_abs):

| case          | fused-safe     | fused-sdpa        | graph-safe     | graph-sdpa        |
|---------------|----------------|-------------------|----------------|-------------------|
| center        | PASS 1.327x 0  | PASS 1.706x 9.9e-4| PASS 1.995x 0  | PASS 2.746x 9.9e-4|
| center-bf16   | PASS 1.322x 0  | FAIL 0.0625       | PASS 2.587x 0  | FAIL 0.0625       |
| center-fp16   | PASS 1.313x 0  | FAIL 5.9e-3       | PASS 2.594x 0  | FAIL 5.9e-3       |
| center-padded | PASS 1.316x 0  | PASS 1.700x       | PASS 2.005x 0  | PASS 2.480x       |
| batch-1       | PASS 1.394x    | PASS 1.665x       | PASS 8.313x    | PASS 7.431x       |
| batch-128     | PASS 1.026x 0  | PASS 1.387x 1.2e-3| PASS 1.022x 0  | PASS 1.392x 1.2e-3|
| wide-1024     | PASS 1.011x 0  | PASS 1.093x 1.1e-3| PASS 1.007x 0  | PASS 1.089x 1.1e-3|
| heads-16      | PASS 1.218x 0  | PASS 2.733x 1.1e-3| PASS 1.267x 0  | PASS 2.890x 1.1e-3|
| seq-1024      | PASS 1.332x 0  | PASS 4.046x 1.1e-3| PASS 1.331x 0  | PASS 4.037x 1.1e-3|

Shipped candidate: `dispatch`, after `python kernels/dispatch.py calibrate`
built `runs/dispatch_table.json` from those rows. Verified end to end vs the
**eager** baseline (median ms, p90 in parens):

| case          | verdict | speedup | max_abs   | baseline ms       | dispatch ms       |
|---------------|---------|---------|-----------|-------------------|-------------------|
| center        | PASS    | 2.498x  | 9.9e-4    | 1.4451 (1.4672)   | 0.5785 (0.5815)   |
| center-bf16   | PASS    | 2.689x  | 0         | 1.5387 (1.5577)   | 0.5722 (0.5890)   |
| center-fp16   | PASS    | 2.551x  | 0         | 1.5367 (1.5593)   | 0.6023 (0.6187)   |
| center-padded | PASS    | 2.531x  | 9.9e-4    | 1.4556 (1.4736)   | 0.5750 (0.5809)   |
| batch-1       | PASS    | 8.306x  | 5.9e-4    | 1.4326 (1.4676)   | 0.1725 (0.1726)   |
| batch-128     | PASS    | 1.408x  | 1.2e-3    | 1.5912 (1.5963)   | 1.1299 (1.1721)   |
| wide-1024     | PASS    | 1.092x  | 1.1e-3    | 9.5425 (9.8847)   | 8.7385 (8.9319)   |
| heads-16      | PASS    | 2.854x  | 1.1e-3    | 2.9323 (2.9462)   | 1.0275 (1.0374)   |
| seq-1024      | PASS    | 4.112x  | 1.1e-3    | 78.0563 (78.1331) | 18.9825 (19.5060) |

9/9 PASS at the problem statement's tolerance, zero bad elements, never slower
than the eager baseline (worst case 1.09x on wide-1024).

Vs the **compiled** baseline (`--compile-baseline --compile-mode max-autotune`,
one case per process because of the dynamo recompile-limit trap; timing taken
with `--benchmark-on-failure`):

| candidate   | case     | verdict | speedup | max_abs | compiled-base ms | opt ms  |
|-------------|----------|---------|---------|---------|------------------|---------|
| passthrough | center   | FAIL    | 0.210x  | 5.3e-3  | 0.3077           | 1.4634  |
| dispatch    | center   | FAIL    | 0.588x  | 5.7e-3  | 0.3406           | 0.5794  |
| dispatch    | seq-1024 | FAIL    | 1.326x  | 5.5e-3  | 24.3590          | 18.3686 |

## 2. What was implemented / changed

Nothing in `kernels/` was modified — the prior candidates measured well once
actually run on a GPU (they had never been; the workspace README said so).
This session produced:

- The first GPU measurement campaign: 53 recorded rows in `runs/benchmark.csv`
  covering all five candidates plus dispatch on the full dev set.
- `runs/dispatch_table.json`, calibrated from those rows. The table routes:
  fp32 small/medium geometries -> graph-sdpa; batch-1 -> graph-safe; bf16/fp16
  -> graph-safe (sdpa fails half precision on this card, exactly as the
  workspace docs predicted for some backends); seq-1024 and wide-1024 ->
  fused-sdpa; everything unproven -> baseline fallback.
- `runs/solutions.jsonl`: 8 nodes with parent links, rejected/parked branches
  included.

## 3. Tried and abandoned / bounded

- **fused-sdpa / graph-sdpa on half precision: rejected for those geometries**
  (recorded `fail` rows). On this card SDPA drifts to max_abs 0.0625 (bf16) and
  5.9e-3 (fp16) against the baseline's fp32-softmax numerics; the
  zero-bad-element rule kills it. Dispatch routes bf16/fp16 to graph-safe
  (zero error by construction, 2.55-2.69x) instead. Did not attempt an
  fp32-cast-SDPA variant: the fp32 attention GEMMs would land near graph-safe's
  number anyway, and budget went to the compiled-baseline question.
- **Beating the max-autotune compiled baseline as a *verdicted* result:
  parked with evidence.** The probe that settles it: `passthrough` — the
  organizer's own eager forward, bit-identical to the reference definition —
  FAILS accuracy against the compiled reference (max_abs 5.3e-3 > atol 2e-3).
  Under `--compile-baseline` the accuracy reference and the speed opponent are
  the same drifted model, so no candidate with eager numerics can pass that
  regime on this card, independent of speed. The speed comparison is still
  real: dispatch is 0.59x of the compiled baseline at the launch-bound center
  shape (the compiled baseline is 4.2x faster than eager there — cudagraphs +
  fused elementwise) but 1.33x faster at seq-1024 where attention is
  O(S^2)-bound and SDPA's flash path beats Inductor's materialized softmax.
- **Not attempted for budget**: a torch.compile'd candidate (to reproduce
  compiled numerics and re-enter that regime), the official 14-shape set, and
  the batch-128/wide-1024 compute-bound weak spots (1.41x / 1.09x).

## 4. Unverified, stated plainly

- Only the **dev** shape set was run. Nothing here is a number for the
  official 14 appendix shapes; in particular stress-100k (S=100000) was never
  attempted and the dispatch table has no entry for any official geometry
  beyond those sharing dev's center geometry. Per the dev set's own note:
  promote nothing on dev alone.
- The dispatch table was calibrated from single-sweep rows (one run per
  candidate per case, though each run's median is over 300 timed iterations
  with the script's alternating-order protocol). No repeat-run stability check
  was done.
- No profiles were captured (ncu denied on this machine; nsys/torch.profiler
  fallback not exercised within budget), so the launch-bound vs compute-bound
  attributions above are inferences from the shape sweep, not counter or
  timeline evidence.
- The compiled-baseline drift probe was run at center and seq-1024 only; the
  claim that the compiled regime is unsatisfiable for eager numerics is proven
  at those two shapes, assumed elsewhere.
