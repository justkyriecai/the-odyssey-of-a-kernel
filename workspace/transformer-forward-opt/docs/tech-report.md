# Tech Report (draft) -- A Transformer Layer, Faster Than Its Own Compiler

TikTok TechJam 2026, Track 3. Draft assembled from the campaign record
(`docs/campaign-log.md`, `runs/`); every number cites the organizer's
unmodified script (`md5 21584e59...`) at the official tolerance
(abs<=0.002 OR rel<=0.02, zero bad elements) unless labeled off-script.

## 1. Environment

RTX 6000 Ada (sm_89, 142 SMs, 48 GB), driver 580.126.20, CUDA 12.8,
torch 2.8.0+cu128, RunPod Secure Cloud, 192-core host, Python 3.12.
Hardware profiler counters are denied on this pod (`ERR_NVGPUCTRPERM`);
all profiling evidence is nsys timelines and torch.profiler, named as such.
Measured ceilings (never spec sheets, `runs/ceilings.json`): TF32 GEMM
70.7 TFLOPS, IEEE fp32 GEMM 30.3 TFLOPS, device copy 803 GB/s.

## 2. What the problem actually is

The eager baseline runs ~116 kernels per forward at the grid's center shape,
median kernel 4-11 us -- the cost of a launch (nsys,
`runs/profile/00-regime-anatomy/`). The grid sweeps one factor at a time, so it
contains four different problems wearing one config format:

| Regime | Shapes | Bottleneck |
|---|---|---|
| Launch-bound | center, batch 1-128, seq-32, narrow-32, heads 1-16 | kernel count |
| Attention-bound | seq-1024 | materialized S x S scores |
| GEMM-bound | wide-1024 | cuBLAS is already near the roof |
| Memory-impossible | stress S=100000 | reference needs 12.8 TB of scores |

## 3. The opponent, corrected twice

`--compile-baseline --compile-mode max-autotune` is one flag away in the
organizer's script and rewrites the baseline into ~5 fused Triton kernels
(0.33 ms center vs eager 1.40) -- but it FAILS the script's own tolerance
(max_abs 0.0053 fp32 / 0.0625 bf16: Triton TF32 template rounding and Inductor
reduced-precision softmax fusion). The honest opponent is the fastest
*numerically admissible* configuration -- `reduce-overhead` -- measured per
case in fresh processes: center 0.368 ms, batch-1 0.133, seq-1024 24.8. In
bf16/fp16 no compiled configuration is admissible at all, at any mode.

The "corrected twice": our first opponent measurement ran all cases in one
process and silently degraded to eager after dynamo's recompile limit --
recording eager numbers as "compiled". Fresh-process-per-case is the protocol
that survived review, and the trap is recorded (`.humanize/bitlesson.md` BL-1).

## 4. What we ship

A dispatch layer (organizer-endorsed: shape checks in forward) routing each
geometry to the candidate that won it under a conservative admission rule:
correct on every case in the group (dense AND padded, measured), worst-case
speedup over 1.05, else fall back to the baseline path.

- **compiled-safe-ro** -- our fused body (one QKV GEMM/layer, masks built
  in-graph in decomposed broadcast form, exact fp32-softmax semantics) under
  `torch.compile(mode=reduce-overhead)` *inside the candidate*: cuBLAS GEMM
  numerics (drift 7e-4), cudagraph trees. Serves the launch-bound group:
  batch-1 0.113 ms (12.3x eager, +18% over the opponent).
- **compiled-sdpa** -- the same compiled body with SDPA attention pinned to the
  mem-efficient backend. Serves batch-10000 (104.7 ms, 3.1x) where the win is
  never materializing 2.6 GB of scores per layer.
- **compiled-base-ro** -- the baseline's own forward compiled inside the
  candidate: the admissible opponent as a servable program. Serves batch-128,
  heads-16, wide-1024 -- so "never behind the opponent" holds by construction.
- **flash-tf32** -- a Triton online-softmax attention kernel (fp32 softmax
  recurrence, TF32 tensor-core dots, padding as per-row lengths, O(S*d)
  memory). Serves seq-1024 at 6.50 ms (12.0x eager, 3.8x opponent) and the
  stress axis: verified by the unmodified script to S=3072 (7.96x), and at
  S=100000 -- unrunnable for the reference on any hardware -- 25.5 s with an
  off-script chunked comparison (orchestrator proven exact against the script
  at S=2048; 0 bad of 3.28e9). Its sibling flash-fp32 (IEEE dots) is the
  correctness spine and the measured explanation of the trade: IEEE fp32 sits
  at 30 TFLOPS against TF32's 71 on this card.
- **graph-safe** -- manual CUDA-graph capture over the eager-exact fused body.
  Serves bf16/fp16 at max_abs = 0, because no compiled path is admissible
  there: the reference rounds scores to bf16 *before* its fp32 softmax, and
  Inductor's fusion skips that rounding (measured: 0.0625 at every mode).

## 5. Results

Official grid through the organizer's script, dispatch serving, official
tolerance, median (p90 in the CSV): worst case 1.23x, best 12.3x; table in
README. Roofline (`docs/assets/roofline.png`, measured roofs, analytic
intensity -- byte counters denied): wide-1024's served point reaches 77% of
the measured TF32 GEMM roof, which is why its 1.24x is near the attainable
limit; the launch-bound group's distance to the roof is launch overhead, not
bandwidth. Precision budget (`docs/precision-budget.md`): shipped fp32 lanes
spend 0.45-0.87 of the absolute budget; reduced-precision lanes spend 0.00.

## 6. AI-assisted method (the bonus criterion)

The campaign ran inside the odyssey harness: phase prompts, an
evaluator-patching verify layer (`verify.py` -- the organizer's `main()`, its
exit codes, appended to an append-only CSV with git/md5 provenance), a
solutions DAG with rejected branches recorded, and an implement->measure->
review loop (RLCR) with adversarial fresh-context reviewers at round
boundaries. The reviewers materially changed the outcome: they caught the
recompile-limit measurement corruption, a dispatch fallback that recursed, a
cherry-picked headline (0.337 restated to the reproducible 0.362), and an
unpinned SDPA backend. 16 DAG nodes, ~350 benchmark rows, 5 bitlessons.

## 7. Limitations and what we would do with more time

- Hardware counters denied on the rented pod: stall reasons, DRAM bytes and
  occupancy are absent; the roofline's intensity axis is analytic. A box with
  counters (or `CAP_SYS_ADMIN`) upgrades this evidence.
- The dispatch table is calibrated per card; on other hardware it degrades
  safely to the baseline path but forfeits the wins until recalibrated
  (`python kernels/dispatch.py calibrate`).
- input_scale beyond 1.0 is a single smoke point, not a swept axis.
- heads-16-bf16 falls back to baseline: graph-safe's 1.049x sits under the
  1.05 admission margin -- a 4.9% win the rule refuses to vouch for.
- The three-arm skill ablation and a second kernel through the same phases
  (methodology deliverables) require fresh sessions and are not part of this
  run's evidence.

## Appendix: reproduce

```bash
./scripts/smoke.sh                       # CPU correctness gate, seconds
python verify.py --list                  # candidates + script md5
python verify.py dispatch --shapes official --record -- --atol 0.002 --rtol 0.02
python scripts/stress_100k.py --self-check && python scripts/stress_100k.py flash-tf32
python scripts/measure_ceilings.py && python scripts/plot_roofline.py
```
