# Regime anatomy -- nsys timelines, D0

**Tool: nsys 2024.6.2 (CUPTI timeline). Hardware counters denied on this pod
(ERR_NVGPUCTRPERM), so no stall reasons, no DRAM bytes, no occupancy -- every
verdict below rests on kernel counts, kernel durations and API time only.**

Protocol: `verify.py <candidate> --shapes dev --case <case> -- --atol 0.002
--rtol 0.02 --accuracy-trials 1 --warmup 5 --repeats 20 --benchmark-rounds 1`,
so each model executes exactly 26 forwards (1 accuracy + 5 warmup + 20 timed).
`passthrough` runs the baseline in both slots: 52 baseline forwards, nothing
else. Text exports (`*.kern_sum.csv`, `*.api_sum.csv`) are committed; binary
`.nsys-rep` files are not. CUDA graph replays are traced in "graph" mode: a
replay counts one API event, and its interior kernels are not re-counted in
kern_sum -- per-forward kernel counts are therefore quoted only for non-graphed
runs.

## Launch-bound verdict for the center shape (B=64, S=128, d=128, L=4)

- `center-eager`: 6017 kernel instances / 52 forwards = **~116 kernels per
  forward**. Median kernel duration 4-11 us -- the same order as a kernel
  launch. GEMMs are cuBLAS/cutlass (`cutlass_80_tensorop_s1688gemm`, 28 per
  forward), 31% of GPU time; LayerNorm 14%; elementwise glue ~28%.
- `center-compiled` (the opponent, `--compile-baseline --compile-mode
  max-autotune`): the eager candidate slot accounts for 26 x 116 = 3016 of the
  3145 instances and **all 728 cutlass instances (26 x 28) come from it** --
  the compiled baseline runs **zero cuBLAS kernels**. Remainder: ~137 instances
  over 26 forwards = **~5 kernels per compiled forward**. Inductor rewrote the
  whole 4-layer stack into a handful of fused Triton kernels.

The opponent's mechanism on the launch-bound end, in one line: **116 launches
becomes ~5**. That is the entire 1.40 -> 0.33 ms on center and 1.43 -> 0.107 ms
on batch-1 (where our graph-sdpa replay of ~50 eager-shaped nodes still beats
launch overhead but replays unfused kernels: 0.172-0.19 ms).

## Attention regime (seq-1024)

`seq1024-fused-sdpa`: the SDPA candidate serves attention with
`fmha_cutlassF_f32_aligned_64x64_rf_sm80` (mem-efficient backend; fp32 has no
flash path) at 3.31 ms median per layer. The eager baseline pays, per layer:
2.73 ms softmax + 1.40 ms scores GEMM + 2.68 ms of masked_fill/elementwise on
the materialized [64,4,1024,1024] scores. Removing the S x S materialization is
the whole 4.08x, and Inductor does not perform this rewrite: the compiled
baseline's seq-1024 median (78.0 ms) is indistinguishable from eager (78.1 ms).

## Where each mechanism stops working

- Compiled fusion: owns tiny shapes; worthless at seq-1024/batch-128/wide-1024
  (medians within noise of eager, ladder L2 rows).
- Our CUDA graph replay: kills launch count but replays ~116 unfused kernels;
  its GPU-busy floor is fused-safe's kernel time, which is why graph-safe
  plateaus at 0.64 ms center where the 5-kernel opponent reaches 0.33 ms.
- SDPA: owns S>=512 and head_dim-8 shapes; numerically ineligible at bf16/fp16
  (softmax not accumulated in fp32 on those paths: 0.0625 / 0.0098 max_abs).
