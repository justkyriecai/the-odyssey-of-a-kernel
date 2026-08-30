# Optimization Report: UserOptimizedTransformer (RTX 6000 Ada)

Implementation: `/opt/ablation/arm-bare/opt_transformer.py` (class
`OptTransformerCompiled`, selected via `OPT_COMPILE=1`), patched into the
unmodified benchmark by `/opt/ablation/arm-bare/run_bench.py`. All runs used
`--causal --atol 0.002 --rtol 0.02` and the script defaults otherwise
(fp32, matmul precision "high"/TF32, padding_ratio 0, warmup 20,
repeats 100, rounds 3). Raw logs: `/opt/ablation/arm-bare/runs/*.log`.

## 1. Measured shapes (script's own verdicts and speedups)

All rows: causal=True, fp32, `--atol 0.002 --rtol 0.02`, script defaults
otherwise (warmup 20, repeats 100, rounds 3; medians are the script's).

| B | S | d_model | heads | ffn | layers | Verdict | Baseline med (ms) | Optimized med (ms) | Speedup | Log |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 128 | 4 | 128 | 4 | PASS | 1.4393 | 0.3518 | 4.091x | runs/b64.log |
| 1 | 128 | 128 | 4 | 128 | 4 | PASS | 1.4718 | 0.1136 | 12.950x | runs/b1.log |
| 4 | 128 | 128 | 4 | 128 | 4 | PASS | 1.4949 | 0.1213 | 12.326x | runs/b4.log |
| 16 | 128 | 128 | 4 | 128 | 4 | PASS | 1.4644 | 0.1432 | 10.228x | runs/b16.log |
| 128 | 128 | 128 | 4 | 128 | 4 | PASS | 1.5960 | 0.6965 | 2.291x | runs/b128.log |
| 10000 | 128 | 128 | 4 | 128 | 4 | PASS | 326.8483 | 93.0473 | 3.513x | runs/b10000.log |
| 64 | 128 | 32 | 4 | 32 | 4 | PASS | 1.5034 | 0.2197 | 6.844x | runs/d32.log |
| 64 | 128 | 1024 | 4 | 1024 | 4 | PASS | 9.6891 | 7.6785 | 1.262x | runs/d1024.log |
| 64 | 128 | 128 | 1 | 128 | 4 | PASS | 1.3254 | 0.2672 | 4.960x | runs/h1.log |
| 64 | 128 | 128 | 2 | 128 | 4 | PASS | 1.5064 | 0.2979 | 5.056x | runs/h2.log |
| 64 | 128 | 128 | 16 | 128 | 4 | PASS | 2.9254 | 0.6750 | 4.334x | runs/h16.log |
| 64 | 32 | 128 | 4 | 128 | 4 | PASS | 1.4668 | 0.1441 | 10.179x | runs/s32.log |
| 64 | 1024 | 128 | 4 | 128 | 4 | PASS | 78.1360 | 9.8090 | 7.966x | runs/s1024.log |
| 32 | 100000 | 1024 | 16 | 1024 | 2 | NO VERDICT (script crashed: baseline CUDA OOM in accuracy trial 1) | - | - | - | runs/big.log |

Supplemental (not one of the goal shapes; run because the S=100000 row cannot
produce a verdict — closest feasible variant of that config):

| B | S | d_model | heads | ffn | layers | Verdict | Baseline med (ms) | Optimized med (ms) | Speedup | Log |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 2048 | 1024 | 16 | 1024 | 2 | PASS | 335.2492 | 70.6694 | 4.744x | runs/big_s2048.log |

## 2. What was implemented and why
The optimized class subclasses `BaselineTransformer` (identical parameters,
so the script's strict `copy_model_weights` works unchanged) and replaces
`forward` with:

1. **Fused QKV projection.** On first forward, the per-layer q/k/v projection
   weights and biases are concatenated once into a single `[3*d_model, d_model]`
   matrix, turning three small GEMMs per layer into one.
2. **`F.scaled_dot_product_attention(..., is_causal=True)`** instead of the
   baseline's materialized `[B, H, S, S]` score tensor + masked_fill + fp32
   softmax. In fp32 PyTorch selects the memory-efficient backend; this removes
   the huge intermediate (decisive for large B*S) and fuses scale/mask/softmax.
3. **Mask elision with a safe fallback.** The benchmark passes an all-True
   `valid_token_mask` (padding_ratio=0). The forward checks the mask once per
   distinct tensor (cached by `data_ptr`, so no per-call sync) and skips all
   masking work when it is all-True. A non-trivial mask falls back to the exact
   baseline path, so correctness does not depend on the benchmark's defaults.
4. **`torch.compile(mode="reduce-overhead", dynamic=False)`** over the fast
   path. Inductor fuses the LayerNorms, residual adds and GELU into few Triton
   kernels and wraps the whole forward in a CUDA graph, eliminating per-kernel
   launch overhead. The small shapes (d_model=128, S=128) are almost entirely
   launch-bound, which is where the 4-13x speedups come from.

Everything stays in fp32 (with the script-enabled TF32 matmuls), matching the
baseline's numerics; observed max_abs error was ~0.0009-0.0012 against the
0.002 budget across the shapes measured.

## 3. Tried and abandoned

* **Manual CUDA-graph capture over the eager fast path** (class
  `OptTransformer`, still in the module): worked and passed (3.446x on the
  main shape, optimized median 0.4276 ms), but the inductor-compiled variant
  was faster (0.3517 ms, 4.113x) because it also fuses the pointwise ops, so
  the compiled variant was shipped. The manual-graph class remains as a
  fallback selectable by omitting `OPT_COMPILE`.
* **fp16/bf16 internal compute** was considered and rejected without shipping:
  fp32+TF32 already consumes ~half the atol budget (max_abs ~0.0009-0.0013 vs
  0.002), so halving the mantissa for the ~2x GEMM headroom (relevant mainly to
  the d_model=1024 shapes) was judged too likely to FAIL the strict per-element
  check, and the time budget did not allow calibrating it as a separate
  candidate.
* **B=32, S=100000, d_model=1024, H=16, ffn=1024, L=2**: no script verdict is
  possible on this 48 GB GPU. The *baseline* inside the unmodified script OOMs
  during accuracy trial 1 -- first at its q-projection `.contiguous()`
  (12.21 GiB request on top of ~37 GiB already live), and its attention would
  need a [32,16,1e5,1e5] fp32 score tensor (~20 TB total; 40 GB even for a
  single batch-head slice) regardless. Log: `runs/big.log`. The optimized path itself is
  memory-efficient (no S x S materialization), but the script requires the
  baseline to run first, so this row is reported as NO VERDICT (script
  crashed), not as a pass or a speedup. A closest-feasible variant of the same
  config with S=2048 was run instead for supplemental evidence (see table).

## 4. Unverified items

* The S=100000 row: nothing was verified there; the script itself cannot run
  its baseline on this GPU. The optimized implementation was NOT separately
  demonstrated on that shape either (no standalone run was done).
* The non-trivial-padding-mask fallback path (mask not all-True) is code that
  simply calls the baseline implementation; the benchmark's padding_ratio=0
  default means no measured run exercised it.
* The manual CUDA-graph variant (`OptTransformer`) was only verified on the
  main shape, not across the sweep; the shipped/measured class everywhere in
  the table is `OptTransformerCompiled`.
