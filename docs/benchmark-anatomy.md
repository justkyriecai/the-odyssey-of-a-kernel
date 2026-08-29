# Reading the Evaluator

Everything below comes from `bench/official/torch_transformer_benchmark.py` and
is checkable line by line. It is written down because several of these facts
change what is worth working on, and two of them contradict the obvious guess.

## 1. The correctness rule is an OR, and it tolerates nothing

```python
abs_ok = abs_error <= atol
rel_ok = abs_error <= rtol * ref.abs()
passed_mask = finite_mask & (abs_ok | rel_ok)
...
passed = failed_elements == 0
```

The effective tolerance is `max(atol, rtol*|ref|)` -- *tighter* than
`torch.isclose`, which allows `atol + rtol*|ref|`. The script says so in a
comment and declines to use `isclose` deliberately.

Two consequences worth internalizing:

- **Zero bad elements.** Not a ratio, not a percentile. One element over
  tolerance fails the trial. For contrast, the MoE task in the MLSys 2026
  FlashInfer contest ran with `--required-matched-ratio 0.9` -- a tenth of the
  elements were allowed to miss. Here, none are.
- **Non-finite values fail outright.** A single NaN from an all-masked softmax
  row costs the entire run.

This is the fact that rules out FP8. E4M3 has three mantissa bits, roughly 6%
relative error. Under a zero-bad-element rule, one tail outlier is total failure.

## 2. The script disagrees with itself about tolerance

The module docstring says `atol=0.001, rtol=0.01`. The `argparse` defaults say
`atol=0.002, rtol=0.02`. The problem statement agrees with `argparse`; the
docstring looks like a leftover from an earlier revision.

We do not get to know which the organizers will run. Everything here targets the
stricter pair, because a candidate that clears `0.001 / 0.01` clears both, and
the cost of the extra margin is small next to the cost of being wrong about it.

## 3. The baseline is stronger than "naive fp32"

`--allow-tf32` defaults to `True` and `--matmul-precision` defaults to `"high"`,
so every fp32 matmul in the *baseline* already runs on TF32 tensor cores. The
usual first win -- "we used tensor cores" -- has already been taken.

The baseline can be made stronger still without touching a line of it:
`--compile-baseline --compile-mode max-autotune` are built-in flags. That is the
real opponent, and it is one flag away for anyone in the room who asks. Measure
it before writing a kernel; see `odyssey ladder`.

## 4. dtype is a test condition, not a lever

```python
baseline  = baseline.to(device=device, dtype=dtype).eval()
optimized = optimized.to(device=device, dtype=dtype).eval()
```

Both models are cast to the same dtype. You cannot win by declaring the model
bf16 -- the baseline is bf16 too. What is available is reducing precision
*inside* your own forward and converting back, which the tolerance budget must
then pay for. See `docs/precision-budget.md`.

## 5. The architecture, precisely

Pre-LN blocks, `num_layers` of them, then a final LayerNorm.

- `q_proj`, `k_proj`, `v_proj`, `out_proj`: all `[d_model, d_model]`, all
  `bias=True`.
- `ffn_in`, `ffn_out`: `nn.Linear` with default bias, i.e. bias everywhere.
- GELU is `approximate="none"` -- the exact erf form. The tanh approximation
  differs by about 1e-3 relative, which is inside `rtol` for most elements and
  outside `atol` for small ones. Under a zero-bad-element rule, do not.
- Softmax is computed in **fp32** and cast back:
  `torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)`. A fused attention
  path that keeps softmax in bf16 is not doing the same arithmetic. Whether that
  matters is measurable, and it does not always go the same way -- see the note
  in `kernels/v1_fused_attention.py`.
- Weights are deep-copied from the baseline with `load_state_dict(strict=True)`,
  so both models compute with identical parameters. Every difference in the
  output is arithmetic, never initialization.

## 6. The timing protocol is already careful

`torch.cuda.Event` on the current stream, 20 warmup iterations, 3 rounds of 100
repeats, and the measurement order alternates between rounds so clock and
thermal drift is shared rather than handed to whoever ran first. Speedup is the
ratio of **medians**; mean, p90 and min are printed alongside.

Two details that matter when comparing runs: the timed input is generated with
`seed + 100000`, a *different* input from any of the accuracy trials
(`seed + trial`); and if accuracy fails, the benchmark is skipped entirely and
the script exits 2 unless `--benchmark-on-failure` is passed.

There is no headroom here to exploit and no reason to build a second timer. Use
this one.

## 7. Two structural inefficiencies, sitting in the baseline source

**Four whole-tensor materializations per layer.** `_split_heads` is
`.view().transpose().contiguous()`, applied separately to q, k and v -- three
full-tensor copies. The attention output is then transposed and made contiguous
again. Four round trips through memory per layer, `num_layers` times, for a
layout change.

**The default shape is small.** `B=8, S=128` is 1024 tokens; `d_model=512`,
`ffn_dim=2048`, `num_layers=6` is about 18.9M parameters and roughly 40 GFLOP of
matmul per forward. On an RTX 4090 that is a few hundred microseconds of
arithmetic; on an H100, tens. The forward issues on the order of a hundred
kernel launches -- count it with a profile rather than trusting that estimate --
and at this size launch overhead and memory round trips plausibly dominate the
arithmetic entirely.

Which means the interesting hypothesis is not *a faster matmul*. It is *fewer
kernels*: CUDA Graph capture, plus deleting the four materializations. That is
a hypothesis, not a conclusion -- `odyssey bench` and an NCU report settle it.

## 8. The reef: padding silently breaks a naive SDPA swap

With `--padding-ratio > 0`, `generate_random_case` builds a `valid_token_mask`
and the model does four separate things with it:

1. Zeroes the *input* at invalid positions.
2. Masks invalid **key** positions to `-inf` inside attention.
3. Zeroes the attention output at invalid positions, after `out_proj`.
4. Zeroes the block output at invalid positions -- and again after `final_norm`.

Steps 3 and 4 are pure output masking and have no analogue in
`scaled_dot_product_attention`. Replacing the attention body without
reproducing them passes at `--padding-ratio 0.0`, which is the default, and
fails the moment anyone turns padding on.

**No row can be fully masked**, so softmax cannot produce NaN. The argument:
`min_valid = max(1, round(S * (1 - padding_ratio)))` and lengths are drawn from
`[min_valid, S]`, so every sequence has length at least 1 and key position 0 is
always valid; the causal triangle always keeps the diagonal. Both halves of that
are load-bearing. This is a reading of the source, and the smoke set exercises
every combination of causal and padding precisely so it is checked rather than
believed -- `bench/shapes/smoke.json`, cases `padded` and `causal-padded`.

## What this implies for where the time goes

| Finding | What it rules out | What it argues for |
|---|---|---|
| OR rule, zero bad elements | FP8, tanh GELU, aggressive approximation | An explicit precision budget, per shape |
| TF32 already on in the baseline | "we used tensor cores" as the story | Beating `torch.compile max-autotune`, not eager |
| dtype is a test condition | Choosing a cheaper dtype | Reduced precision inside the forward, paid for from the budget |
| 1024 tokens, ~100 launches | Starting with matmul tuning | CUDA Graphs and materialization removal, measured first |
| Output masking in three places | A drop-in SDPA swap | Running the padded cases on every candidate |
| Timing already alternates and takes medians | Building a second timer | Reporting p90 next to the median |
