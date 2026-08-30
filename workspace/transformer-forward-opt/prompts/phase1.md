# Transformer Layer -- Phase 1: Get It Right

Produce a correct replacement for `UserOptimizedTransformer` that passes every
shape in the development set, including the padded and causal ones. Speed is
secondary this phase. A clean, correct design you can profile beats a fast one
you cannot trust.

<!-- BEGIN shared -->

## Kernel Information

- Task: TikTok TechJam 2026 Track 3 -- Implement a GPU Kernel for a Transformer Layer.
- Evaluator: `bench/official/torch_transformer_benchmark.py` (organizer-provided,
  unmodified, `md5 21584e5923680ce0455554bd0b45bda2`).
- What you replace: `UserOptimizedTransformer.forward`. The whole model is in
  scope -- fusion across layers, CUDA Graphs, custom kernels.
- Signature, fixed: `forward(x, valid_token_mask=None) -> [batch, seq_len, d_model]`.
- Weights: copied from the baseline with `load_state_dict(strict=True)`. Keep the
  parameter names. Derive fused weights lazily on the first forward; register any
  extra buffers with `persistent=False`.

### Variable axes

`batch_size`, `seq_len`, `d_model`, `num_heads`, `ffn_dim`, `num_layers`,
`causal`, `dtype` (`float32` / `float16` / `bfloat16`), `padding_ratio`.

The evaluation grid (`bench/shapes/official.json`, from Appendix 3.7) is a
one-factor sweep around `B=64, S=128, d_model=128, heads=4, ffn=128, layers=4`:
batch to 10000, width 32-1024 with `ffn_dim = d_model` (1:1, not 4:1), heads
1-16 so `head_dim` spans 8 to 256, sequence 32-1024, plus a `S=100000` stress
shape the eager baseline cannot even run (see `docs/benchmark-anatomy.md` §9).
**Every shape in the grid is causal.** The center point is ~0.4M parameters --
the shape profile of a ranking or retrieval model, and even more launch-bound
than the script's own defaults. dtype, padding and input scale are not grid
columns; they default to the script's defaults and can be moved by flags.

### The reference computation, exactly

Pre-LN blocks, `L` of them, then a final LayerNorm:

1. `h = norm1(x)`; `q, k, v = q_proj(h), k_proj(h), v_proj(h)` -- all
   `bias=True`, all `[d_model, d_model]`.
2. `scores = q @ k^T * head_dim**-0.5`.
3. If `causal`, mask the strict upper triangle to `-inf`.
4. If `valid_token_mask` is given, mask invalid **key** positions to `-inf`.
5. `probs = softmax(scores.float(), dim=-1).to(x.dtype)` -- **softmax in fp32**,
   then back to the working dtype.
6. `context = probs @ v`, merge heads, `out_proj`.
7. If masking, zero the attention output at invalid token positions.
8. `x = x + attention_out`.
9. `x = x + ffn_out(gelu(ffn_in(norm2(x)), approximate="none"))` -- the exact
   erf GELU, not the tanh approximation.
10. If masking, zero the block output at invalid token positions.
11. After the last block: `final_norm`, then zero again if masking.

Steps 5, 7, 9, 10 and the second zeroing in 11 are where reimplementations go
wrong. Steps 7, 10 and 11 are invisible at `--padding-ratio 0`.

## Official Acceptance

Every output element must satisfy:

```text
abs(user - ref) <= atol   OR   abs(user - ref) <= rtol * abs(ref)
```

An OR, not a sum -- the effective tolerance is `max(atol, rtol*|ref|)`, tighter
than `torch.isclose`. **Zero bad elements**, and any non-finite value fails.

The problem statement sets the numbers: *"relative error < 0.02, abs error <
0.002"*. That is what the argparse defaults already are; the docstring's
`0.001 / 0.01` is a leftover from an earlier revision. Target `0.002 / 0.02`.

Run the gate through the organizer's own `main()`:

```bash
python verify.py <candidate> --shapes dev --case center -- --atol 0.002 --rtol 0.02
```

## Development shapes

Use these nine before running the full grid. Cheap enough to run after every
change; broad enough that a win here is unlikely to be a fluke.

All cases are causal (as is every official shape) and anchored on the grid's
center point:

| Case | Deviation from center | Why it is in the set |
|---|---|---|
| `center` | -- | The grid's center. 8192 tokens, ~0.4M params. Launch-bound. |
| `center-bf16` | bf16 | Does the tolerance budget survive reduced precision. |
| `center-fp16` | fp16 | Wider mantissa than bf16, narrower range. |
| `center-padded` | padding 0.4 | Three output-masking sites become live. |
| `batch-1` | B=1 | The latency floor; launch overhead is the wall clock. |
| `batch-128` | B=128 | The large-batch end that still iterates quickly. |
| `wide-1024` | d=1024, ffn=1024 | head_dim 256, the flash upper limit. |
| `heads-16` | H=16 | head_dim 8, the fused-backend lower edge. |
| `seq-1024` | S=1024 | O(S^2) attention; nearest cheap proxy for stress-100k. |

`batch-10000` and `stress-100k` live only in the official set -- the first is
too slow to iterate on, the second needs an 80 GB card and a chunked reference.

## Workflow Requirements

- Record every measurement in `runs/benchmark.csv`. `python verify.py ...
  --record` appends the script's own numbers; anything measured another way is
  recorded by hand in the same columns, never summarized from memory.
- Record every candidate in `runs/solutions.jsonl` with a parent link, forming
  a DAG. One JSON object per line: `node_id`, `parent`, `candidate`, `case`,
  `status` (`pass` / `fail` / `skipped` / `error`), `speedup`,
  `max_abs_error`, `decision` (`keep` / `reject` / `park`), `evidence` (a
  path, a commit, a run), `notes`, `timestamp`. Rejected branches included --
  especially rejected branches.
- Keep an NCU report per major direction under `runs/profile/<direction>/`,
  with a text export next to the binary.
- At most five iterations per direction. Then record the evidence and move on.
- Use `KernelWiki` for kernel research and `ncu-report-skill` for reading
  profiles.
- Do not modify the official script. Do not tune against a single shape and
  report it as a general result. Do not report a best-of run; report the median,
  with p90 next to it. Do not redefine the target and do not substitute an
  easier baseline.

## Hardware

**Development card: NVIDIA GeForce RTX 4090 (AD102, sm_89, 128 SMs, 24 GB
GDDR6X, ~1008 GB/s).** Rented by the hour; a second card is used late for a
cross-hardware check.

Available and worth trying:

- 4th-generation tensor cores: fp16/bf16 with fp32 accumulate, and TF32 for
  fp32 matmuls (`allow_tf32` is on by default in the evaluator, so the baseline
  already gets it -- it is not free headroom).
- `cp.async` for global-to-shared copies.
- CUDA Graphs. At ~90 launches per forward for 1024 tokens, this is the first
  thing to measure, not the last.
- Triton, `torch.compile` / Inductor, custom CUDA extensions.
- Large L2 (72 MB on AD102): at these shapes the whole working set may be
  resident, which changes what "memory-bound" means here.

**Not available on this card.** Do not spend iterations on:

- TMA and the tensor memory accelerator (Hopper and later).
- Thread block clusters and distributed shared memory (Hopper and later).
- `tcgen05` and TMEM (Blackwell).
- Hopper-style hardware warp specialization (`setmaxnreg`).

FP8 exists on Ada but is **out of scope by decision, not by capability**: E4M3
carries 3 mantissa bits, roughly 6% relative error, and the rule tolerates zero
bad elements. One tail outlier fails the whole trial.

> **Replace this entire section when the card changes.** Prompts written for one
> architecture send an agent chasing features that do not exist on another. This
> is the first thing to adapt, and the most expensive to get wrong.

<!-- END shared -->

## Phase 1 Goal

Research first, then implement once.

**Research.** Read the official script end to end before writing anything --
particularly `compare_outputs`, `generate_random_case` and `benchmark_models`.
Understand what is measured, what is masked, and where the tolerance actually
binds. Then read what is already known about fusing this shape of model:
`scaled_dot_product_attention` backends and their accumulation behaviour, fused
QKV projections, LayerNorm/residual fusion, CUDA Graphs on small-batch inference.

**Establish the opponent before you build.** Run the ladder first:

```bash
./scripts/run_ladder.sh
```

It runs the official script three ways: candidates against its eager baseline
(TF32 already on), candidates against `--compile-baseline --compile-mode
max-autotune`, and `passthrough` against that compiled baseline so you know
what compile alone buys. `torch.compile` at `max-autotune` is one flag away in
the official script; the margin over it is the number anyone in the room will
ask about. Do not write a kernel before you know it.

**Implement.** One correct candidate. Priorities in order:

1. Correct under every combination of `causal` and `padding_ratio > 0`. The
   three output-zeroing sites are invisible at `padding_ratio 0` and fail loudly
   the moment padding is on.
2. Correct at every dtype. The baseline computes softmax in fp32 and casts back;
   a fused attention path that does not may drift out of tolerance in bf16.
   Measure it -- do not assume either way.
3. Only then, the free structural wins: one QKV GEMM instead of three, no
   per-head `.contiguous()` materializations, the mask built once per forward
   instead of once per layer.

Nothing in this phase should trade accuracy for speed. If a candidate needs that
trade, it belongs in a separate candidate whose name says so, measured against
the one that does not.

**Verify.** Every time:

```bash
./scripts/smoke.sh <candidate>                # every causal x padding combination, CPU, seconds
python verify.py <candidate> --shapes dev --record -- --atol 0.002 --rtol 0.02
```

## Draft First

Write the implementation plan to `docs/draft.md` before implementing. It must
contain: what the reference computation actually is (op by op, including the
masking sites), which structural inefficiencies you found in the baseline and
what each is worth, the candidate directions ranked by expected benefit against
risk, and how you will tell a real improvement from timing noise.

Then run `/humanize:gen-plan` on that draft.
