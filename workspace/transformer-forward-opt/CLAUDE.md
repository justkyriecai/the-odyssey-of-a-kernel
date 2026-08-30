# Agent Instructions: transformer-forward-opt

This workspace is TikTok TechJam 2026 Track 3 (GPU kernel for a transformer
layer). The repository-level `CLAUDE.md` applies; these are the task's hard
facts. Read `README.md` and `docs/benchmark-anatomy.md` before changing
anything that measures.

## Hard rules

- **Never modify `bench/official/torch_transformer_benchmark.py`.** It is the
  organizer's evaluator, vendored so that every number is traceable to a known
  file. Its md5 (`21584e5923680ce0455554bd0b45bda2`) is recorded in
  `bench/official/README.md` and printed by `python verify.py --list`.
- **Never reimplement the correctness rule or the timing protocol.** Run the
  script through `python verify.py <candidate> ...`, which patches
  `UserOptimizedTransformer` at runtime and calls the script's own `main()`.
  If a number anywhere disagrees with the script's own output, the script is
  right.
- **Target `--atol 0.001 --rtol 0.01`**, the stricter of the two readings in the
  script (its argparse defaults are `0.002 / 0.02`; `verify.py` passes the
  script's defaults unless you say otherwise, so say otherwise). Zero bad
  elements; non-finite values fail outright.
- Work here. The search, its dead ends and its evidence all belong in this
  directory; `runs/` is committed.

## Adding a candidate

Candidates live in `kernels/`; each module exposes `CANDIDATES`, mapping a name
to `(description, factory)`, where the factory takes the loaded script module
and returns the class to substitute. The contract is the script's, not ours:

1. `forward(x, valid_token_mask=None) -> [batch, seq_len, d_model]`.
2. Parameter names compatible with `BaselineTransformer` -- `copy_model_weights`
   loads the baseline state dict with `strict=True`. Subclass it and override
   `forward`; derive fused weights lazily on the first call; register extra
   buffers with `persistent=False`.
3. Candidates that need CUDA fall back to the wrapped path elsewhere and say so
   in their description; a CPU number for `graph-*` is not a graph number.
4. Run `./scripts/smoke.sh` before anything else. It covers every combination
   of causal and padding on CPU in seconds, and that cross is where
   reimplemented attention breaks -- invisibly at `--padding-ratio 0`.

**A speed/accuracy trade is a separate named candidate, never a flag on an
existing one.** `fused-safe` and `fused-sdpa` differ only in whether softmax
happens in fp32; they are two candidates so that calibration can measure both
on the actual hardware and admit whichever is correct *and* faster.

## The recording discipline

`python verify.py ... --record` appends one row per case to `runs/benchmark.csv`
with the script's own numbers, the script's md5, the git sha and the exact
flags. Every candidate appends a node to `runs/solutions.jsonl` with a parent
link (schema in `prompts/_shared.md`). Record rejected branches with
`decision: reject` and the evidence that killed them.

Cap each optimization direction at five iterations. If it cannot be implemented
cleanly, fails correctness, or shows no credible path after five, write down
what you learned and take the next-ranked direction.

## Reporting numbers

- Median, with p90 next to it. Never a best-of run.
- Always name the baseline. "2.7x" without "than what" is not a result. The
  number to quote is the one measured with `--compile-baseline --compile-mode
  max-autotune`.
- Run the full shape set, not the one case that looks good. Tuning on `center`
  until everything else regresses is the standard way to lose this.
- A roofline uses ceilings **measured** on the card (a large GEMM, a large
  copy), never spec-sheet numbers.

## Prompts

After editing `prompts/_shared.md`, run `python ../../scripts/build_prompts.py`
-- the phase prompts each carry an expanded copy so they stay self-contained
when pasted into a session.
