# transformer-forward-rtx6000ada-20260831

Round 5 of the task carried in `workspace/transformer-forward-opt`, on the same
card, with an empty ledger. The task, the evaluator, the shape sets,
`verify.py` and the candidates were carried over on 2026-08-31; the
measurements were not.

- Task: TikTok TechJam 2026 Track 3 -- Implement a GPU Kernel for a Transformer
  Layer.
- Evaluator: `bench/official/torch_transformer_benchmark.py`, provided by the
  organizer, unmodified, `md5 21584e5923680ce0455554bd0b45bda2` -- re-checked
  against the sibling on 2026-08-31 and unchanged. Same evaluator, same task.
- What the candidate replaces: `UserOptimizedTransformer.forward`, signature
  fixed at `forward(x, valid_token_mask=None) -> [batch, seq_len, d_model]`.
  The whole model is in scope -- fusion across layers, CUDA Graphs, custom
  kernels.
- Hardware: **NVIDIA RTX 6000 Ada Generation** (AD102, sm_89, 142 SMs, 48 GB),
  rented by the hour. The same card the sibling measured on, which is why its
  numbers are a usable prior here rather than merely a hypothesis.
- Target: beat the fastest **numerically admissible** `torch.compile`
  configuration of the baseline on every measured lane, and never be slower
  than the eager baseline anywhere on the official grid. Set by the user on
  2026-08-31, same framing as the sibling. It is a framing, not a fixed
  multiple: the number falls out of the card, and the worst lane is the one
  that counts.
- Tolerance: `--atol 0.002 --rtol 0.02`, the problem statement's own numbers and
  the script's argparse defaults. The docstring's `0.001 / 0.01` is a leftover
  from an earlier revision and is not the rule. Zero bad elements; any
  non-finite value fails outright.

The repository rules in the root `CLAUDE.md` apply here unchanged. In
particular: the evaluator is never edited, its correctness rule and its timing
protocol are never re-implemented, and every measurement lands in
`runs/benchmark.csv`.

## Same card, so read the carry-over correctly

The usual branch warning is that a candidate tuned on another card is only a
hypothesis here. That is **not** the situation. The card did not change, so the
sibling's tile sizes, launch bounds and occupancy assumptions should still
hold, and its `runs/benchmark.csv` is a legitimate prior for what to expect.

What did change is the ledger. `runs/` starts empty, and a number is only a
number if this workspace measured it. So:

- Re-establish the floor before anything else: `./scripts/smoke.sh` must read
  ~1.00x with zero error on the `passthrough` control here.
- Re-measure the eager baseline and the admissible-compile opponent on this
  box, in this workspace, before ranking anything. Fresh process per case.
- Quote the sibling's numbers as *expectation*, never as this round's result.
  If a re-measurement disagrees with the sibling, the new measurement wins and
  the disagreement is worth a line in `docs/campaign-log.md`.

One correction inherited from the sibling: its `prompts/_shared.md` and
`docs/reproduction.md` named an RTX 4090 as the development card, while every
row in its `runs/benchmark.csv` records `NVIDIA RTX 6000 Ada Generation`. The
hardware section here has been rewritten to name the card that was actually
measured. Both are AD102/sm_89, so no architectural claim changed -- but the
SM count, VRAM, bandwidth and L2 figures did.

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
existing one**, so calibration can measure both and admit whichever is correct
*and* faster.

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
- Always name the baseline. "2.7x" without "than what" is not a result.
- Run the full shape set, not the one case that looks good.
- A roofline uses ceilings **measured** on the card, never spec-sheet numbers.

## Prompts

After editing `prompts/_shared.md`, run `../../.venv/bin/python
../../scripts/build_prompts.py` -- the phase prompts each carry an expanded copy
so they stay self-contained when pasted into a session.
