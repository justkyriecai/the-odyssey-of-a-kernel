# transformer-forward-rtx6000ada-20260831

TikTok TechJam 2026, Track 3 -- **Implement a GPU Kernel for a Transformer
Layer**. Round 5, branched from `workspace/transformer-forward-opt` on
2026-08-31 onto the **same card** (NVIDIA RTX 6000 Ada Generation, AD102,
sm_89) with an empty ledger.

Carried over: `bench/`, `kernels/`, `verify.py`, `docs/`, `scripts/`, `infra/`
and the prompts. Not carried over: `runs/`, which starts empty on purpose --
a number is only a number if this workspace measured it.

Paths below are relative to this directory. Commands assume the repository's
`.venv` (`../../scripts/setup_env.sh`).

## Done at branch time

- [x] Evaluator md5 confirmed unchanged from the sibling
      (`21584e5923680ce0455554bd0b45bda2`). Same evaluator, same task.
- [x] `CLAUDE.md`: task, evaluator, card, machine and this round's target.
- [x] `prompts/_shared.md`: hardware section rewritten to name the **RTX 6000
      Ada** -- the card the sibling actually measured on. Its prompts said
      RTX 4090 while every recorded row said RTX 6000 Ada; both are AD102/sm_89,
      so no architectural claim moved, but SM count, VRAM, bandwidth and L2 did.
      Prompts rebuilt.

## Before the first number is quoted

- [ ] `./scripts/check_gpu.sh` on the box. It is the D0 gate, including the
      profiler permission probe -- run it on day zero, not the night you need
      a profile.
- [ ] `./scripts/smoke.sh` -- the `passthrough` control must read ~1.00x with
      zero error here. Until it does, nothing else measured here is trustworthy.
- [ ] Re-measure the eager baseline and the admissible-`torch.compile` opponent
      in this workspace, fresh process per case. The sibling's figures are the
      expectation, not this round's result.
- [ ] Re-measure every candidate the round intends to build on, then rank by
      what this ledger says.
- [ ] `docs/draft.md`: this round's directions, ranked by expected benefit
      against implementation risk, each split into subtasks.

## The sibling's result, as the prior to beat

Four RLCR rounds on this card produced a calibrated dispatch whose validation
over the official grid read, against the **eager** baseline denominator:

```
seq-1024    16.1x   batch-1    12.3x   batch-4    10.9x   seq-32     9.0x
batch-16     8.3x   narrow-32   6.7x   heads-1     4.7x   heads-2    4.4x
batch-10000  3.9x   center      3.9x   heads-16    3.5x   batch-128  2.3x
wide-1024    1.2x
```

Worst case 1.23x, never slower than the baseline anywhere, and never behind the
strongest *numerically admissible* `torch.compile` configuration on any
measured lane. Shape #14 (`S=100000`), which the evaluator's own reference
cannot run on any hardware, was served off-script and never claimed as an
official pass.

That is the bar this round starts from -- and it is a bar recorded in another
directory. Reproduce it here before improving on it.

## Contents

| Path | Purpose |
|---|---|
| `verify.py` | The organizer's script, unmodified, with a candidate patched in. Its `main()`, its output, its exit code. `--record` appends its numbers to `runs/benchmark.csv` |
| `bench/official/` | The evaluator, vendored unmodified, md5 recorded, plus a note on its two contradictions |
| `bench/shapes/` | `smoke` (CPU correctness, every causal x padding combination), `dev` (nine cases around the grid's center), `official` (the 14 appendix shapes) |
| `kernels/` | Candidates: `passthrough` (control), `fused-*`, `graph-*`, `compiled-*`, `flash-*`, `dispatch` (the shipping layer) |
| `prompts/` | The three phase prompts, built from `_shared.md` |
| `docs/benchmark-anatomy.md` | Eight things the evaluator's source says, two of which contradict the obvious guess |
| `docs/precision-budget.md` | What the tolerance actually buys, and where we refuse to spend it |
| `docs/reproduction.md` | Environment, the card, and how to re-run the workflow |
| `docs/runpod.md` | Renting the card: the network volume, the ten-minute D0 gate, the NCU permission probe |
| `infra/runpod/` | Bootstrap for a RunPod network volume |
| `scripts/` | `smoke.sh` (CPU gate), `run_ladder.sh` (the opponent), `demo.sh` (the judge-picks-a-shape moment) |
| `runs/` | This round's evidence. Empty until measured here |
