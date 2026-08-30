# transformer-forward-opt

The first workspace: **TikTok TechJam 2026, Track 3 -- Implement a GPU Kernel
for a Transformer Layer**. The submission is a transformer layer that runs
faster than the organizer's baseline while passing a zero-bad-element
correctness check. Everything about that task is in this directory: the
organizer's evaluator, the shapes it scores, the candidates, the phase prompts,
the docs, the machine, and the evidence.

Paths below are relative to this directory. Commands assume the repository's
`.venv` (`../../scripts/setup_env.sh`).

## Status

Campaign complete through two RLCR rounds on an RTX 6000 Ada (sm_89, driver
580.126.20, torch 2.8.0+cu128). Every number below is the organizer's script at
the official tolerance (`--atol 0.002 --rtol 0.02`), median latency, recorded
in `runs/benchmark.csv`; `runs/dispatch_table.json` routes each geometry to the
candidate that won it under the worst-case admission rule.

Dispatch validation, official grid (eager baseline denominator):

```
batch-1   12.3x   seq-1024  11.8x   batch-10000  3.1x   center     3.9x
seq-32     9.0x   batch-16   8.3x   narrow-32    6.7x   heads-1    4.7x
heads-2    4.4x   heads-16   3.5x   batch-128    2.3x   wide-1024  1.2x
```

Worst case 1.23x -- never slower than the baseline anywhere, and never behind
the strongest *numerically admissible* `torch.compile` configuration of the
baseline on any measured lane (the plain `max-autotune` baseline fails the
official tolerance itself: max_abs 0.0053 fp32, 0.0625 bf16). Shape #14
(S=100000), which the script's own reference cannot run on any hardware, is
served at 25.5 s with an off-script chunked comparison recording 0 bad
elements of 3.28e9 -- labeled off-script, never claimed as an official pass.
bf16/fp16 lanes ship the bit-exact graph-captured path (max_abs = 0).

The full narrative with every measurement's provenance: `docs/campaign-log.md`.

## Contents

| Path | Purpose |
|---|---|
| `verify.py` | The organizer's script, unmodified, with a candidate patched in. Its `main()`, its output, its exit code. `--record` appends its numbers to `runs/benchmark.csv` |
| `bench/official/` | The evaluator, vendored unmodified, md5 recorded, plus a note on its two contradictions |
| `bench/shapes/` | `smoke` (CPU correctness, every causal × padding combination), `dev` (nine cases around the grid's center), `official` (the 14 appendix shapes) |
| `kernels/` | Candidates: `passthrough` (control), `fused-safe`, `fused-sdpa`, `graph-safe`, `graph-sdpa`, `dispatch` (the shipping layer; `python kernels/dispatch.py calibrate` builds its table) |
| `prompts/` | The three phase prompts, built from `_shared.md` |
| `docs/benchmark-anatomy.md` | Eight things the evaluator's source says, two of which contradict the obvious guess |
| `docs/precision-budget.md` | What the tolerance actually buys, and where we refuse to spend it |
| `docs/reproduction.md` | Environment, the card, and how to re-run the workflow |
| `docs/runpod.md` | Renting the card: the network volume, the ten-minute D0 gate, the NCU permission probe, evidence sync |
| `docs/deliverables.md` | What the project must be able to show, and which artifact shows it |
| `docs/pitch.md` | The three-minute script and the Q&A preparation |
| `docs/submission-checklist.md` | The last-night list |
| `infra/runpod/` | Bootstrap for a RunPod network volume: tools, agent state and this checkout persist across pods |
| `scripts/` | `smoke.sh` (CPU gate), `run_ladder.sh` (the opponent), `demo.sh` (the judge-picks-a-shape moment) |
| `runs/` | The evidence: `benchmark.csv`, `solutions.jsonl`, `profile/`, `dispatch_table.json` |

## Run it

```bash
./scripts/smoke.sh                                   # correctness, CPU, seconds
python verify.py --list                              # candidates, shape sets, the script's md5
python verify.py fused-safe --shapes dev --record    # a sweep, the script's own numbers, recorded
./scripts/run_ladder.sh                              # eager, then torch.compile max-autotune as the baseline
python kernels/dispatch.py calibrate                 # the dispatch table from runs/benchmark.csv
./scripts/demo.sh center                             # the official script, official tolerance, exit 0 or 2
```

Flags after `--` go to the organizer's script verbatim; `--atol 0.002 --rtol
0.02` is the problem statement's rule and what this workspace targets,
`--compile-baseline --compile-mode max-autotune` turns its baseline into the
opponent worth quoting against.

## Running the agent

From this directory, in tmux, start a fresh Claude Code session and paste
`prompts/phase1.md`. Then phase 2, then phase 3, raising the target between
rounds. `docs/runpod.md` covers the machine; `../../docs/method.md` covers the
loop.

## Two caveats stated up front

**Appendix shape #14 breaks the evaluator's own baseline.** At
`S=100000, d=1024, B=32`, the eager reference would need a ~10 GB causal mask
and ~12.8 TB of score tensors -- it cannot run on any hardware, and in fp32 its
q/k/v alone overflow a 24 GB card. Memory-efficient causal attention is a
requirement on that shape, not an optimization, and its reference output has to
be recomputed in chunks. `docs/benchmark-anatomy.md` §9 has the arithmetic.

**The runs are not deterministic.** Search order, profiling noise, GPU
scheduling and model behaviour all vary. Re-running the prompts will not
reproduce the same kernels or the same path. What is documented here is the
workflow, not an outcome.
