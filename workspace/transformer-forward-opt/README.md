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

Framework in place; the optimization campaign has not been run on a GPU.
Nothing here reports a GPU speedup yet. What does work, end to end, on CPU
today (`./scripts/smoke.sh`; timings move between runs, the error columns do
not):

```
candidate    case           verdict  speedup    max_abs
fused-safe   plain          PASS      1.096x          0
fused-safe   bf16           PASS      1.411x          0
fused-sdpa   plain          PASS      1.451x    7.2e-07
fused-sdpa   bf16           FAIL          --    0.03125   exit 2
dispatch     bf16           PASS      0.991x          0   no table yet: baseline path
```

That `FAIL` is the workflow working. `scaled_dot_product_attention` does not
reproduce the baseline's fp32 softmax on every backend; in bf16 the drift
accumulates across the residual layers and clears `atol`. Calibration will see
it, decline to admit `fused-sdpa` for bf16 geometries, and route those shapes to
the variant whose error is zero by construction. Two candidates, one
measurement, no guessing.

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
