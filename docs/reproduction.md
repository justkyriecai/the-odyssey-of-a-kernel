# Reproduction

Two levels: run the harness against a candidate on any machine, or re-run the
whole agent workflow from a fresh workspace.

## Environment

Python 3.10+ and a PyTorch build matching your CUDA runtime.

```bash
git clone https://github.com/justkyriecai/the-odyssey-of-a-kernel.git
cd the-odyssey-of-a-kernel

# CPU-only (correctness work, laptops, CI)
./scripts/setup_env.sh

# With a specific CUDA wheel index
TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/setup_env.sh
```

Renting the card on RunPod — pod spec, the first-ten-minutes gate, the NCU
permission question, and how evidence leaves the pod — is covered in
`docs/runpod.md`.

Then confirm the machine is actually ready:

```bash
./scripts/check_gpu.sh
```

That script is a D0 gate, not a formality. It checks the driver, the torch CUDA
build, **and whether Nsight Compute is permitted to profile** -- which on many
rented boxes it is not, because `NVreg_RestrictProfilingToAdminUsers` defaults
to restricting it. Discovering that on day two costs a day.

## The card

**RTX 4090 for development, rented by the hour. A second card late, for the
cross-hardware check.**

The problem statement removes hardware from the competition: *"Optimize & test
your codes on your own machine. Different methods may be used to optimize the
codes depending on the machine (GPU cards) you use."* There is no shared
hardware baseline, so a bigger card buys no points -- and buys a *worse* project.

At the default shape, ~40 GFLOP of matmul over 1024 tokens is roughly 40µs of
arithmetic on an H100 against a launch overhead an order of magnitude larger.
The roofline degenerates: every point sits in the bottom-left corner, the answer
collapses to "use CUDA Graphs", and there is nothing else to find. On a 4090 the
same work is a few hundred microseconds -- the same order as the overhead. The
smaller card makes the problem *more* interesting, not less.

Approximate on-demand rates as of late August 2026 -- verify before booking:
RTX 4090 around $0.28-0.50/hr, A100 80GB around $1/hr, H100 from roughly
$1/hr spot. Forty hours of 4090 development plus a few hours of H100 validation
is tens of dollars, which is why none of this needs a purchase.

One caveat from the real grid: appendix shape #14 (`S=100000, d=1024, B=32`)
does not fit a 24 GB card at fp32 -- q/k/v alone are ~38 GB -- so the 4090
covers 13 of the 14 shapes and the stress shape belongs in the 80 GB-class
validation hours, not the development loop. See `docs/benchmark-anatomy.md` §9.

If you use a different card, **rewrite the hardware section of the phase
prompts** before running them. Prompts written for one architecture send an
agent chasing features the hardware does not have, and that is the single most
expensive porting mistake available here.

## Reduce timing variance

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=<min>,<max>
sudo nvidia-smi --lock-memory-clocks=<min>,<max>
```

The official script already alternates measurement order between rounds and
reports medians, so it is robust to drift. Locking clocks makes p90 mean
something.

## Run it

```bash
# What is here
python -m odyssey list

# Correctness only, CPU, seconds
./scripts/smoke.sh

# The opponent, before writing anything: L0 eager / L1 compile / L2 max-autotune
./scripts/run_ladder.sh

# Sweep, recalibrate dispatch, redraw the roofline, then the official gate
./scripts/run_sweep.sh
```

Every command that measures writes to `runs/benchmark.csv` and appends a node to
`runs/solutions.jsonl`. Pass `--no-record` for throwaway runs.

## The gate

The harness is the inner loop. The gate is the organizer's own script, run
unmodified, with the candidate patched into `UserOptimizedTransformer` at
runtime:

```bash
python -m odyssey official dispatch --shapes official --case center -- --atol 0.001 --rtol 0.01
```

This prints exactly what the organizers would see and returns their exit code:
0 for pass, 2 for an accuracy failure. Nothing gets reported that has not
cleared this.

## The workload grid

`bench/shapes/official.json` holds the 14 evaluation shapes from Appendix 3.7
of the problem statement -- a one-factor sweep around `B=64, S=128, d=128,
H=4, ffn=128, L=4`, every shape causal, plus the `S=100000` stress case. The
table does not specify dtype, padding or input scale; those default to the
script's own defaults and the organizers can move them with flags.

`bench/shapes/dev.json` is the iteration set, aligned with the grid's center
point, and `bench/shapes/smoke.json` is a CPU-only correctness gate covering
every combination of causal and padding.

## Re-run the agent workflow

1. Set up this repository and confirm the GPU.
2. Install the agent tooling:

```bash
# humanize: the plan/execute/verify harness, as a Claude Code plugin
/plugin marketplace add PolyArch/humanize
/plugin install humanize@PolyArch

# two skills
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone https://github.com/mit-han-lab/KernelWiki.git
git clone https://github.com/mit-han-lab/ncu-report-skill.git
```

3. Create a separate implementation workspace. Do not run the search in this
   repository -- the search history and the release tree should not share a
   directory.

```bash
mkdir -p workspaces
git clone . workspaces/round-1
cd workspaces/round-1
export TECHJAM_BENCHMARK="$OLDPWD/bench/official/torch_transformer_benchmark.py"
```

4. Paste `prompts/transformer-layer/phase1.md` into a fresh agent session and
   work the loop. Then phase 2, then phase 3, raising the target between rounds.

Runs are not deterministic. Search order, profiling noise, GPU scheduling and
model behaviour all vary, so a re-run will not reproduce the same kernels or the
same path. What the prompts document is the workflow, not an outcome.

## Reproducing the evidence artifacts

```bash
python -m odyssey peak --dtype bfloat16       # measured roofline ceilings
python -m odyssey roofline                    # migration plot -> docs/assets/
python -m odyssey ablation --scaffold         # three arms under runs/ablation/
python -m odyssey ablation --plot             # once all three have run
python -m odyssey dag --out runs/search.dot   # dot -Tsvg runs/search.dot
```

Use `odyssey peak` before showing any roofline. A ceiling drawn from a
spec-sheet number is a ceiling drawn from marketing; `DEVICE_PEAKS` entries are
all marked `verified=False` for that reason.
