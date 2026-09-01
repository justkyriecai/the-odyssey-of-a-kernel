# Reproduction

Two levels: run a candidate through the organizer's script on any machine, or
re-run the whole agent workflow in this workspace.

## Environment

From the repository root, two levels up:

```bash
git clone --recurse-submodules https://github.com/justkyriecai/the-odyssey-of-a-kernel.git
cd the-odyssey-of-a-kernel

# Inherits the interpreter's torch when it already has one (a CUDA image), and
# downloads one when it does not (a laptop, CI).
./scripts/setup_env.sh

# A runtime the image does not carry, and a venv off the checkout's filesystem
SYSTEM_TORCH=0 TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./scripts/setup_env.sh
VENV_DIR=/opt/odyssey-venv ./scripts/setup_env.sh

# The agent workflow: humanize plugin, KernelWiki, ncu-report-skill
./scripts/setup_agent.sh
```

Renting the card on RunPod -- the network volume, the first-ten-minutes gate,
the NCU permission question, and how evidence leaves the pod -- is covered in
`docs/runpod.md`.

Then confirm the machine is actually ready:

```bash
./scripts/check_gpu.sh transformer-forward-opt
```

That script is a D0 gate, not a formality. It checks the driver, the torch CUDA
build, **which profilers the box permits** and ends by running this workspace's
CPU smoke test. Profiling is two answers, not one: Nsight Compute needs the
hardware performance counters, which many rented boxes withhold because
`NVreg_RestrictProfilingToAdminUsers` defaults to restricting them, while
Nsight Systems traces through CUPTI and usually still works. The gate reports
both and only fails when neither is available. Discovering a denied profiler on
day two costs a day; `docs/runpod.md` has what to do about it.

## The card

**RTX 6000 Ada (48 GB) for development, rented by the hour** -- the card
every row of `runs/benchmark.csv` records.

The problem statement removes hardware from the competition: *"Optimize & test
your codes on your own machine. Different methods may be used to optimize the
codes depending on the machine (GPU cards) you use."* There is no shared
hardware baseline, so a bigger card buys no points -- and buys a *worse*
project.

At the script's default shape, ~40 GFLOP of matmul over 1024 tokens is roughly
40µs of arithmetic on an H100 against a launch overhead an order of magnitude
larger. The roofline degenerates: every point sits in the bottom-left corner,
the answer collapses to "use CUDA Graphs", and there is nothing else to find.
On an AD102-class workstation card the same work is a few hundred
microseconds -- the same order as the overhead -- so every regime of the grid
stays interesting. Rented on-demand by the hour, the whole campaign cost tens
of dollars.

One caveat from the real grid: appendix shape #14 (`S=100000, d=1024, B=32`)
never fits a 24 GB card at fp32 -- q/k/v alone are ~38 GB -- and its eager
reference cannot run anywhere at any size. On this card's 48 GB the
batch-sliced flash kernel serves it (the recorded run peaked at 36.8 GiB),
with the chunked reference comparison running on host memory. See
`docs/benchmark-anatomy.md` §9.

If you use a different card, **rewrite the hardware section of
`prompts/_shared.md`** and rebuild the prompts before running them. Prompts
written for one architecture send an agent chasing features the hardware does
not have, and that is the single most expensive porting mistake available here.

## Reduce timing variance

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=<min>,<max>
sudo nvidia-smi --lock-memory-clocks=<min>,<max>
```

The official script already alternates measurement order between rounds and
reports medians, so it is robust to drift. Locking clocks makes p90 mean
something. A rented pod usually cannot lock clocks; see `docs/runpod.md`.

## Run it

From this directory:

```bash
python verify.py --list                       # candidates, shape sets, the script's md5

# Correctness only, CPU, seconds
./scripts/smoke.sh

# The opponent, before writing anything: eager, then torch.compile max-autotune
./scripts/run_ladder.sh

# A recorded sweep of every specialization over the official grid
python verify.py fused-safe fused-sdpa graph-safe graph-sdpa --shapes official --record -- --atol 0.002 --rtol 0.02

# The dispatch table from what was recorded
python kernels/dispatch.py calibrate
```

`verify.py` runs the organizer's script -- its own `main()`, its own output --
with the candidate patched in; flags after `--` go to the script verbatim.
`--record` appends one row per case to `runs/benchmark.csv` with the script's
own numbers, its md5, the git sha and the exact flags. Leave it off for
throwaway runs.

## The gate

The gate is the organizer's own script, run unmodified, at the problem
statement's tolerance, with the shipping candidate patched in:

```bash
./scripts/demo.sh center
```

This prints exactly what the organizers would see and returns their exit code:
0 for pass, 2 for an accuracy failure. Nothing gets reported that has not
cleared this on every official shape it is claimed to pass.

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

1. Set up the repository, run `./scripts/setup_agent.sh`, and confirm the GPU.
2. In tmux, start a fresh Claude Code session **in this directory**.
3. Paste `prompts/phase1.md` and work the loop: the draft in `docs/draft.md`,
   `/humanize:gen-plan`, `/humanize:start-rlcr-loop`. Then phase 2, then
   phase 3, raising the target between rounds.

The search, its dead ends and its evidence all stay in this directory:
`runs/benchmark.csv`, `runs/solutions.jsonl`, `runs/profile/`. They are
committed. Runs are not deterministic -- search order, profiling noise, GPU
scheduling and model behaviour all vary, so a re-run will not reproduce the
same kernels or the same path. What the prompts document is the workflow, not
an outcome.

## The evidence artifacts

No tool produces these; the agent does, from `runs/`, and each is checked in
under `docs/assets/`:

- **The roofline.** Ceilings measured on the card -- a large GEMM for peak
  arithmetic, a large device-to-device copy for bandwidth, both under-reporting
  the spec sheet, which is the right direction of error for a ceiling you claim
  to have approached. Every passing row of `runs/benchmark.csv` is one point.
- **The precision budget.** The table in `docs/precision-budget.md`, from the
  `max_abs`, `atol` and `passed` columns of the final official sweep.
- **The skill ablation.** Three arms, each its own agent session on the same
  shape set with the same wall-clock budget: `runs/ablation/<arm>/benchmark.csv`
  and a `meta.json` recording the wall clock, the iteration count and the tools
  in the loop, so a reader can see what was held fixed rather than take it on
  faith.
- **The search DAG.** `runs/solutions.jsonl` rendered with `dot`, rejected
  branches included.
