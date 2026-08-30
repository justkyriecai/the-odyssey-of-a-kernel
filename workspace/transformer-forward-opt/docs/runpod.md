# RunPod Runbook

How to rent the development card, keep everything that matters when the card
goes away, and prove the box is usable in the first ten minutes. Prices are
late-August 2026 ballparks -- check the console before booking.

## What to rent

| Session | Card | Where | Why |
|---|---|---|---|
| Development (most hours) | RTX 4090 24 GB | Secure Cloud, on-demand | Covers 13 of the 14 official shapes; the shape profile is launch-bound, so a bigger card makes the problem *less* interesting (`docs/reproduction.md`) |
| Validation (a few hours, late) | A100 80 GB or H100 80 GB | Secure Cloud, on-demand | `stress-100k` needs ~38 GB for q/k/v alone at fp32; plus the cross-hardware convergence comparison |

- **Secure Cloud, not Community**: network volumes only exist there, and an
  agent session carries an API key.
- **On-demand over spot** while an agent session is mid-search. Spot is fine
  for pure re-measurement, since the evidence lands on the volume.
- Total budget stays tens of dollars: ~40 h of 4090 plus a few hours of 80 GB.

## Storage: a network volume, not a volume disk

RunPod offers three kinds of storage on a pod. Only one of them survives the
pod, and it is the one that makes everything else in this runbook cheap.

| | Container disk | Volume disk | **Network volume** |
|---|---|---|---|
| Survives a stop | no | yes | yes |
| Survives a terminate | no | **no** | **yes** |
| Moves to another pod / GPU type | no | no | yes, within its data center |
| Price | $0.10/GB/mo | $0.10/GB/mo running, **$0.20/GB/mo stopped** | **$0.07/GB/mo** |
| Mounted at | system-managed | `/workspace` | `/workspace` (replaces the volume disk) |

The volume disk is the trap: it is tied to one pod, so keeping it means keeping
the pod, and a *stopped* pod bills its disk at double rate while not even
guaranteeing the same host -- or any GPU at all -- when it restarts. A network
volume exists on its own. Terminate the pod when you are done for the day,
create a new one tomorrow on whatever card is free, attach the same volume, and
the tools, the checkout, the venv and the evidence are all there.

What has to be true for that to work:

- **Attach it at pod creation.** A network volume cannot be attached to or
  detached from a running pod. In the deploy dialog, choose **Network Volume**
  first, then the GPU.
- **It pins you to one data center.** The GPU list in the deploy dialog is
  filtered to that volume's location. Before creating the volume, check the
  data center's availability for *both* the 4090 and an 80 GB card, because
  the validation session has to attach the same volume. Moving a volume means
  two pods and an `rsync` between them.
- **Size only grows.** 50 GB is plenty: node, uv and its caches, this checkout
  with its venv, and profiler reports come to a few GB.
- **One pod at a time.** Two pods writing the same volume is how files get
  corrupted. Terminate before you create the next one.
- **It bills while pods do not.** When the account balance hits $0, pods stop
  but the volume keeps accruing; if it stays unpaid the volume is eventually
  deleted and cannot be recovered. Turn on low-balance notifications.
- **It is a network disk**: 200-400 MB/s. Fine for code, packages and evidence;
  not somewhere to stream tensors from. The benchmark's tensors live in GPU
  memory and never touch it.

Two facts about the filesystem itself that bite scripts: it is MooseFS over
FUSE in a user namespace, so `chown` is refused and `cp -p` / `cp -a` fail on
ownership -- use `install`. `infra/runpod/` is written around both.

What lives on it, after `infra/runpod/bootstrap.sh`:

```text
/workspace/
  env.sh                 sourced by every shell: PATH and the uv/npm cache locations
  tools/                 node (nvm), uv, uv's caches and managed pythons, claude, npm cache
  tools/.claude*         Claude Code state, linked from ~ by pod.sh
  odyssey/               this repository, with its .venv
```

## Pod configuration

- **Template**: a PyTorch `-devel` image (e.g. `runpod/pytorch:*-cuda12.x-devel-ubuntu22.04`).
  The `-devel` part carries `nvcc`, which `torch.utils.cpp_extension` needs the
  moment a candidate is CUDA C++ rather than Triton.
- **CUDA filter**: filter hosts to CUDA ≥ 12.6 so the default PyPI torch
  wheel's bundled runtime is satisfied by the host driver.
- **Storage**: **Network Volume** (above), 50 GB. Container disk 20 GB; it holds
  the OS and nothing you care about.
- **SSH**: add your public key in RunPod account settings *before* creating the
  pod; connect with the "SSH over exposed TCP" command the pod page shows.
  The web terminal is the fallback, not the plan.

## First ten minutes

**First pod on a new volume** -- once per volume, ~10 minutes, mostly
downloads:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/justkyriecai/the-odyssey-of-a-kernel.git odyssey
bash odyssey/workspace/transformer-forward-opt/infra/runpod/bootstrap.sh   # tools + venv onto the volume
bash odyssey/workspace/transformer-forward-opt/infra/runpod/pod.sh         # links, .bashrc hook, tmux, D0 gate
```

**Every pod after that** -- the volume already has everything:

```bash
bash /workspace/odyssey/workspace/transformer-forward-opt/infra/runpod/pod.sh
```

`pod.sh` ends by running `scripts/check_gpu.sh`, which is the D0 gate: driver,
torch CUDA build, **whether ncu may profile**, and the CPU smoke test. If it
ends `READY`, continue in order -- opponent first, ceilings second, kernels
never before either:

```bash
source /workspace/env.sh
cd /workspace/odyssey/workspace/transformer-forward-opt
./scripts/run_ladder.sh          # eager, then torch.compile max-autotune, before writing anything
```

## The NCU question -- settle it before booking serious hours

Nsight Compute needs access to GPU performance counters
(`CAP_SYS_ADMIN`, or `NVreg_RestrictProfilingToAdminUsers=0` on the host
driver). **Containerized pods frequently do not have it**, and the failure is
`ERR_NVGPUCTRPERM`, not a missing binary. `check_gpu.sh` probes exactly this;
run it in the first ten minutes of the *first* pod, before committing to the
provider for the week.

If `ncu` is missing but permission is plausible, install it inside the image
(`apt-get update && apt-get install -y cuda-nsight-compute-12-x` matching the
image's CUDA) and re-probe. If the probe still says denied:

- **Plan A -- stay on RunPod, downgrade the evidence, not the discipline.**
  `torch.profiler` works everywhere: per-kernel wall time, launch counts, and
  the timeline that shows gaps between kernels. That is enough to answer the
  first question of Phase 2 -- launch-bound, memory-bound, or compute-bound --
  and enough for the "fewer kernels" narrative. What it cannot give is hardware
  counters (stall reasons, DRAM bytes), which the measured-intensity roofline
  and `ncu-report-skill` want.
- **Plan B -- move profiling sessions to a VM provider.** A provider that hands
  you a VM with root (Lambda, and some others) lets `ncu` run as root or lets
  you relax the driver restriction. A few hours of profiling there, with the
  candidates rsynced over, feeds the NCU evidence while the cheap 4090 hours
  stay on RunPod.

Decide by probe, not by forum thread.

## Timing variance without clock locking

`nvidia-smi --lock-gpu-clocks` needs host root, which a pod does not have.
Accept it: the official protocol (median of 300 samples, order alternated
between rounds, p90 reported) is designed for drift. Re-run the ladder whenever
numbers look off by more than the p90 spread, and treat any cross-day
comparison as suspect unless both ends were re-measured that day.

## Evidence survives; pods do not

`runs/` is the product of a session, it is committed, and with a network volume
it survives a terminate by construction. Two more copies cost nothing:

```bash
# from the pod, when git credentials are set up there
cd /workspace/odyssey && git add workspace/transformer-forward-opt/runs && git commit -m "..." && git push

# from the laptop, at every checkpoint, using the pod's SSH port
rsync -avz -e "ssh -p <PORT>" \
  root@<POD_IP>:/workspace/odyssey/workspace/transformer-forward-opt/runs/ ./runs-pod/
```

The volume is also reachable without a pod through RunPod's S3-compatible API,
which is the way to pull a forgotten file after the last pod is gone.

**Stop vs terminate**: with a network volume, *terminate*. A stopped pod keeps
billing its container disk, may come back with zero GPUs, and buys nothing the
volume does not already keep.

## Running the agent loop on the pod

Long searches must survive an SSH drop:

```bash
tmux new -s odyssey          # reattach later with: tmux attach -t odyssey
```

Inside tmux, on the first pod of a volume:

```bash
cd /workspace/odyssey && ./scripts/setup_agent.sh   # humanize, KernelWiki, ncu-report-skill; verifies each
```

It ends `READY` or `NOT READY`, and a `NOT READY` here is worth more than the
five minutes it costs: a missing `humanize` is the one dependency the ablation
says carries most of the method. `~/.claude` is linked into the volume, so the
plugin, the skills and the login all persist to the next pod.

Then start the session in the workspace directory and paste the phase prompt:

```bash
cd /workspace/odyssey/workspace/transformer-forward-opt && claude
```

One caution: the API key you log in with lives on that volume while the
session runs. Secure Cloud, and `claude logout` (or wipe
`/workspace/tools/.claude*`) before the last pod is terminated.

## Optional: let the agent drive RunPod

RunPod publishes an agent setup at `https://docs.runpod.io/agent-setup.md`
that installs a `runpod` plugin (skills plus a hosted MCP server) into Claude
Code on the laptop, so an agent can list pods, create them and check billing:

```bash
claude plugin marketplace add runpod/runpod-plugins-official
claude plugin install runpod@runpod
```

then `/reload-plugins` and `/mcp` → **runpod** → sign in (OAuth; no API key is
stored). Useful for automating the create-attach-terminate cycle once the
volume exists. It is not needed for anything above.

## Teardown checklist

- [ ] `runs/` committed, or `rsync`ed to the laptop and spot-checked
- [ ] measured ceilings for this card saved under `runs/` (they ride along)
- [ ] `claude logout` if this is the last pod on the volume for a while
- [ ] pod **terminated**, not stopped
- [ ] volume left attached to nothing, billing $0.07/GB/mo until the next pod
