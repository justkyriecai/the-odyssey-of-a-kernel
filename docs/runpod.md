# RunPod Runbook

How to rent the development card, prove the box is usable in the first ten
minutes, and leave without losing evidence. Prices below are late-August 2026
ballparks -- check the console before booking.

## What to rent

| Session | Card | Where | Why |
|---|---|---|---|
| Development (most hours) | RTX 4090 24 GB | Secure Cloud, on-demand | Covers 13 of the 14 official shapes; the shape profile is launch-bound, so a bigger card makes the problem *less* interesting (`docs/reproduction.md`) |
| Validation (a few hours, late) | A100 80 GB or H100 80 GB | Secure Cloud, on-demand | `stress-100k` needs ~38 GB for q/k/v alone at fp32; plus the cross-hardware convergence comparison |

- **Secure Cloud over Community** for the main sessions: Community is cheaper
  (~$0.3/hr vs ~$0.5-0.7/hr for a 4090) but runs on third-party hosts — fine
  for throwaway measurement, not where an Anthropic API key lives while the
  agent loop runs.
- **On-demand over spot** while an agent session is mid-search. Spot is fine
  for pure re-measurement runs where `runs/` is synced continuously.
- Total budget stays tens of dollars: ~40 h of 4090 plus a few hours of 80 GB.

## Pod configuration

- **Template**: a PyTorch `-devel` image (e.g. `runpod/pytorch:*-cuda12.x-devel-ubuntu22.04`).
  The `-devel` part matters: it carries `nvcc`, which `torch.utils.cpp_extension`
  needs the moment a candidate is CUDA C++ rather than Triton.
- **CUDA filter**: when deploying, filter hosts to CUDA ≥ 12.6 so the default
  PyPI torch wheel's bundled runtime is satisfied by the host driver.
- **Disks**: 20 GB container disk, 30+ GB volume mounted at `/workspace`.
  Everything you care about lives under `/workspace` — the container disk does
  not survive a terminate.
- **SSH**: add your public key in RunPod account settings *before* creating the
  pod; connect with the "SSH over exposed TCP" command the pod page shows.
  The web terminal is the fallback, not the plan.

## First ten minutes (the D0 gate)

```bash
cd /workspace
git clone https://github.com/justkyriecai/the-odyssey-of-a-kernel.git
cd the-odyssey-of-a-kernel

# uv is not preinstalled
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

./scripts/setup_env.sh        # default PyPI wheel bundles CUDA 12.x; no TORCH_INDEX needed
./scripts/check_gpu.sh        # driver, torch CUDA build, AND the ncu permission probe
```

If `check_gpu.sh` ends `READY`, continue in order — opponent first, ceilings
second, kernels never before either:

```bash
.venv/bin/python -m odyssey peak --dtype float32   # measured ceilings, and again for bfloat16
./scripts/run_ladder.sh                            # L0 / L1 / L2 before writing anything
```

## The NCU question — settle it before booking serious hours

Nsight Compute needs access to GPU performance counters
(`CAP_SYS_ADMIN`, or `NVreg_RestrictProfilingToAdminUsers=0` on the host
driver). **Containerized pods frequently do not have it**, and the failure is
`ERR_NVGPUCTRPERM`, not a missing binary. `check_gpu.sh` probes exactly this;
run it in the first ten minutes of the *first* pod, before committing to the
provider for the week.

If `ncu` is missing but permission is plausible, install it inside the image
(`apt-get update && apt-get install -y cuda-nsight-compute-12-x` matching the
image's CUDA) and re-probe. If the probe still says denied:

- **Plan A — stay on RunPod, downgrade the evidence, not the discipline.**
  `torch.profiler` works everywhere: per-kernel wall time, launch counts, and
  the timeline that shows gaps between kernels. That is enough to answer the
  first question of Phase 2 — launch-bound, memory-bound, or compute-bound —
  and enough for the "fewer kernels" narrative. What it cannot give is hardware
  counters (stall reasons, DRAM bytes), which the roofline's measured-intensity
  variant and `ncu-report-skill` want.
- **Plan B — move profiling sessions to a VM provider.** A provider that hands
  you a VM with root (Lambda, and some others) lets `ncu` run as root or lets
  you relax the driver restriction. A few hours of profiling there, with the
  candidates rsynced over, feeds the NCU evidence while the cheap 4090 hours
  stay on RunPod.

Decide by probe, not by forum thread.

## Timing variance without clock locking

`nvidia-smi --lock-gpu-clocks` needs host root, which a pod does not have.
Accept it: the official protocol (median of 300 samples, order alternated
between rounds, p90 reported) is designed for drift. Prefer Secure Cloud,
re-run the ladder whenever numbers look off by more than p90 spread, and treat
any cross-day comparison as suspect unless both ends were re-measured that day.

## Evidence survives; pods do not

`runs/` is the product of a session and it is git-ignored by design. Sync it
*to your laptop* at every checkpoint, not at teardown:

```bash
# from the laptop, using the pod's SSH port
rsync -avz -e "ssh -p <PORT>" root@<POD_IP>:/workspace/the-odyssey-of-a-kernel/runs/ ./runs-pod/
```

`runpodctl send` / `receive` is the alternative when SSH is awkward. Git
credentials on the pod are optional — commits to the release repo happen from
the laptop; agent workspaces under `workspaces/` are local to the pod and their
worthwhile output is `runs/` evidence, which rsync carries.

**Stop vs terminate**: *Stop* releases the GPU but keeps (and bills) the
volume, and the same host may not be free later. *Terminate* releases
everything — only after the rsync.

## Running the agent loop on the pod

Long searches must survive an SSH drop:

```bash
tmux new -s odyssey          # reattach later with: tmux attach -t odyssey
```

Inside tmux, install Claude Code and the workflow dependencies from
`README.md` (humanize plugin, KernelWiki and ncu-report-skill under
`~/.claude/skills/`), then create the implementation workspace per
`docs/reproduction.md` — the search runs in `workspaces/`, never in the
release clone — and paste `prompts/transformer-layer/phase1.md`.

One caution: the API key you log in with lives on that pod while the session
runs. Secure Cloud, and log out (`claude logout` or wipe the credential file)
before terminating.

## Teardown checklist

- [ ] `rsync` of `runs/` completed and spot-checked on the laptop
- [ ] anything worth keeping from `workspaces/` synced (the evidence usually is `runs/`)
- [ ] `odyssey peak` output for this card saved (`runs/measured_peak.json` rides along)
- [ ] agent credentials removed from the pod
- [ ] pod **terminated**, not left stopped, unless you are returning within hours
