# Agent Instructions

This repository is an agent harness for kernel optimization: phase prompts,
research skills, and the rules below. Each task lives in `workspace/<name>/`
with its own `CLAUDE.md`, which adds the task's hard facts; read it before
touching anything in that directory.

## Repository rules

- English for everything repository-facing: code, comments, docs, prompts,
  commit messages. This will be read by people who did not write it.
- **Never modify a vendored evaluator** (`workspace/*/bench/official/`). Its md5
  is recorded next to it. A number is only a number if it came from that file.
- **Never re-implement an evaluator's correctness rule or timing protocol.**
  Run the evaluator; read its verdict. Each workspace's `verify.py` exists for
  exactly that: it calls the evaluator's own `main()` with a candidate patched
  in and returns its exit code.
- Generated evidence lives under `workspace/<name>/runs/` and is committed:
  `benchmark.csv`, `solutions.jsonl`, dispatch tables, text exports of
  profiles. Binary profiler dumps (`*.ncu-rep`, `*.nsys-rep`) are not.
- `ref/` is reference material and is git-ignored.
- Candidates already present in a workspace's `kernels/` are prior output of
  this workflow, not a specification. Measure them before trusting them.

## Expected workflow for a task

1. Environment: `./scripts/setup_env.sh`, `./scripts/setup_agent.sh`, then
   `./scripts/check_gpu.sh` on the GPU box -- the profiler permission probe in
   it is the reason to run it on day zero. `odyssey-onboarding` walks a first
   clone through it; `odyssey-create-workspace` starts the task directory.
2. Work inside `workspace/<name>/`. Read its `README.md` and `CLAUDE.md`, then
   the evaluator under `bench/official/` end to end.
3. Read the phase prompt under `prompts/`. Consult the workspace's `docs/` when
   the evaluator's rules are unclear.
4. Use `odyssey-kernelwiki` for kernel research and `odyssey-ncu-report` for
   profiling -- including its fallback path when the machine denies counters.
5. Draft the plan in `docs/draft.md`, then `/humanize:gen-plan`, then
   `/humanize:start-rlcr-loop`.
6. Every measurement goes to `runs/benchmark.csv` (`verify.py --record`).
   Every candidate goes to `runs/solutions.jsonl` with a parent link, forming a
   DAG; rejected branches included. Every major direction keeps a profile under
   `runs/profile/` -- NCU where counters are permitted, an `nsys` timeline or
   `torch.profiler` table where they are not, naming which it was.

<!-- BEGIN odyssey-skills -->
## Skills

Skills live in `skills/` and are linked into `.claude/skills/` by
`./scripts/setup_agent.sh`. Run it after cloning, and again after pulling a
change to `skills/`. They are project-scoped: nothing is installed into
`~/.claude/`.

| Skill | Use it when |
|---|---|
| `odyssey-onboarding` | First clone on a machine, or the skills stopped loading |
| `odyssey-create-workspace` | Starting a task: creates `workspace/<task>-<gpu>-<date>/` |
| `odyssey-kernelwiki` | Kernel research: Blackwell/Hopper techniques, PR references |
| `odyssey-ncu-report` | Profiling: Nsight Compute, and the `nsys` / `torch.profiler` fallback when counters are denied |

The three phases are prompts, not skills: paste
`workspace/<name>/prompts/phase1.md` into a fresh session started in that
workspace directory, then phase 2, then phase 3. Inside a phase, run humanize's
loop -- draft in `docs/draft.md`, `/humanize:gen-plan`,
`/humanize:start-rlcr-loop`.
<!-- END odyssey-skills -->

## Disciplines

- **Five iterations per direction, then move on.** If a direction cannot be
  implemented cleanly, fails correctness, or shows no credible path after five,
  record the evidence with `decision: reject` and take the next-ranked
  direction.
- **Rank before implementing.** The draft lists candidate directions, orders
  them by expected benefit against implementation risk, and splits each into
  subtasks.
- **Never silently move the goalposts.** The target speedup is set by the human.
  Reach it, or produce benchmark and profiling evidence for why this round
  cannot. Do not redefine the target or swap in an easier baseline.
- **A speed/accuracy trade is a separate named candidate**, never a flag on an
  existing one, so calibration can measure both and admit whichever is correct
  *and* faster.
- **Report the median with p90 next to it**, never a best-of run. Always name
  the baseline. Run the full shape set, not the one case that looks good.

## Prompts

After editing any `_shared.md`, run `python scripts/build_prompts.py` -- the
phase prompts each carry an expanded copy so they stay self-contained when
pasted into a session.

## Commits

Commit messages describe the change and match the existing history.
