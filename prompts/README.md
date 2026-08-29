# Prompts

Three phases, one per stage of the optimization workflow. They are meant to be
pasted into a fresh agent session, one at a time, and re-run with a higher bar
rather than rewritten.

```text
prompts/
  transformer-layer/
    phase1.md   research, then one correct implementation
    phase2.md   profiling-guided optimization
    phase3.md   shape-group specialization
```

## Where to run them

**Not in this repository.** Create a separate implementation workspace and start
the agent there, so the search history, the failed branches, and the scratch
files do not contaminate the release tree:

```bash
mkdir -p workspaces
git clone . workspaces/round-1     # or start from a bare directory
cd workspaces/round-1
export TECHJAM_BENCHMARK="$OLDPWD/bench/official/torch_transformer_benchmark.py"
```

The harness reads `$TECHJAM_BENCHMARK`, so every workspace measures against the
same file, and `md5` proves it.

## The loop

1. Paste the phase prompt into a fresh session.
2. Have the agent investigate first: the official script, the shape grid, the
   profiler output, `KernelWiki`, public documentation.
3. Require the plan draft in `docs/draft.md` **before** any implementation.
4. `/humanize:gen-plan` turns the draft into a detailed plan.
5. `/humanize:start-rlcr-loop` runs implement-and-review.
6. Every performance-relevant commit goes into `benchmark.csv`; every candidate
   goes into `solutions.jsonl` with a parent link; every major direction keeps
   its NCU report.

## Three disciplines worth more than any single optimization

**Five iterations per direction, then move on.** If a direction cannot be
implemented cleanly, fails correctness, or shows no credible path to improvement
after five iterations, record the evidence and take the next-ranked direction.
Over seventy-two hours this is the only thing standing between a search and a
rabbit hole.

**Rank before implementing.** The draft must list candidate directions, order
them by expected benefit against implementation risk, and split each into
concrete subtasks. Not whatever occurs to the agent next.

**Never silently move the goalposts.** The target speedup is set by the human.
The agent's job is to reach it or produce benchmark and profiling evidence for
why this round cannot. It does not get to redefine the target or quietly swap
the baseline for an easier one.

## Raising the bar between rounds

These prompts are starting points, not scripts. Re-invoke the same phase with a
higher explicit target, a stricter validation requirement, or a tighter
promotion rule. Write human knowledge directly into the prompt when you have it:

- Which hardware features to try -- and which do not exist on this card.
- Which bottlenecks you expect, from having read the profile yourself.
- Which directions are known to be risky or unlikely to pay.
- Which shapes matter most this round.

## Shared requirements

Unless a phase overrides them:

- Pass the official correctness check at `--atol 0.001 --rtol 0.01`, the
  stricter of the two readings in the script. Zero bad elements; non-finite
  values fail outright.
- Optimize median latency, and report p90 alongside it.
- Any implementation the contest allows: PyTorch, `torch.compile`, Triton, CUDA
  C++ extensions, CUDA Graphs.
- Keep the official script unmodified. Substitute the candidate through
  `odyssey official <candidate>`, which patches `UserOptimizedTransformer` at
  runtime and calls the script's own `main()`.
- Record everything. A dead end recorded on the day it died is evidence; the
  same dead end reconstructed on the last night is a guess.

## Phase semantics

**Phase 1** -- research and one correct implementation. Speed matters less than
a clean design that passes every shape, including the padded ones.

**Phase 2** -- profiling-guided bottleneck analysis and iterative optimization.
Repeatable with progressively higher explicit targets.

**Phase 3** -- workload-shape analysis and per-group specialization. The problem
statement announces the shape grid in advance, which makes this the phase the
task was practically designed for.
