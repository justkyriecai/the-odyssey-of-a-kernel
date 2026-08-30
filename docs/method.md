# The Method

Three phases, one prompt each, and a loop inside every phase. This page is the
part of the framework that does not change between tasks; the prompts under
`prompts/template/` are its text.

## Three phases

**Phase 1 -- get it right.** Research, then one correct implementation. The
agent reads the evaluator end to end, measures the strongest available
baseline before writing anything, and produces a candidate that passes every
shape in the development set -- including the edge conditions that are
invisible at the evaluator's default flags. Speed is secondary. A clean,
correct design you can profile beats a fast one you cannot trust.

**Phase 2 -- make it fast, with evidence.** Profile first, to learn whether
the workload is launch-bound, memory-bound or compute-bound, because that
decides which whole class of optimization deserves any iterations. Then
enumerate directions, rank them by expected benefit against risk, and work
down the list with the profiler deciding what survives. The target speedup is
set by the human before the phase starts; the agent reaches it or produces the
evidence for why this round cannot. Repeatable with a higher target.

**Phase 3 -- specialize by shape.** When the evaluation shapes are public, or
the workload distribution has been measured, group shapes by bottleneck rather
than by size and specialize only where a measured win justifies the branch.
Then make it safe to ship: a dispatch policy that serves a geometry with a
specialization only if it passed correctness on every case in the group and
its worst speedup clears a margin, and falls back to the reference path
everywhere else. Phase 3 is also when the artifacts that outlive the search get
made -- the roofline, the precision budget, the skill ablation, and a second
unrelated operator through the same three phases.

## The loop inside a phase

1. Paste the phase prompt into a fresh agent session started in the workspace.
2. The agent investigates first: the evaluator, the shape grid, the profiler
   output, `odyssey-kernelwiki`, public documentation.
3. It writes the plan draft to `docs/draft.md` **before** any implementation.
4. `/humanize:gen-plan` turns the draft into a detailed plan.
5. `/humanize:start-rlcr-loop` runs implement-and-review until the plan is
   done or the evidence says it cannot be.
6. Every measurement goes into `runs/benchmark.csv`; every candidate into
   `runs/solutions.jsonl` with a parent link; every major direction keeps its
   profile -- NCU where the machine permits counters, an `nsys` timeline where
   it does not.

The ablation HAN Lab ran after their contest is the reason the loop is the
centre of the method: on a sparse-attention indexer, the plan/execute/verify
harness alone took a bare generation loop from 1.37x to 3.71x, the knowledge
base to 6.14x, and the profiler skill to 8.58x. The largest single contributor
was discipline, not knowledge.

## Three disciplines worth more than any single optimization

**Five iterations per direction, then move on.** If a direction cannot be
implemented cleanly, fails correctness, or shows no credible path to
improvement after five iterations, record the evidence and take the next
ranked direction. This is the only thing standing between a search and a
rabbit hole.

**Rank before implementing.** The draft must list candidate directions, order
them by expected benefit against implementation risk, and split each into
concrete subtasks. Not whatever occurs to the agent next.

**Never silently move the goalposts.** The target speedup is set by the human.
The agent's job is to reach it or produce benchmark and profiling evidence for
why this round cannot. It does not get to redefine the target or quietly swap
the baseline for an easier one.

## The record

Three files, written as the search happens, never reconstructed afterwards:

| File | One entry per | Why it exists |
|---|---|---|
| `runs/benchmark.csv` | measurement | The speedup is reproducible: median and p90, git sha, the evaluator's md5, the flags it ran with |
| `runs/solutions.jsonl` | candidate, with a parent link | The search was a search and not a story: the DAG, including the branches that died and why |
| `runs/profile/<direction>/` | major direction | "It got faster" is not a reason to keep something; knowing which stall reason went away is |

A dead end recorded on the day it died is evidence. Reconstructed on the last
night, it is a guess.

## Raising the bar between rounds

The prompts are starting points, not scripts. Re-invoke the same phase with a
higher explicit target, a stricter validation requirement, or a tighter
promotion rule. Write human knowledge directly into the prompt when you have
it:

- Which hardware features to try -- and which do not exist on this card.
- Which bottlenecks you expect, from having read the profile yourself.
- Which directions are known to be risky or unlikely to pay.
- Which shapes matter most this round.

## What the framework is not

It is not a measurement library. The evaluator measures; the agent runs it and
records what it printed. It is not a kernel. The candidates in a workspace are
that workspace's output. And it is not deterministic: search order, profiling
noise, GPU scheduling and model behaviour all vary, so a re-run documents the
workflow, not an outcome.
