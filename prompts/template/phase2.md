# <<Task>> -- Phase 2: Make It Fast, With Evidence

Start from the best correct Phase 1 candidate. This phase is exploration:
enumerate plausible optimization directions, rank them, and work down the list
with profiler evidence deciding what survives.

Target speedup for this round: **set explicitly by the human before starting.**
Reach it, or produce benchmark and profiling evidence for why this round cannot.
Do not redefine the target and do not substitute an easier baseline.

<!-- BEGIN shared -->

## Kernel Information

- Task: <<one line: the contest or the internal request, and the operator>>.
- Evaluator: `bench/official/<<script>>` (provided by <<whom>>, unmodified,
  `md5 <<...>>`).
- What you replace: <<the class or function the evaluator calls, and how wide
  the scope is -- one kernel, one layer, the whole forward>>.
- Signature, fixed: <<...>>.
- Weights and inputs: <<how the reference and the candidate are made to compute
  the same thing -- copied state dict, shared inputs, seeds -- and what that
  forbids, e.g. renaming parameters>>.

### Variable axes

<<The parameters the evaluator can move, and the grid it will be scored on:
where the shapes come from, what the center is, which axes are swept, which
are not in the grid and default to script flags. Say what the shape profile
implies about the bottleneck regime before the agent guesses.>>

### The reference computation, exactly

<<The reference, op by op, numbered. Include every place a reimplementation
diverges silently: reductions done in a wider dtype, masking applied at more
than one site, approximations the reference does not make. Then name the steps
that are invisible at the default flags -- those are where the agent will
pass its own tests and fail the evaluator.>>

## Official Acceptance

<<The correctness rule, quoted from the evaluator's source, and what it
implies: per element or aggregate, what fails outright, which of two readings
the workspace targets and why.>>

Run the gate through the evaluator's own `main()`:

```bash
python verify.py <candidate> --shapes dev --case <<case>> -- <<strict flags>>
```

## Development shapes

Use these before running the full grid. Cheap enough to run after every
change; broad enough that a win here is unlikely to be a fluke.

| Case | Deviation from center | Why it is in the set |
|---|---|---|
| <<name>> | <<...>> | <<...>> |

<<Which official cases live only in the official set, and why they are not
iterated on: too slow, needs a bigger card, needs a chunked reference.>>

## Workflow Requirements

- Record every measurement in `runs/benchmark.csv`. `python verify.py ...
  --record` appends the evaluator's own numbers; anything measured another way
  is recorded by hand in the same columns, never summarized from memory.
- Record every candidate in `runs/solutions.jsonl` with a parent link, forming
  a DAG. One JSON object per line: `node_id`, `parent`, `candidate`, `case`,
  `status` (`pass` / `fail` / `skipped` / `error`), `speedup`,
  `max_abs_error`, `decision` (`keep` / `reject` / `park`), `evidence` (a
  path, a commit, a run), `notes`, `timestamp`. Rejected branches included --
  especially rejected branches.
- Keep a profile per major direction under `runs/profile/<direction>/`, with a
  text export next to the binary dump. NCU when counters are available; when
  they are not -- `ERR_NVGPUCTRPERM` is the normal case on a rented pod -- an
  `nsys` timeline or a `torch.profiler` table, with the report saying which
  instrument produced it and what is therefore not measured. See
  `odyssey-ncu-report`.
- At most five iterations per direction. Then record the evidence and move on.
- Use `odyssey-kernelwiki` for kernel research and `odyssey-ncu-report` for reading
  profiles.
- Do not modify the evaluator. Do not tune against a single shape and report
  it as a general result. Do not report a best-of run; report the median, with
  p90 next to it. Do not redefine the target and do not substitute an easier
  baseline.

## Hardware

**Development card: <<name (architecture, sm_XX, SMs, memory, bandwidth)>>.**

Available and worth trying:

- <<the features this card has that matter for this operator>>

**Not available on this card.** Do not spend iterations on:

- <<features of newer architectures the agent will otherwise chase>>

<<Anything out of scope by decision rather than capability, stated as such,
with the reason.>>

> **Replace this entire section when the card changes.** Prompts written for one
> architecture send an agent chasing features that do not exist on another. This
> is the first thing to adapt, and the most expensive to get wrong.

<!-- END shared -->

## Phase 2 Goal

**Profile before optimizing.** Collect a profile for the central shape and read
it. NCU if this machine allows counters; if it answers `ERR_NVGPUCTRPERM`, that
is the platform's configuration and not something to retry -- switch to the
`nsys` timeline path in `odyssey-ncu-report`, which answers the question below
without counters, and record the limitation. The question to answer first is not "which kernel is slowest" but
"is this workload launch-bound, memory-bound, or compute-bound" -- because the
answer determines which whole class of optimization is worth any iterations at
all. <<State the prior for this shape profile, and tell the agent not to assume
the obvious kernel is the bottleneck.>>

**Directions worth ranking** (rank them yourself, with reasons; this is a
starting list, not an ordering):

- <<direction>>
- <<direction>>
- <<direction>>

**Five iterations per direction.** If it cannot be implemented cleanly, fails
correctness, or shows no credible path after five, record the evidence in
`runs/solutions.jsonl` with `decision: reject` and take the next direction. This
limit is not a suggestion; it is the only thing that bounds a search.

**Evidence per direction.** Before/after numbers on every development shape,
plus enough profile evidence to say *why* it helped or did not -- the stall
reason that went away, or, without counters, the launches that stopped
happening and the gaps that closed. "It got faster" is not a reason to keep
something.

**Guard against the two ways this goes wrong.** Tuning against one shape until
everything else regresses -- run the full development set, not one case. And
reporting a best-of run -- the median is the metric, p90 is the honesty check.

## Draft First

Write the plan to `docs/draft.md`: the ranked directions with expected benefit
and implementation risk for each, the subtasks each decomposes into, the
profiling evidence that would confirm or kill it, and the target for this round.

Then run `/humanize:gen-plan`, then `/humanize:start-rlcr-loop`.
