# <<Task>> -- Phase 1: Get It Right

Produce a correct replacement for <<what>> that passes every shape in the
development set, including <<the edge conditions the evaluator can turn on>>.
Speed is secondary this phase. A clean, correct design you can profile beats a
fast one you cannot trust.

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
- Keep an NCU report per major direction under `runs/profile/<direction>/`,
  with a text export next to the binary.
- At most five iterations per direction. Then record the evidence and move on.
- Use `KernelWiki` for kernel research and `ncu-report-skill` for reading
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

## Phase 1 Goal

Research first, then implement once.

**Research.** Read the evaluator end to end before writing anything -- what is
measured, what is masked, where the tolerance actually binds. Then read what is
already known about <<this operator family>>: <<the two or three obvious
directions and their known pitfalls>>. Use KernelWiki.

**Establish the opponent before you build.** <<How to measure the strongest
available baseline with the evaluator's own flags -- e.g. its compiled mode.>>
That is the number anyone in the room will ask about. Do not write a kernel
before you know it.

**Implement.** One correct candidate. Priorities in order:

1. Correct under every <<edge condition>>. <<Name the sites that are invisible
   at default flags.>>
2. Correct at every dtype the evaluator can select. Measure -- do not assume
   either way.
3. Only then, the free structural wins: <<...>>.

Nothing in this phase should trade accuracy for speed. If a candidate needs that
trade, it belongs in a separate candidate whose name says so, measured against
the one that does not.

**Verify.** Every time:

```bash
./scripts/smoke.sh                                    # every edge combination, CPU, seconds
python verify.py <candidate> --shapes dev --record    # the evaluator's own main(), recorded
```

## Draft First

Write the implementation plan to `docs/draft.md` before implementing. It must
contain: what the reference computation actually is (op by op, including the
sites that are invisible at default flags), which structural inefficiencies you
found in the reference and what each is worth, the candidate directions ranked
by expected benefit against risk, and how you will tell a real improvement from
timing noise.

Then run `/humanize:gen-plan` on that draft.
