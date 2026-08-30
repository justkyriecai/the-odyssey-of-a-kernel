# Starting a Workspace for a New Operator

A workspace is one directory that holds everything about one task: the
evaluator, the shapes it is scored on, the candidates, the prompts, the docs,
the machine it runs on, and the evidence. Nothing outside it knows the task
exists.

```bash
./scripts/new_workspace.sh my-op-h100-20260830
```

creates:

```text
workspace/my-op-h100-20260830/
  README.md            the checklist below, as a file
  verify.py            copied from the first workspace; three parts to adapt
  bench/official/      the evaluator goes here, unmodified, with its md5
  bench/shapes/        smoke.json, dev.json, official.json
  kernels/             candidates; CANDIDATES maps name -> (description, factory)
  prompts/             _shared.md and phase1-3.md from the template
  docs/                what a reader needs to trust the numbers
  scripts/smoke.sh     the correctness gate, CPU, seconds
  runs/                benchmark.csv, solutions.jsonl, profile/, dispatch tables
  infra/               how to get a machine for this task
```

Name a workspace `<task>-<gpu>-<date>`. The card is in the name because the
hardware section of `prompts/_shared.md` is the first thing to rewrite when the
card changes and the most expensive to get wrong: when it changes, the
workspace changes too, and the old one stays as evidence rather than being
edited into a lie.

The `odyssey-create-workspace` skill does all of this from a task description
plus the card, and fills in whatever the description already answers -- the
workspace `CLAUDE.md` and the matching slots in `prompts/_shared.md`. The
checklist below is what is left either way.

## The same task on another card

A workspace is card-specific: the hardware section of `prompts/_shared.md` is
written for one card, and every number in `runs/` was measured on one. So when
the machine changes, the workspace changes too.

```bash
./scripts/new_workspace.sh my-op-b200-20260912 --from my-op-h100-20260830
```

carries over what belongs to the task -- `bench/`, `kernels/`, `verify.py`,
`docs/`, `scripts/`, `infra/`, the prompts -- and deliberately not `runs/`,
which starts empty. The old workspace is not edited and not deleted; it is the
evidence for what happened on that card, and it stays readable as such.

What has to be redone on arrival, in order: the hardware section of
`_shared.md` (then `build_prompts.py`), this round's target, the `passthrough`
control reading ~1.00x, and a re-measurement of the baseline and of every
candidate you intend to build on. A candidate arrived because it won on another
card. That is a hypothesis about this one -- tile sizes, launch bounds and
occupancy assumptions rarely survive a change of card.

## The checklist

**1. Vendor the evaluator.** Copy it into `bench/official/` unmodified and
record its md5 in `bench/official/README.md`. Then read it end to end and write
down what it actually does -- what it measures, what it masks, where the
tolerance binds, whether its own defaults and its documentation agree. The
first workspace's `docs/benchmark-anatomy.md` is the model: eight things the
source says, two of which contradicted the obvious guess.

**2. Write the shape sets.** Three JSON files under `bench/shapes/`, each a
`defaults` block plus `cases`, each case a dict of the evaluator's own
parameters with a `name` and a `note` saying why it is in the set:

- `smoke.json` -- tiny, CPU, seconds. Every combination of the edge conditions
  the evaluator can turn on. A correctness gate, not a performance signal.
- `dev.json` -- the iteration set, anchored on the center of the official grid.
  Cheap enough to run after every change; broad enough that a win is unlikely
  to be a fluke.
- `official.json` -- what gets scored, transcribed from the problem statement.

**3. Adapt `verify.py`.** Three evaluator-specific parts: which name in the
evaluator is replaced by the candidate's class, how a case's fields map to the
evaluator's flags (`CASE_FLAGS`), and the regular expressions that read the
verdict off what the evaluator prints. Everything else -- loading the script by
path, patching, capturing, recording -- stays.

**4. Fill the prompts.** Every `<<slot>>` in `prompts/_shared.md`, then the
task-specific slots in each phase. The hardware section first; it is the most
expensive slot to get wrong. Then `python scripts/build_prompts.py` from the
repository root. See `prompts/README.md` for what each slot is for.

**5. A `passthrough` control.** Before any optimization, a candidate that runs
the reference unchanged. `./scripts/smoke.sh` must read ~1.00x and zero error;
if it does not, the measurement is wrong and nothing else can be trusted.

**6. The machine.** Under `infra/`, whatever it takes to get a card and keep
the evidence when the card goes away. The first workspace's `infra/runpod/` is
one answer.

## Then

Start a fresh agent session in the workspace directory and paste
`prompts/phase1.md`. The agent's first job is to measure the strongest opponent
the evaluator can produce, before writing anything.

## Why the second operator matters

One workspace proves the workflow ran once. Two, on different operator
families, are what let the claim "the method transfers" be a statement rather
than a hope -- which is the difference between a kernel and a framework. Keep
the second one small: a softmax, a LayerNorm, a plain GEMM. The point is not
the kernel.
