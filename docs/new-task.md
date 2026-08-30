# Starting a Workspace for a New Operator

A workspace is one directory that holds everything about one task: the
evaluator, the shapes it is scored on, the candidates, the prompts, the docs,
the machine it runs on, and the evidence. Nothing outside it knows the task
exists.

```bash
./scripts/new_workspace.sh my-op
```

creates:

```text
workspace/my-op/
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
