---
name: odyssey-create-workspace
description: Start a task workspace under workspace/, or move an existing task to a different GPU. Ask the user for a task description and the card, name the directory <task>-<gpu>-<date>, create it from the template (or branch it from a sibling workspace with --from), and fill in what the description already answers. Use when the user wants to start a kernel task, has an evaluator to optimize against, or has moved the same task onto another machine.
allowed-tools: "Bash Read Edit Write Glob Grep AskUserQuestion"
---

# Odyssey: Create a Workspace

A workspace is one directory holding everything about one task **on one card**:
the evaluator, the shapes it is scored on, the candidates, the prompts, the
docs, the machine it runs on, and the evidence. Nothing outside it knows the
task exists.

This skill creates that directory. It does not vendor the evaluator or write
the shape sets -- those need the evaluator in hand, and the workspace README
carries them as a checklist.

## 1. Ask for the task description

Ask for one thing:

> Describe the task: what is being optimized, what evaluator scores it, and
> what the target is. Paste the problem statement if you have one.

Take whatever comes back. A pasted contest description, three sentences, a
link -- all fine. Read it and pull out what it happens to answer: the operator,
the evaluator and who provides it, what the candidate replaces, the target and
its baseline, the shape grid. **Do not interrogate the user for the rest.**
Anything the description does not answer stays a `<<slot>>` in the files this
creates, where the next session can see it is missing. An unfilled slot the
agent can see beats a guess it cannot.

Then two short questions, because they decide the directory name and cannot be
inferred:

- **The card.** `h100`, `b200`, `a6000`, `a100`, `rtx4090`.
- **A short name**, if the description does not obviously supply one:
  lowercase letters, digits and hyphens. `transformer-forward`, `sparse-attn`.

## 2. Check whether this task already has a workspace

```bash
ls workspace/ 2>/dev/null
```

If a directory exists whose short name matches the one you are about to use,
this is **the same task on a different card** -- go to section 4. Ask the user
if it is ambiguous; a task can legitimately be restarted from scratch, and
branching the wrong one is worse than asking.

## 3. New task: create it

The name is:

```text
<short-name>-<gpu>-<YYYYMMDD>
```

for example `transformer-forward-b200-20260830`, with today's date from
`date -u +%Y%m%d`. Confirm it with the user, then:

```bash
./scripts/new_workspace.sh <short-name>-<gpu>-<date>
```

from the repository root. It creates the skeleton, copies the three phase
prompts and `_shared.md` from `prompts/template/`, and writes a `CLAUDE.md` and
`README.md` of slots plus a `scripts/smoke.sh`. If the checkout has no earlier
workspace to copy a `verify.py` from, say so plainly: it has to be written from
`docs/new-task.md` rather than adapted.

Then fill in, from the description and nowhere else:

- **`workspace/<name>/CLAUDE.md`** -- the task line, the evaluator, what the
  candidate replaces, the card and machine, the target and who set it. Leave
  every slot the description did not answer, and list it under "Still open".
- **`workspace/<name>/prompts/_shared.md`** -- the same slots, in the prompt's
  own wording. Then, from the repository root:

  ```bash
  python scripts/build_prompts.py
  ```

  which expands `_shared.md` into all three phase prompts. Re-run it after
  every later edit to that file, or the three phases quietly disagree.
- **`workspace/<name>/infra/README.md`** -- one paragraph on how to get this
  machine back: provider, image, CUDA version, what is on the network volume.

## 4. Same task, new card: branch it

The card is in the directory name because a workspace is card-specific: the
hardware section of `prompts/_shared.md` is written for one card, and every
number in `runs/` was measured on one card. When the machine changes, the
workspace changes too. The old one is not edited and not deleted -- it stays as
the evidence for what happened on that card.

```bash
./scripts/new_workspace.sh <short-name>-<new-gpu>-<date> --from <old-workspace>
```

`--from` carries over what belongs to the task -- `bench/`, `kernels/`,
`verify.py`, `docs/`, `scripts/`, `infra/` and the prompts -- and deliberately
does not carry `runs/`, which starts empty. It writes a `CLAUDE.md` and
`README.md` that name the sibling and list what has to be redone.

Then, in this order:

1. **Rewrite the hardware section of `prompts/_shared.md`** for the new card,
   and re-run `python scripts/build_prompts.py`. Until this is done the prompts
   describe a card the agent is not sitting on, which is the most expensive
   error in the repository.
2. **Set this round's target** in `CLAUDE.md`. The sibling's target does not
   carry over: a different card is a different baseline and a different
   denominator. Ask the user for the number; do not reuse the old one silently.
3. **Re-run the correctness gate.** `./scripts/smoke.sh` must read ~1.00x with
   zero error on the `passthrough` control here before any carried-over number
   is quoted.
4. **Re-measure the strongest available baseline, then the candidates you
   intend to build on.** A candidate arrived in `kernels/` because it won on
   another card; that is a hypothesis about this one. Tile sizes, launch
   bounds and occupancy assumptions rarely survive a change of card.
5. **Confirm the evaluator's md5 is unchanged** from the sibling. If the
   organizer shipped a new one, this is a new evaluator and a new task.

Record the branch in `runs/solutions.jsonl` the same way any other lineage is
recorded: the carried-over candidates keep their parent links, and the first
measurement on the new card is a new node, not an edit of an old one.

## 5. Hand off

Print the workspace path and the remaining checklist from its `README.md`, in
the order of what blocks what. For a new task that is: vendor the evaluator
(unmodified, md5 recorded, read end to end), write
`bench/shapes/{smoke,dev,official}.json`, adapt `verify.py`, then a
`passthrough` candidate reading ~1.00x. Until that holds, no other number can
be trusted.

Then: a fresh session started in the workspace directory, `prompts/phase1.md`
pasted into it.

Do not start Phase 1 yourself in this session. The phase prompts are pasted
into a fresh session by design -- the context they assume is the workspace, not
this conversation.
