# Prompts

The framework's main artifact. Three phases, one prompt each, pasted into a
fresh agent session one at a time and re-run with a higher bar rather than
rewritten. `docs/method.md` explains the phases; this directory holds the text.

```text
prompts/
  template/
    _shared.md    the block every phase carries: the task, the acceptance rule,
                  the shapes, the workflow requirements, the hardware
    phase1.md     research, then one correct implementation
    phase2.md     profiling-guided optimization
    phase3.md     shape-group specialization
workspace/<task>/prompts/
    the same four files, filled in for that task
```

## How a workspace gets its prompts

`scripts/new_workspace.sh <name>` copies `template/` into
`workspace/<name>/prompts/`. Fill every `<<slot>>` in `_shared.md` and the
task-specific slots in each phase, then from the repository root:

```bash
python scripts/build_prompts.py
```

That expands `_shared.md` into each phase file between the
`<!-- BEGIN shared -->` / `<!-- END shared -->` markers. Prompts are pasted
into sessions as raw text, so each phase file has to be self-contained; the
shared block is edited in one place and copied into three so the three stay
in agreement. Re-run it after every edit to a `_shared.md`.

`workspace/transformer-forward-opt/prompts/` is the filled-in example.

## Writing the slots

The slots that decide whether the agent succeeds, in order of how expensive
they are to get wrong:

1. **Hardware.** A prompt written for one card sends the agent chasing
   features another card does not have. List what is there, list what is not,
   and say what is out of scope by decision rather than capability.
2. **The reference computation, exactly.** Op by op. Every site where a
   reimplementation diverges silently -- a reduction in a wider dtype, masking
   applied more than once, an approximation the reference does not make -- and
   which of those are invisible at the evaluator's default flags. Human
   knowledge the agent would otherwise have to pay for in iterations.
3. **The acceptance rule**, quoted from the evaluator's source, with what it
   rules out.
4. **The development shapes**, each with the reason it is in the set.
5. **The Phase 2 direction list.** A starting list the agent must rank, not
   an ordering.

Keep the Workflow Requirements block as it is. It is the method.
