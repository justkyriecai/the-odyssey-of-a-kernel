# Skill-ablation arms (protocol; not run in this session)

The methodology deliverable compares three arms, each a FRESH agent session
running phase 1+2 on this workspace from the same starting commit, same card:

- arm-full/      harness + skills + humanize loop (this campaign's setup)
- arm-noskills/  harness prompts only; odyssey-* skills unlinked
- arm-bare/      a plain agent session given only the evaluator and the goal

Each arm directory gets: its `benchmark.csv`, its wall-clock log, and the
commit range it produced. Arms must not share inductor caches
(`TORCHINDUCTOR_CACHE_DIR` per arm) or dispatch tables. This session cannot be
its own control group -- the arms need sessions that have not seen this
campaign's findings.
