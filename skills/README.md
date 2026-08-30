# Skills

Four skills, all prefixed `odyssey-`, all maintained here and linked into
`.claude/skills/` by `./scripts/setup_agent.sh`. The link is project-scoped:
nothing is installed into `~/.claude/`, so two checkouts on one machine stay
independent, and editing a file here changes what the next session loads.

| Skill | Use it when |
|---|---|
| `odyssey-onboarding` | First clone on a machine, or the skills stopped loading |
| `odyssey-create-workspace` | Starting a task: creates `workspace/<task>-<gpu>-<date>/` |
| `odyssey-kernelwiki` | Kernel research: Blackwell/Hopper techniques, PR references |
| `odyssey-ncu-report` | Reading an Nsight Compute report, deciding what to optimize next |

The first two are this repository's own. The last two are copies of
[KernelWiki](https://github.com/mit-han-lab/KernelWiki) and
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill), vendored
rather than tracked as submodules so their names can carry the prefix and so a
clone needs no `--recurse-submodules`. Each records its upstream commit in its
own `VENDORED.md`; that file is where to look before pulling a newer revision.

A skill's directory name and the `name:` in its `SKILL.md` must agree.
`setup_agent.sh` fails loudly when they do not, because a mismatch loads the
skill under a name nobody will type.

Adding one: a directory here with a `SKILL.md`, then re-run
`./scripts/setup_agent.sh`, then add a row to the routing table in the root
`CLAUDE.md` -- the block between the `odyssey-skills` markers, which
`odyssey-onboarding` rewrites in place.
