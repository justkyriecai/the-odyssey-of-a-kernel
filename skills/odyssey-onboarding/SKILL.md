---
name: odyssey-onboarding
description: Configure this repository the first time it is cloned onto a machine - link the repository's skills into the project's .claude/skills/ so they load, install the humanize plugin, and write the project usage and skill routing into the project CLAUDE.md (or AGENTS.md). Use when the user has just cloned the odyssey harness, says the odyssey skills are not loading, asks how to set the repository up, or asks to refresh the skill routing block.
allowed-tools: "Bash Read Edit Write Glob Grep"
---

# Odyssey Onboarding

Everything this skill installs is **project-scoped**. Nothing is written to
`~/.claude/`, and nothing outside the repository is modified except the
`humanize` plugin, which the CLI only knows how to install at user scope. Two
odyssey checkouts on one machine get two independent setups.

Run this once per clone. It is idempotent: re-running relinks and rewrites
rather than duplicating.

## 1. Confirm where you are

```bash
git rev-parse --show-toplevel   # the repository root; everything below is relative to it
ls skills/                      # odyssey-create-workspace  odyssey-kernelwiki  odyssey-ncu-report  odyssey-onboarding
```

If `skills/` is missing, this is not the odyssey harness. Stop and say so.

## 2. Link the repository's skills into the project

Claude Code loads project skills from `<repo>/.claude/skills/`. The skills are
maintained in `skills/`, so the link is what makes them live. `.claude/` is
git-ignored, which is why this is a setup action and not a committed file.

```bash
./scripts/setup_agent.sh
```

That is the whole step: the script links every `skills/*/` directory that has a
`SKILL.md` into `.claude/skills/`, installs `humanize`, and prints READY or
NOT READY. Read its output. Do not report success if it printed NOT READY.

The equivalent by hand, from the repository root, if the script is unavailable:

```bash
mkdir -p .claude/skills
for d in skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$(cd "$d" && pwd)" ".claude/skills/$(basename "$d")"
done
```

Then check that every name resolves:

```bash
for d in .claude/skills/*/; do
  printf '%s -> %s\n' "$(basename "$d")" "$(grep -m1 '^name:' "$d/SKILL.md")"
done
```

The directory name and the `name:` field must agree, or the skill loads under a
name nobody will type.

## 3. Write the routing block into the project CLAUDE.md

The repository already has a `CLAUDE.md` holding the working rules. Onboarding
adds one section to it; it does not rewrite the rules. If the project uses
`AGENTS.md` instead, write the block there. If both exist, put it in
`CLAUDE.md` and leave `AGENTS.md` pointing at it.

Insert this section, or replace it if the markers are already present:

```markdown
<!-- BEGIN odyssey-skills -->
## Skills

Skills live in `skills/` and are linked into `.claude/skills/` by
`./scripts/setup_agent.sh`. Run it after cloning, and again after pulling a
change to `skills/`.

| Skill | Use it when |
|---|---|
| `odyssey-onboarding` | First clone on a machine, or the skills stopped loading |
| `odyssey-create-workspace` | Starting a task: creates `workspace/<task>-<gpu>-<date>/` |
| `odyssey-kernelwiki` | Kernel research: Blackwell/Hopper techniques, PR references |
| `odyssey-ncu-report` | Reading an Nsight Compute report, deciding what to optimize next |

The three phases are prompts, not skills: paste
`workspace/<name>/prompts/phase1.md` into a fresh session started in that
workspace directory, then phase 2, then phase 3. Inside a phase, run humanize's
loop -- draft in `docs/draft.md`, `/humanize:gen-plan`,
`/humanize:start-rlcr-loop`.
<!-- END odyssey-skills -->
```

Keep the markers. They are how a re-run finds the block instead of appending a
second copy.

## 4. Report what happened

Say which skills are linked, whether `humanize` installed, whether the routing
block was added or updated, and that the session has to be restarted before the
newly linked skills load. Then point at the next step:
`odyssey-create-workspace`.

## What this skill does not do

- It does not install PyTorch or touch the venv. That is `./scripts/setup_env.sh`.
- It does not check the GPU. That is `./scripts/check_gpu.sh`, and it is the
  day-zero gate because of the profiler permission probe in it.
- It does not create a workspace. That is `odyssey-create-workspace`.
