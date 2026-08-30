#!/usr/bin/env bash
# Install the agent workflow: the plan/execute/verify harness and this
# repository's skills. Nothing here knows what operator is being optimized --
# this is the method's toolchain, and it is the same for every workspace.
#
# The skills are installed **project-scoped**, into <repo>/.claude/skills/, so
# two checkouts on one machine stay independent and nothing in $HOME is
# rewritten. `humanize` is the exception: the CLI only installs plugins at user
# scope.
#
# Run it on the rented box, inside tmux, before pasting a phase prompt. The
# first ten minutes of a pod are the most expensive ten minutes of the project;
# a manual checklist is where they go.
#
# Idempotent: re-running relinks and updates rather than duplicating.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="${SKILLS_DIR:-$ROOT_DIR/.claude/skills}"
status=0

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '   %s\n' "$1"; }
bad()  { printf '   %s\n' "$1"; status=1; }

step "claude CLI"
if command -v claude >/dev/null 2>&1; then
  ok "$(claude --version)"
else
  bad "claude not found. Install Claude Code first: https://claude.com/claude-code"
  exit 1
fi

step "humanize plugin (the plan/execute/verify harness)"
# The ablation says this is the single largest contributor -- larger than the
# knowledge base and the profiler skill combined. It is not optional.
if claude plugin list 2>/dev/null | grep -q "humanize"; then
  ok "already installed"
  claude plugin update humanize@PolyArch >/dev/null 2>&1 && ok "updated" || true
else
  claude plugin marketplace add PolyArch/humanize >/dev/null 2>&1 \
    && ok "marketplace PolyArch added" \
    || ok "marketplace PolyArch already present"
  if claude plugin install humanize@PolyArch --scope user --yes >/dev/null 2>&1; then
    ok "installed"
  else
    bad "install failed -- run interactively: /plugin install humanize@PolyArch"
  fi
fi

step "skills -> $SKILLS_DIR"
# The skills are maintained in skills/, next to the prompts that cite them, and
# linked into the project so Claude Code loads them. Editing skills/ edits what
# the session sees; there is no second copy to drift.
mkdir -p "$SKILLS_DIR"
found=0
for vendored in "$ROOT_DIR"/skills/*/; do
  [[ -f "$vendored/SKILL.md" ]] || continue
  found=$((found + 1))
  name="$(basename "$vendored")"
  target="$SKILLS_DIR/$name"
  declared="$(sed -n 's/^name: *//p' "$vendored/SKILL.md" | head -1)"
  if [[ -n "$declared" && "$declared" != "$name" ]]; then
    bad "$name: SKILL.md declares name '$declared' -- rename one so they agree"
  fi
  if [[ -L "$target" || ! -e "$target" ]]; then
    ln -sfn "${vendored%/}" "$target" && ok "$name linked"
  else
    bad "$target exists and is not a symlink -- move it aside and re-run"
  fi
done
[[ $found -gt 0 ]] || bad "no skills found under $ROOT_DIR/skills/"

step "verify"
claude plugin list 2>/dev/null | grep -i humanize \
  && ok "humanize visible" \
  || bad "humanize not visible to the CLI"
for target in "$SKILLS_DIR"/*/; do
  name="$(basename "$target")"
  if [[ -f "$target/SKILL.md" ]]; then
    ok "$name: SKILL.md found"
  else
    bad "$name: no SKILL.md at $target -- the skill will not load"
  fi
done

printf '\n=== %s ===\n' "$([[ $status -eq 0 ]] && echo READY || echo NOT READY)"
[[ $status -eq 0 ]] && cat <<'EOF'

Restart the Claude Code session so the plugin and skills load. Then run
odyssey-create-workspace to start a task, or, from an existing workspace
directory, paste prompts/phase1.md into a fresh session.
EOF
exit $status
