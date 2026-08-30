#!/usr/bin/env bash
# Install the agent workflow dependencies: the plan/execute/verify harness and
# the two research skills. Nothing here is specific to the transformer layer --
# this is the method's toolchain, and it is the same for any operator.
#
# Run it on the rented box, inside tmux, before pasting a phase prompt. The
# first ten minutes of a pod are the most expensive ten minutes of the project;
# a manual checklist is where they go.
#
# Idempotent: re-running updates rather than duplicating.
set -uo pipefail

SKILLS_DIR="${SKILLS_DIR:-$HOME/.claude/skills}"
SKILL_ORG="${SKILL_ORG:-mit-han-lab}"   # DongyunZou also hosts both mirrors
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

step "research skills"
mkdir -p "$SKILLS_DIR"
for skill in KernelWiki ncu-report-skill; do
  target="$SKILLS_DIR/$skill"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" pull --quiet --ff-only 2>/dev/null \
      && ok "$skill up to date" \
      || ok "$skill present (pull skipped)"
  elif [[ -e "$target" ]]; then
    bad "$target exists but is not a git clone -- move it aside and re-run"
  else
    if git clone --quiet --depth 1 \
        "https://github.com/$SKILL_ORG/$skill.git" "$target" 2>/dev/null; then
      ok "$skill cloned"
    else
      bad "$skill clone failed from $SKILL_ORG"
    fi
  fi
done

step "verify"
claude plugin list 2>/dev/null | grep -i humanize \
  && ok "humanize visible" \
  || bad "humanize not visible to the CLI"
for skill in KernelWiki ncu-report-skill; do
  if [[ -f "$SKILLS_DIR/$skill/SKILL.md" ]]; then
    ok "$skill: SKILL.md found"
  else
    bad "$skill: no SKILL.md at $SKILLS_DIR/$skill -- the skill will not load"
  fi
done

printf '\n=== %s ===\n' "$([[ $status -eq 0 ]] && echo READY || echo NOT READY)"
[[ $status -eq 0 ]] && cat <<'EOF'

Next, in order:
  ./scripts/check_gpu.sh                       driver, torch, and the ncu permission probe
  .venv/bin/python -m odyssey peak             measured ceilings for this card
  ./scripts/run_ladder.sh                      the opponent, before writing anything

Restart the Claude Code session so the plugin and skills load.
EOF
exit $status
