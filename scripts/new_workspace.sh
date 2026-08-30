#!/usr/bin/env bash
# Start a workspace for a new kernel task, or move an existing task to another
# card.
#
#   ./scripts/new_workspace.sh <name>
#   ./scripts/new_workspace.sh <name> --from <existing-workspace>
#
# Creates workspace/<name>/ with the directory skeleton, the three phase
# prompts copied from prompts/template/, a verify.py to adapt, and a README
# that lists what is still yours to do before an agent session can start.
# Nothing here is specific to any operator; everything specific goes into the
# files this creates.
#
# `--from` is the same task on a different GPU. The task-invariant work carries
# over -- the evaluator, the shape sets, verify.py, the candidates, the docs --
# and the card-specific work does not: `runs/` starts empty, because a number
# measured on one card is not evidence about another, and the hardware section
# of prompts/_shared.md has to be rewritten before the prompts are true again.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: new_workspace.sh <name> [--from <existing-workspace>]}"
shift
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="${2:?--from needs a workspace name}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
WS="$ROOT_DIR/workspace/$NAME"

if [[ -e "$WS" ]]; then
  echo "workspace/$NAME already exists" >&2
  exit 1
fi
if [[ ! "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "use lowercase letters, digits and hyphens for the name" >&2
  exit 1
fi
SRC=""
if [[ -n "$FROM" ]]; then
  SRC="$ROOT_DIR/workspace/${FROM#workspace/}"
  if [[ ! -d "$SRC" ]]; then
    echo "--from: workspace/${FROM#workspace/} does not exist" >&2
    exit 1
  fi
fi

mkdir -p "$WS"/{bench/official,bench/shapes,kernels,prompts,docs/assets,runs,scripts,infra}
touch "$WS"/bench/official/.gitkeep "$WS"/bench/shapes/.gitkeep "$WS"/docs/assets/.gitkeep \
      "$WS"/runs/.gitkeep "$WS"/scripts/.gitkeep "$WS"/infra/.gitkeep

cp "$ROOT_DIR"/prompts/template/_shared.md "$ROOT_DIR"/prompts/template/phase{1,2,3}.md "$WS/prompts/"

# The first workspace's verify.py is the worked example of the pattern: load
# the evaluator by path, patch one class, build argv from a shape-set case,
# read the verdict off what the evaluator printed. Copy it as a starting point.
EXAMPLE="$ROOT_DIR/workspace/transformer-forward-opt/verify.py"
if [[ -f "$EXAMPLE" ]]; then
  cp "$EXAMPLE" "$WS/verify.py"
fi

cat > "$WS/kernels/__init__.py" <<'PY'
"""Candidate implementations.

Each module exposes `CANDIDATES`: name -> (description, factory), where the
factory takes the loaded evaluator module and returns the class to substitute.
`verify.py` collects them from here. Start with a `passthrough` control that
runs the reference unchanged: it must read ~1.00x and zero error, or the
measurement is wrong and nothing else can be trusted.
"""

from __future__ import annotations

CANDIDATES: dict = {}
PY

# Every task carries its own CLAUDE.md holding that task's hard facts. It is a
# skeleton of slots here; `odyssey-create-workspace` fills them from the
# interview, and whoever vendors the evaluator fills the rest.
cat > "$WS/CLAUDE.md" <<EOF
# $NAME

- Task: <<one line: the contest or the internal request, and the operator>>.
- Evaluator: \`bench/official/<<script>>\`, provided by <<whom>>, unmodified,
  \`md5 <<...>>\`.
- What the candidate replaces: <<the class or function the evaluator calls>>,
  signature fixed.
- Hardware: <<card>>, <<machine>>. The prompts here are written for this card
  only.
- Target: <<speedup>> against <<baseline>>, set by <<who>> on <<date>>.

The repository rules in the root \`CLAUDE.md\` apply here unchanged. In
particular: the evaluator is never edited, its correctness rule and its timing
protocol are never re-implemented, and every measurement lands in
\`runs/benchmark.csv\`.

## Still open

Every \`<<slot>>\` above, and the checklist in \`README.md\`.
EOF

cat > "$WS/README.md" <<EOF
# $NAME

Started from the odyssey template on $(date -u +%Y-%m-%d). Nothing here has
been measured yet. Before pasting \`prompts/phase1.md\` into an agent session:

- [ ] \`CLAUDE.md\`: fill every \`<<slot>>\` -- the task, the evaluator, the card,
      the target and who set it.
- [ ] \`bench/official/\`: vendor the evaluator **unmodified**; record its md5 in
      \`bench/official/README.md\` and read it end to end -- what it measures,
      what it masks, where the tolerance actually binds.
- [ ] \`bench/shapes/\`: \`smoke.json\` (correctness, CPU, seconds), \`dev.json\`
      (the iteration set), \`official.json\` (what gets scored).
- [ ] \`verify.py\`: adapt the three evaluator-specific parts -- which class is
      patched, \`CASE_FLAGS\`, and the output patterns the verdict is read from.
- [ ] \`prompts/_shared.md\`: fill every \`<<...>>\` slot, especially the hardware
      section, then \`python scripts/build_prompts.py\` from the repository root.
- [ ] \`kernels/\`: a \`passthrough\` control first. \`./scripts/smoke.sh\` must
      read ~1.00x and zero error before anything else is written.
- [ ] \`docs/\`: what a reader needs to trust the numbers.

Paths in this directory's docs and prompts are relative to this directory.
EOF

cat > "$WS/scripts/smoke.sh" <<'EOF'
#!/usr/bin/env bash
# Correctness only, CPU only, seconds. A candidate that fails here is wrong,
# not slow. Adapt the flags after `--` to the evaluator's own CLI.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$WS/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
CANDIDATES=("$@")
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  CANDIDATES=(passthrough)
fi
exec "$PY" "$WS/verify.py" "${CANDIDATES[@]}" --shapes smoke --quiet -- --device cpu
EOF
chmod +x "$WS/scripts/smoke.sh"
rm -f "$WS/scripts/.gitkeep"

if [[ -n "$SRC" ]]; then
  # Same task, different card. Everything task-invariant is carried over and
  # overwrites what the template just wrote; `runs/` is deliberately not, and
  # neither is the sibling's own CLAUDE.md -- both describe a card this
  # workspace is not running on.
  for dir in bench kernels docs scripts prompts infra; do
    if [[ -d "$SRC/$dir" ]]; then
      mkdir -p "$WS/$dir"
      cp -R "$SRC/$dir/." "$WS/$dir/"
    fi
  done
  if [[ -f "$SRC/verify.py" ]]; then
    cp "$SRC/verify.py" "$WS/verify.py"
  fi
  rm -rf "$WS/runs"
  mkdir -p "$WS/runs"
  touch "$WS/runs/.gitkeep"

  cat > "$WS/CLAUDE.md" <<EOF
# $NAME

Continues \`workspace/$(basename "$SRC")\` on a different card. The task, the
evaluator, the shape sets, \`verify.py\` and the candidates were carried over on
$(date -u +%Y-%m-%d); the measurements were not.

- Task, evaluator and what the candidate replaces: copy these three lines from
  \`workspace/$(basename "$SRC")/CLAUDE.md\`. They belong to the task and did
  not change with the card. Re-check the evaluator's md5 while you are there.
- Hardware: <<card>>, <<machine>>. **Rewritten for this card.** Anything the
  sibling's prompts claim about the hardware is about the other card until this
  is filled in.
- Target: <<speedup>> against <<baseline>>, set by <<who>> on <<date>>. The
  sibling's target does not carry over -- a different card is a different
  denominator.

The repository rules in the root \`CLAUDE.md\` apply here unchanged. In
particular: the evaluator is never edited, its correctness rule and its timing
protocol are never re-implemented, and every measurement lands in
\`runs/benchmark.csv\`.

## Carried over, and what that means

A candidate in \`kernels/\` arrived here because it was good on the other card.
That is a hypothesis about this one, not a result. Re-measure before ranking,
and re-run the correctness gate: a tile size or a launch bound tuned elsewhere
can be wrong here, and occupancy assumptions rarely survive a change of card.

## Still open

Every \`<<slot>>\` above, and the checklist in \`README.md\`.
EOF

  cat > "$WS/README.md" <<EOF
# $NAME

Branched from \`workspace/$(basename "$SRC")\` on $(date -u +%Y-%m-%d) for a
different card. Carried over: \`bench/\`, \`kernels/\`, \`verify.py\`, \`docs/\`,
\`scripts/\`, \`infra/\` and the prompts. Not carried over: \`runs/\`, which starts
empty on purpose.

Before pasting \`prompts/phase1.md\` into an agent session:

- [ ] \`CLAUDE.md\`: the card, the machine, and this round's target.
- [ ] \`prompts/_shared.md\`: rewrite the **hardware** section for this card --
      what is there, what is not, what is out of scope by decision rather than
      capability -- then \`python scripts/build_prompts.py\` from the repository
      root. This is the slot that is most expensive to get wrong, and it is
      currently describing another card.
- [ ] \`infra/\`: how this machine is obtained and bootstrapped.
- [ ] \`./scripts/smoke.sh\`: the \`passthrough\` control must still read ~1.00x
      with zero error here before any carried-over number is quoted.
- [ ] Re-measure the strongest available baseline on this card, and re-measure
      every candidate you intend to build on. Rank by what this card says.
- [ ] \`bench/official/\`: confirm the evaluator's md5 is unchanged from the
      sibling. If the organizer shipped a new one, it is a new evaluator.

Paths in this directory's docs and prompts are relative to this directory.
EOF
fi

(cd "$ROOT_DIR" && python3 scripts/build_prompts.py >/dev/null)

echo "workspace/$NAME"
find "$WS" -type f | sed "s#$ROOT_DIR/##" | sort
echo
if [[ -n "$SRC" ]]; then
  echo "Branched from workspace/$(basename "$SRC"). runs/ is empty by design; re-measure before trusting a carried-over candidate."
fi
echo "Next: work through workspace/$NAME/README.md, then paste prompts/phase1.md into a fresh session."
