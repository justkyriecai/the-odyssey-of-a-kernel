#!/usr/bin/env bash
# Start a workspace for a new kernel task.
#
#   ./scripts/new_workspace.sh <name>
#
# Creates workspace/<name>/ with the directory skeleton, the three phase
# prompts copied from prompts/template/, a verify.py to adapt, and a README
# that lists what is still yours to do before an agent session can start.
# Nothing here is specific to any operator; everything specific goes into the
# files this creates.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: new_workspace.sh <name>}"
WS="$ROOT_DIR/workspace/$NAME"

if [[ -e "$WS" ]]; then
  echo "workspace/$NAME already exists" >&2
  exit 1
fi
if [[ ! "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "use lowercase letters, digits and hyphens for the name" >&2
  exit 1
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

cat > "$WS/README.md" <<EOF
# $NAME

Started from the odyssey template on $(date -u +%Y-%m-%d). Nothing here has
been measured yet. Before pasting \`prompts/phase1.md\` into an agent session:

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

(cd "$ROOT_DIR" && python3 scripts/build_prompts.py >/dev/null)

echo "workspace/$NAME"
find "$WS" -type f | sed "s#$ROOT_DIR/##" | sort
echo
echo "Next: work through workspace/$NAME/README.md, then paste prompts/phase1.md into a fresh session."
