#!/usr/bin/env bash
# Install the conditional Phase 3 pre-commit hook into .git/hooks/
#
# The installed hook:
#   - always runs scripts/validate.py
#   - runs scripts/repo_size_check.py only if data/phase3-size-budget.yaml exists
#   - runs scripts/check_dod_fixtures.py only if data/phase3-dod-fixtures.yaml exists

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

if [ ! -d "$REPO_ROOT/.git" ]; then
  echo "ERROR: $REPO_ROOT/.git not a directory (is this a git work tree?)" >&2
  exit 2
fi

cat > "$HOOK_PATH" <<'HOOK'
#!/usr/bin/env bash
# Auto-generated Phase 3 pre-commit hook
set -u
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "[pre-commit] validate.py"
python3 scripts/validate.py
status=$?
if [ $status -ne 0 ]; then
  echo "[pre-commit] validate.py failed"; exit $status
fi

if [ -f "data/phase3-size-budget.yaml" ]; then
  echo "[pre-commit] repo_size_check.py"
  python3 scripts/repo_size_check.py
  status=$?
  if [ $status -ne 0 ]; then
    echo "[pre-commit] repo_size_check.py failed"; exit $status
  fi
fi

if [ -f "data/phase3-dod-fixtures.yaml" ]; then
  echo "[pre-commit] check_dod_fixtures.py"
  python3 scripts/check_dod_fixtures.py
  status=$?
  if [ $status -ne 0 ]; then
    echo "[pre-commit] check_dod_fixtures.py failed"; exit $status
  fi
fi

exit 0
HOOK

chmod +x "$HOOK_PATH"
echo "Installed pre-commit hook at $HOOK_PATH"
