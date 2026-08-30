#!/usr/bin/env bash
# Full sweep, then recalibrate the dispatch table, then redraw the roofline.
# This is the loop that runs after every accepted optimization.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-.venv/bin/python}"
SHAPES="${SHAPES:-dev}"

"$PY" -m odyssey bench all --shapes "$SHAPES" --notes "${NOTES:-sweep}"
"$PY" -m odyssey calibrate --shapes "$SHAPES"
"$PY" -m odyssey roofline || echo "roofline skipped (matplotlib or peaks missing)"

echo
echo "Final gate -- the organizer's own script, unmodified:"
"$PY" -m odyssey official dispatch --shapes "$SHAPES" --case center -- --atol 0.001 --rtol 0.01
