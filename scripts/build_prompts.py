#!/usr/bin/env python3
"""Expand `_shared.md` into every phase prompt, between the marker comments.

Prompts get pasted into agent sessions, so each file has to be self-contained.
Keeping one editable copy of the shared block and expanding it here is the only
way three self-contained files stay in agreement. Idempotent: run it after every
edit to `_shared.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

BEGIN = "<!-- BEGIN shared -->"
END = "<!-- END shared -->"


def _strip_editor_notes(text: str) -> str:
    """Drop leading HTML comments. They are notes to whoever edits `_shared.md`;
    the expanded prompts get pasted into agent sessions as raw text, where a note
    about editing the source file is only a distraction."""
    lines = text.strip().splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("<!--")):
        lines.pop(0)
    return "\n".join(lines).strip()


def expand(task_dir: Path) -> list[Path]:
    shared_path = task_dir / "_shared.md"
    if not shared_path.exists():
        return []
    shared = _strip_editor_notes(shared_path.read_text())
    touched = []

    for path in sorted(task_dir.glob("phase*.md")):
        text = path.read_text()
        if BEGIN not in text or END not in text:
            print(f"  {path.name}: no markers, skipped", file=sys.stderr)
            continue
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        rebuilt = f"{head}{BEGIN}\n\n{shared}\n\n{END}{tail}"
        if rebuilt != text:
            path.write_text(rebuilt)
            touched.append(path)
    return touched


def main() -> int:
    tasks = [d for d in sorted(PROMPTS.iterdir()) if d.is_dir()]
    if not tasks:
        print("no task directories under prompts/", file=sys.stderr)
        return 1
    for task in tasks:
        touched = expand(task)
        print(f"{task.name}: {len(touched)} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
