#!/usr/bin/env python3
"""Expand `_shared.md` into every phase prompt, between the marker comments.

Prompts get pasted into agent sessions, so each phase file has to be
self-contained. Keeping one editable copy of the shared block per prompt
directory and expanding it here is the only way three self-contained files
stay in agreement. Idempotent: run it after every edit to a `_shared.md`.

Covers `prompts/template/` and every `workspace/*/prompts/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def prompt_dirs() -> list[Path]:
    dirs = [ROOT / "prompts" / "template"]
    dirs += sorted((ROOT / "workspace").glob("*/prompts"))
    return [d for d in dirs if (d / "_shared.md").is_file()]


def expand(prompt_dir: Path) -> list[Path]:
    shared = _strip_editor_notes((prompt_dir / "_shared.md").read_text())
    touched = []
    for path in sorted(prompt_dir.glob("phase*.md")):
        text = path.read_text()
        if BEGIN not in text or END not in text:
            print(f"  {path.relative_to(ROOT)}: no markers, skipped", file=sys.stderr)
            continue
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        rebuilt = f"{head}{BEGIN}\n\n{shared}\n\n{END}{tail}"
        if rebuilt != text:
            path.write_text(rebuilt)
            touched.append(path)
    return touched


def main() -> int:
    dirs = prompt_dirs()
    if not dirs:
        print("no prompt directories with a _shared.md", file=sys.stderr)
        return 1
    for prompt_dir in dirs:
        touched = expand(prompt_dir)
        print(f"{prompt_dir.relative_to(ROOT)}: {len(touched)} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
