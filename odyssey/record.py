"""The three records that survive the hackathon.

`benchmark.csv`     one row per measurement. Append-only, never edited.
`solutions.jsonl`   one line per candidate evaluated, with a parent pointer.
                    The parent links form the search DAG.
`runs/profile/`     Nsight Compute reports, one directory per direction.

These are not bookkeeping. They are the evidence for four separate claims the
work has to make later: that the speedup is reproducible (median and p90 across
runs, not a best-of), that the search was a search and not a story told
afterwards (the DAG, including the branches that died), that accuracy was
budgeted rather than hoped for (per-shape error against tolerance), and that a
rejected direction was rejected for a reason (the note on a dead node).

A dead end recorded on the day it died is evidence. Reconstructed on the last
night, it is a guess.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .paths import BENCHMARK_CSV, SOLUTIONS_JSONL, ensure_runs_dir

BENCHMARK_FIELDS = [
    "timestamp",
    "run_id",
    "git_sha",
    "git_dirty",
    "candidate",
    "case",
    "case_key",
    "dispatch_key",
    "device",
    "gpu",
    "torch",
    "script_md5",
    "dtype",
    "batch_size",
    "seq_len",
    "d_model",
    "num_heads",
    "ffn_dim",
    "num_layers",
    "causal",
    "padding_ratio",
    "atol",
    "rtol",
    "accuracy_passed",
    "failed_elements",
    "total_elements",
    "max_abs_error",
    "max_relative_error",
    "abs_budget_spent",
    "rel_budget_spent",
    "baseline_median_ms",
    "baseline_mean_ms",
    "baseline_p90_ms",
    "baseline_min_ms",
    "optimized_median_ms",
    "optimized_mean_ms",
    "optimized_p90_ms",
    "optimized_min_ms",
    "speedup",
    "tokens_per_second",
    "compile_baseline",
    "compile_candidate",
    "compile_mode",
    "allow_tf32",
    "matmul_precision",
    "skipped",
    "error",
    "notes",
]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason to lose a run
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def new_run_id() -> str:
    return os.environ.get("ODYSSEY_RUN_ID") or uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def result_to_row(result: Any, *, run_id: str, sha: str, dirty: bool) -> dict[str, Any]:
    case = result.case
    accuracy = result.accuracy
    headroom = accuracy.headroom() if accuracy else {}
    return {
        "timestamp": _now(),
        "run_id": run_id,
        "git_sha": sha,
        "git_dirty": dirty,
        "candidate": result.candidate,
        "case": case.name,
        "case_key": case.key,
        "dispatch_key": case.dispatch_key,
        "device": result.device,
        "gpu": result.gpu_name,
        "torch": result.torch_version,
        "script_md5": result.script_md5,
        "dtype": case.dtype,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "d_model": case.d_model,
        "num_heads": case.num_heads,
        "ffn_dim": case.ffn_dim,
        "num_layers": case.num_layers,
        "causal": case.causal,
        "padding_ratio": case.padding_ratio,
        "atol": accuracy.atol if accuracy else "",
        "rtol": accuracy.rtol if accuracy else "",
        "accuracy_passed": accuracy.passed if accuracy else "",
        "failed_elements": accuracy.failed_elements if accuracy else "",
        "total_elements": accuracy.total_elements if accuracy else "",
        "max_abs_error": accuracy.max_abs_error if accuracy else "",
        "max_relative_error": accuracy.max_relative_error if accuracy else "",
        "abs_budget_spent": headroom.get("abs_spent", ""),
        "rel_budget_spent": headroom.get("rel_spent", ""),
        "baseline_median_ms": result.baseline.median_ms if result.baseline.samples_ms else "",
        "baseline_mean_ms": result.baseline.mean_ms if result.baseline.samples_ms else "",
        "baseline_p90_ms": result.baseline.p90_ms if result.baseline.samples_ms else "",
        "baseline_min_ms": result.baseline.min_ms if result.baseline.samples_ms else "",
        "optimized_median_ms": result.optimized.median_ms if result.optimized.samples_ms else "",
        "optimized_mean_ms": result.optimized.mean_ms if result.optimized.samples_ms else "",
        "optimized_p90_ms": result.optimized.p90_ms if result.optimized.samples_ms else "",
        "optimized_min_ms": result.optimized.min_ms if result.optimized.samples_ms else "",
        "speedup": result.speedup if result.optimized.samples_ms else "",
        "tokens_per_second": result.tokens_per_second if result.optimized.samples_ms else "",
        "compile_baseline": result.compile_baseline,
        "compile_candidate": result.compile_candidate,
        "compile_mode": result.compile_mode,
        "allow_tf32": result.allow_tf32,
        "matmul_precision": result.matmul_precision,
        "skipped": result.skipped or "",
        "error": result.error or "",
        "notes": result.notes,
    }


class BenchmarkLog:
    """Append-only CSV. The header is written once and never reordered."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or BENCHMARK_CSV)
        self.run_id = new_run_id()
        self._sha = git_sha()
        self._dirty = git_dirty()

    def append(self, results: Iterable[Any]) -> int:
        rows = [
            result_to_row(r, run_id=self.run_id, sha=self._sha, dirty=self._dirty)
            for r in results
        ]
        if not rows:
            return 0

        ensure_runs_dir()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return len(rows)

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as handle:
            return list(csv.DictReader(handle))


@dataclass
class SolutionNode:
    node_id: str
    parent: Optional[str]
    candidate: str
    case: str
    status: str
    speedup: Optional[float]
    max_abs_error: Optional[float]
    decision: str
    evidence: str
    notes: str
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class SolutionDAG:
    """One JSON object per line, each with a `parent` pointing at what it came from.

    A node is written whether it won or lost. `decision` says what happened to
    it -- `keep`, `reject`, `park` -- and `evidence` says where to look: an NCU
    report path, a commit, a benchmark run id. Rejected nodes are the point;
    they are what makes "we explored this and it did not work" checkable.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or SOLUTIONS_JSONL)

    def append(
        self,
        *,
        candidate: str,
        case: str,
        status: str,
        parent: Optional[str] = None,
        node_id: Optional[str] = None,
        speedup: Optional[float] = None,
        max_abs_error: Optional[float] = None,
        decision: str = "",
        evidence: str = "",
        notes: str = "",
    ) -> SolutionNode:
        node = SolutionNode(
            node_id=node_id or f"{candidate}:{case}:{uuid.uuid4().hex[:8]}",
            parent=parent,
            candidate=candidate,
            case=case,
            status=status,
            speedup=speedup,
            max_abs_error=max_abs_error,
            decision=decision,
            evidence=evidence,
            notes=notes,
            timestamp=_now(),
        )
        ensure_runs_dir()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(node.to_json() + "\n")
        return node

    def append_result(
        self, result: Any, *, parent: Optional[str] = None, decision: str = "", **kw: Any
    ) -> SolutionNode:
        status = (
            "skipped" if result.skipped
            else "error" if result.error
            else "pass" if result.passed
            else "fail"
        )
        return self.append(
            candidate=result.candidate,
            case=result.case.name,
            status=status,
            parent=parent,
            speedup=result.speedup if result.optimized.samples_ms else None,
            max_abs_error=(
                result.accuracy.max_abs_error if result.accuracy else None
            ),
            decision=decision,
            notes=result.notes or result.skipped or result.error or "",
            **kw,
        )

    def nodes(self) -> Iterator[SolutionNode]:
        if not self.path.exists():
            return iter(())
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(SolutionNode(**json.loads(line)))
        return iter(out)

    def to_dot(self) -> str:
        """Render the search DAG for the report. `dot -Tsvg` from here."""
        colors = {
            "pass": "#2f6f4f",
            "fail": "#8c3b3b",
            "skipped": "#7a7a7a",
            "error": "#8c3b3b",
        }
        lines = ["digraph search {", "  rankdir=LR;", '  node [shape=box, style=rounded];']
        for node in self.nodes():
            label = f"{node.candidate}\\n{node.case}"
            if node.speedup:
                label += f"\\n{node.speedup:.2f}x"
            color = colors.get(node.status, "#444444")
            lines.append(f'  "{node.node_id}" [label="{label}", color="{color}"];')
            if node.parent:
                lines.append(f'  "{node.parent}" -> "{node.node_id}";')
        lines.append("}")
        return "\n".join(lines)
