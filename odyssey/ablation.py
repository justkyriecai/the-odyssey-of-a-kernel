"""The three-arm skill ablation.

Claiming "an agent wrote these kernels" is a claim. Measuring *which part of the
agent loop did the work* is an experiment, and an experiment with a control is
the difference between a demo and a result. HAN Lab ran this after their MLSys
2026 submissions and found the harness -- the plan/execute/verify discipline --
dominated the knowledge base and the profiler skill. Reproducing that shape on a
different card and a different operator family is a finding either way: if it
replicates, the method transfers; if it does not, that is more interesting.

Three arms, each a separate agent session on the same shape set, same wall-clock
budget, same starting workspace:

    A  bare        generate, run the benchmark, keep what is faster.
    B  +profiler   arm A plus Nsight Compute evidence in the decision loop.
    C  +harness    arm B plus the full plan/execute/verify loop and the
                   five-iteration cap per direction.

The comparison is only worth anything if the budget is held equal. That is the
one thing this module cannot enforce -- write the wall-clock and the iteration
count into `runs/ablation/<arm>/meta.json` and let the chart show them, so a
reader can see what was held fixed rather than take it on faith.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .paths import RUNS_DIR

ABLATION_DIR = RUNS_DIR / "ablation"


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    description: str
    tools: tuple[str, ...]


ARMS: tuple[Arm, ...] = (
    Arm(
        "bare",
        "Generation loop only",
        "Write a candidate, run the official script, keep it if the median improves. "
        "No profiler, no plan, no iteration cap.",
        ("official benchmark",),
    ),
    Arm(
        "profiler",
        "+ profiling feedback",
        "Arm A plus Nsight Compute. The agent reads counters before choosing the next "
        "direction instead of inferring bottlenecks from wall-clock alone.",
        ("official benchmark", "ncu-report-skill"),
    ),
    Arm(
        "harness",
        "+ full harness",
        "Arm B plus plan/execute/verify: rank directions before implementing, cap each "
        "at five iterations, record the evidence for every kept or rejected branch.",
        ("official benchmark", "ncu-report-skill", "KernelWiki", "humanize"),
    ),
)


@dataclass
class ArmResult:
    arm: Arm
    best_speedup: Optional[float] = None
    best_candidate: str = ""
    rows: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_clock_minutes(self) -> Optional[float]:
        value = self.meta.get("wall_clock_minutes")
        return float(value) if value is not None else None


def arm_dir(arm: Arm, root: Optional[Path] = None) -> Path:
    return Path(root or ABLATION_DIR) / arm.key


def scaffold(root: Optional[Path] = None) -> list[Path]:
    """Create one directory per arm with a meta.json stub. Run this before D3."""
    created = []
    for arm in ARMS:
        directory = arm_dir(arm, root)
        directory.mkdir(parents=True, exist_ok=True)
        meta = directory / "meta.json"
        if not meta.exists():
            meta.write_text(
                json.dumps(
                    {
                        "arm": arm.key,
                        "label": arm.label,
                        "tools": list(arm.tools),
                        "shape_set": "dev",
                        "wall_clock_minutes": None,
                        "iterations": None,
                        "session_notes": "",
                        "held_fixed": [
                            "same shape set",
                            "same wall-clock budget",
                            "same starting workspace",
                            "same base model",
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            created.append(meta)
    return created


def collect(root: Optional[Path] = None) -> list[ArmResult]:
    """Read each arm's own `benchmark.csv` and take its best passing speedup."""
    results = []
    for arm in ARMS:
        directory = arm_dir(arm, root)
        result = ArmResult(arm=arm)

        meta_path = directory / "meta.json"
        if meta_path.exists():
            result.meta = json.loads(meta_path.read_text())

        csv_path = directory / "benchmark.csv"
        if csv_path.exists():
            with csv_path.open(newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if str(row.get("accuracy_passed", "")).lower() == "true"
                    and row.get("speedup")
                ]
            result.rows = len(rows)
            if rows:
                best = max(rows, key=lambda r: float(r["speedup"]))
                result.best_speedup = float(best["speedup"])
                result.best_candidate = best.get("candidate", "")
        results.append(result)
    return results


def to_markdown(results: Sequence[ArmResult]) -> str:
    lines = [
        "| Arm | Tools in the loop | Best passing speedup | Candidate | Runs | Wall clock |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in results:
        speed = f"{r.best_speedup:.3f}x" if r.best_speedup else "_not run_"
        wall = (
            f"{r.wall_clock_minutes:.0f} min" if r.wall_clock_minutes is not None else "--"
        )
        lines.append(
            f"| {r.arm.label} | {', '.join(r.arm.tools)} | {speed} | "
            f"`{r.best_candidate or '--'}` | {r.rows} | {wall} |"
        )
    return "\n".join(lines)


def plot(results: Sequence[ArmResult], out_path: Path, *, title: str = "Skill ablation") -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is needed for the ablation chart: pip install matplotlib"
        ) from exc

    labels = [r.arm.label for r in results]
    values = [r.best_speedup or 0.0 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=["#b8c4cc", "#7f9aa8", "#2f5d6e"], width=0.55)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="#8c3b3b")
    ax.text(
        len(labels) - 0.45, 1.02, "baseline", fontsize=8, color="#8c3b3b", ha="right"
    )

    for bar, result in zip(bars, results):
        height = bar.get_height()
        text = f"{height:.2f}x" if height else "not run"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Best passing speedup vs L0")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
