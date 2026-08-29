"""Where things live. One place, so nothing hard-codes a relative path."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCH_DIR = ROOT / "bench"
SHAPES_DIR = BENCH_DIR / "shapes"
OFFICIAL_SCRIPT = BENCH_DIR / "official" / "torch_transformer_benchmark.py"

DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"

RUNS_DIR = Path(os.environ.get("ODYSSEY_RUNS", ROOT / "runs"))
BENCHMARK_CSV = RUNS_DIR / "benchmark.csv"
SOLUTIONS_JSONL = RUNS_DIR / "solutions.jsonl"
PROFILE_DIR = RUNS_DIR / "profile"
DISPATCH_TABLE = RUNS_DIR / "dispatch_table.json"


def ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR
