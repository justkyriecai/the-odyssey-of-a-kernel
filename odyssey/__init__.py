"""Harness for the TikTok TechJam 2026 Track 3 kernel work.

The organizer's `torch_transformer_benchmark.py` is the scoreboard. This package
never replaces it: it imports the script's correctness rule, data generator and
timing primitives and wraps them in something that returns values instead of
printing them, so a sweep can be recorded, replayed and plotted.

Layout:
    odyssey.official   loading and running the organizer's script
    odyssey.shapes     the shape grid (cases), loaded from bench/shapes/*.json
    odyssey.registry   candidate implementations, by name
    odyssey.harness    evaluate(candidate, case) -> EvalResult
    odyssey.ladder     L0 eager / L1 compile / L2 max-autotune / L3 ours
    odyssey.record     runs/benchmark.csv and runs/solutions.jsonl
    odyssey.roofline   FLOP accounting and the roofline migration plot
    odyssey.ablation   the three-arm skill ablation
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "official",
    "shapes",
    "registry",
    "harness",
    "ladder",
    "record",
    "roofline",
    "ablation",
]
