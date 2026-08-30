"""The shipping layer: never slower than what it replaces.

Every other candidate is a bet. This one is the policy that decides when to
take the bet. For a given geometry it uses the specialization a calibration run
proved both correct and faster; for everything else it runs the baseline path.
The floor is therefore the baseline, not the best case -- an unfamiliar shape
degrades to "no worse", never to "wrong" and never to "slower".

Two rules make that claim true rather than aspirational:

**Correctness is checked before speed.** `calibrate` only admits a candidate
that passed the zero-bad-element check on every case in the group.

**Padding is not a dispatch axis.** The official generator always supplies a
mask -- all-ones when `--padding-ratio 0` -- so a running model cannot tell
dense from padded without a device sync, which would also break graph capture.
Rather than guess, a candidate must clear both the dense and the padded case
for a geometry before it is allowed to serve either.

Without a table this class is exactly the baseline, which is the correct
behaviour on day zero and the reason it is safe to leave wired in.

## Calibration

    python verify.py fused-safe fused-sdpa graph-safe graph-sdpa --shapes official --record
    python kernels/dispatch.py calibrate            # reads runs/benchmark.csv, writes runs/dispatch_table.json

The rule is deliberately conservative, because the thing it protects is the
claim that the shipped model is never slower and never wrong:

1. Group recorded rows by geometry -- everything a running model can identify
   from its own inputs: batch, sequence, width, heads, ffn, layers, causal,
   dtype. Padded and dense variants of one geometry land in the same group.
2. A candidate is eligible for a group only if it passed on **every** case
   in that group, and only rows measured against the eager baseline count
   (a row taken with `--compile-baseline` compares against a different
   denominator).
3. Its score is its **worst** speedup across the group, not its average.
4. It is admitted only if that worst speedup clears `1 + margin`. The margin
   exists so run-to-run noise cannot promote a tie.

Anything unadmitted falls through to the baseline. A geometry with no winner is
a normal outcome, not a failure -- it is the fallback doing its job.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

WORKSPACE = Path(__file__).resolve().parent.parent
BENCHMARK_CSV = WORKSPACE / "runs" / "benchmark.csv"
DISPATCH_TABLE = WORKSPACE / "runs" / "dispatch_table.json"

DEFAULT_MARGIN = 0.05

# Never dispatch to these: `passthrough` is a control and `dispatch` is us.
NOT_DISPATCHABLE = frozenset({"passthrough", "dispatch"})


# -- the key ------------------------------------------------------------------


def _key(batch: Any, seq_len: Any, d_model: Any, heads: Any, ffn: Any, layers: Any,
         causal: bool, dtype: str) -> str:
    return (
        f"b{batch}_s{seq_len}_d{d_model}_h{heads}_f{ffn}_l{layers}"
        f"_{'causal' if causal else 'full'}_{dtype}"
    )


def runtime_dispatch_key(config: Any, x: torch.Tensor) -> str:
    batch, seq_len, _ = x.shape
    dtype = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(x.dtype, str(x.dtype))
    return _key(batch, seq_len, config.d_model, config.num_heads, config.ffn_dim,
                config.num_layers, config.causal, dtype)


def row_dispatch_key(row: dict[str, str]) -> str:
    return _key(row["batch_size"], row["seq_len"], row["d_model"], row["num_heads"],
                row["ffn_dim"], row["num_layers"], row["causal"] == "True", row["dtype"])


# -- the candidate ------------------------------------------------------------


def load_table(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    path = Path(path or DISPATCH_TABLE)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return raw.get("entries", raw)


@functools.lru_cache(maxsize=None)
def dispatch_class(official: Any) -> type:
    class DispatchTransformer(official.BaselineTransformer):
        def __init__(self, config: Any, table: Optional[dict] = None) -> None:
            super().__init__(config)
            # Plain dict, so delegates never enter this module's state_dict and
            # the strict weight copy keeps working.
            self._delegates: dict[str, Any] = {}
            self._table = load_table() if table is None else table
            self._resolved: dict[str, Optional[str]] = {}

        def _delegate(self, name: str) -> Any:
            cached = self._delegates.get(name)
            if cached is not None:
                return cached

            from . import CANDIDATES  # noqa: PLC0415 - the package imports this module

            module = CANDIDATES[name][1](official)(self.config)
            # Share this model's parameters by reference. The delegate's own
            # freshly-allocated layers are discarded here; that costs one CPU
            # allocation the first time a geometry is served, and nothing after.
            module.layers = self.layers
            module.final_norm = self.final_norm
            module.eval()
            self._delegates[name] = module
            return module

        def _choose(self, key: str) -> Optional[str]:
            if key in self._resolved:
                return self._resolved[key]
            entry = self._table.get(key)
            name = entry.get("candidate") if isinstance(entry, dict) else entry
            if name in NOT_DISPATCHABLE:
                name = None
            self._resolved[key] = name
            return name

        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            name = self._choose(runtime_dispatch_key(self.config, x))
            if name is None:
                return super().forward(x, valid_token_mask)
            return self._delegate(name)(x, valid_token_mask)

    return DispatchTransformer


CANDIDATES = {
    "dispatch": (
        "Per-geometry dispatch to a calibrated candidate; baseline fallback otherwise.",
        dispatch_class,
    ),
}


# -- calibration ----------------------------------------------------------------


def calibrate(csv_path: Path = BENCHMARK_CSV, *, margin: float = DEFAULT_MARGIN) -> dict[str, Any]:
    """Build the table from `runs/benchmark.csv`. Latest row per (geometry, candidate, case) wins."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist; run verify.py --record first")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    latest: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    cases_in: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["candidate"] in NOT_DISPATCHABLE or "--compile-baseline" in row.get("script_args", ""):
            continue
        key = row_dispatch_key(row)
        latest[key][row["candidate"]][row["case"]] = row
        cases_in[key].add(row["case"])

    entries: dict[str, Any] = {}
    no_winner: dict[str, Any] = {}
    for key in sorted(latest):
        expected = cases_in[key]
        winner: Optional[str] = None
        best: Optional[float] = None
        rejected: dict[str, str] = {}
        for name, by_case in sorted(latest[key].items()):
            missing = expected - set(by_case)
            if missing:
                rejected[name] = f"did not run on {sorted(missing)}"
                continue
            failures = sorted(c for c, r in by_case.items() if r["passed"] != "True")
            if failures:
                rejected[name] = f"failed correctness on {failures}"
                continue
            try:
                worst = min(float(r["speedup"]) for r in by_case.values())
            except ValueError:
                rejected[name] = "no speedup recorded"
                continue
            if worst < 1.0 + margin:
                rejected[name] = f"worst speedup {worst:.3f}x below 1+{margin:g}"
                continue
            if best is None or worst > best:
                winner, best = name, worst
        if winner is None:
            no_winner[key] = rejected
        else:
            entries[key] = {
                "candidate": winner,
                "worst_speedup": best,
                "cases": sorted(expected),
                "margin": margin,
                "rejected": rejected,
            }

    return {
        "note": (
            "Generated by `python kernels/dispatch.py calibrate`. A geometry appears in "
            "`entries` only if one candidate passed every case in its group and beat the "
            "eager baseline on the worst of them by the margin. Absent geometries fall back "
            "to the baseline path."
        ),
        "source": str(csv_path),
        "margin": margin,
        "entries": entries,
        "no_winner": no_winner,
    }


def to_markdown(table: dict[str, Any]) -> str:
    lines = [
        "| Geometry | Serving | Worst-case speedup | Why not the others |",
        "|---|---|---:|---|",
    ]
    for key, entry in table["entries"].items():
        why = "; ".join(f"`{k}`: {v}" for k, v in entry["rejected"].items()) or "--"
        lines.append(f"| `{key}` | `{entry['candidate']}` | {entry['worst_speedup']:.3f}x | {why} |")
    for key, rejected in table["no_winner"].items():
        why = "; ".join(f"`{k}`: {v}" for k, v in rejected.items()) or "--"
        lines.append(f"| `{key}` | _baseline fallback_ | -- | {why} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="calibrate the dispatch table from runs/benchmark.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--csv", type=Path, default=BENCHMARK_CSV)
    cal.add_argument("--out", type=Path, default=DISPATCH_TABLE)
    cal.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    cal.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    table = calibrate(args.csv, margin=args.margin)
    print(to_markdown(table))
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
