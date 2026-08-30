#!/usr/bin/env python3
"""Run the organizer's benchmark script, unmodified, with a candidate patched in.

    python verify.py fused-safe --shapes dev
    python verify.py fused-safe --shapes official --case center -- --atol 0.002 --rtol 0.02
    python verify.py fused-safe fused-sdpa --shapes dev -- --compile-baseline --compile-mode max-autotune
    python verify.py --list

Everything after `--` is handed to the script verbatim. The script's own
`main()` runs, its own output prints, and its own exit code decides. This file
computes no number of its own: it builds the script's argv from a shape-set
case, swaps `UserOptimizedTransformer` for the candidate's class, and reads the
verdict back off what the script printed. `--record` appends one row per case
to `runs/benchmark.csv`, so every number in the record is the script's number.

The script is vendored at `bench/official/`; `$TECHJAM_BENCHMARK` points a
checkout at a different copy. Its md5 goes into every recorded row.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
SCRIPT = Path(
    os.environ.get("TECHJAM_BENCHMARK", HERE / "bench" / "official" / "torch_transformer_benchmark.py")
).expanduser().resolve()
SHAPES_DIR = HERE / "bench" / "shapes"
BENCHMARK_CSV = HERE / "runs" / "benchmark.csv"

# A case is a dict of the script's own flags. Keys are shape-set fields.
CASE_FLAGS = {
    "batch_size": "--batch-size",
    "seq_len": "--seq-len",
    "d_model": "--d-model",
    "num_heads": "--heads",
    "ffn_dim": "--ffn-dim",
    "num_layers": "--layers",
    "dtype": "--dtype",
    "padding_ratio": "--padding-ratio",
    "input_scale": "--input-scale",
}
CASE_FIELDS = (*CASE_FLAGS, "causal")

RECORD_FIELDS = [
    "timestamp", "git_sha", "git_dirty", "script_md5", "device", "gpu", "torch", "cuda", "driver",
    "candidate", "case", *CASE_FIELDS, "script_args", "atol", "rtol",
    "passed", "max_abs", "max_rel",
    "baseline_median_ms", "baseline_p90_ms", "optimized_median_ms", "optimized_p90_ms",
    "speedup", "exit_code", "notes",
]

# What the script prints, and where the numbers are in it.
_DEVICE = re.compile(r"^device=(\S+), dtype=\S+, torch=(\S+)", re.M)
_CRITERION = re.compile(r"^criterion: abs_error <= (\S+) OR relative_error <= ([\d.]+)%", re.M)
_SUMMARY = re.compile(r"^summary: (PASS|FAIL) \| max_abs=(\S+) \| max_rel=(\S+)", re.M)
_BASELINE = re.compile(r"^baseline : median=([\d.]+) ms \| mean=[\d.]+ ms \| p90=([\d.]+) ms", re.M)
_OPTIMIZED = re.compile(r"^optimized: median=([\d.]+) ms \| mean=[\d.]+ ms \| p90=([\d.]+) ms", re.M)
_SPEEDUP = re.compile(r"^speedup  : ([\d.]+)x", re.M)


# -- the script -----------------------------------------------------------------


def script_md5() -> str:
    return hashlib.md5(SCRIPT.read_bytes()).hexdigest()


def load_script() -> Any:
    if not SCRIPT.exists():
        sys.exit(f"benchmark script not found at {SCRIPT}; vendor it or set TECHJAM_BENCHMARK")
    spec = importlib.util.spec_from_file_location("techjam_official_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidates() -> dict[str, tuple[str, Any]]:
    """`kernels.CANDIDATES`: name -> (description, factory(script_module) -> class)."""
    sys.path.insert(0, str(HERE))
    import kernels  # noqa: PLC0415 - the workspace is the package root

    return kernels.CANDIDATES


# -- shapes ---------------------------------------------------------------------


def load_shapes(name_or_path: str) -> list[dict[str, Any]]:
    path = Path(name_or_path)
    if not path.exists():
        path = SHAPES_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SHAPES_DIR.glob("*.json")))
        sys.exit(f"no shape set {name_or_path!r}; available: {available}")
    raw = json.loads(path.read_text())
    defaults = raw.get("defaults", {})
    return [{**defaults, **case} for case in raw["cases"]]


def case_argv(case: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, flag in CASE_FLAGS.items():
        if key in case:
            argv += [flag, str(case[key])]
    if case.get("causal"):
        argv.append("--causal")
    return argv


# -- one run --------------------------------------------------------------------


class _Tee(io.TextIOBase):
    def __init__(self, *streams: Any) -> None:
        self._streams = [s for s in streams if s is not None]

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def run_case(module: Any, cls: Any, argv: Sequence[str], *, quiet: bool) -> tuple[int, str]:
    """The script's own `main()`, with the class swapped in and its output captured."""
    original = module.UserOptimizedTransformer
    saved_argv = sys.argv
    captured = io.StringIO()
    tee = _Tee(captured, None if quiet else sys.__stdout__)
    try:
        module.UserOptimizedTransformer = cls
        sys.argv = [str(SCRIPT), *argv]
        with contextlib.redirect_stdout(tee):
            try:
                code = int(module.main())
            except SystemExit as exc:  # argparse errors and the like
                code = int(exc.code or 0) if isinstance(exc.code, int) or exc.code is None else 1
            except Exception as exc:  # noqa: BLE001 - a crashing candidate is a result
                print(f"ERROR {type(exc).__name__}: {exc}")
                code = 1
    finally:
        module.UserOptimizedTransformer = original
        sys.argv = saved_argv
    return code, captured.getvalue()


def parse(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "device": "", "torch": "", "atol": "", "rtol": "",
        "passed": "", "max_abs": "", "max_rel": "",
        "baseline_median_ms": "", "baseline_p90_ms": "",
        "optimized_median_ms": "", "optimized_p90_ms": "", "speedup": "",
    }
    if m := _DEVICE.search(text):
        out["device"], out["torch"] = m.group(1), m.group(2)
    if m := _CRITERION.search(text):
        # The script prints rtol as a percentage; record the fraction it was given.
        out["atol"], out["rtol"] = m.group(1), f"{float(m.group(2)) / 100:g}"
    if m := _SUMMARY.search(text):
        out["passed"], out["max_abs"], out["max_rel"] = m.group(1) == "PASS", m.group(2), m.group(3)
    if m := _BASELINE.search(text):
        out["baseline_median_ms"], out["baseline_p90_ms"] = m.group(1), m.group(2)
    if m := _OPTIMIZED.search(text):
        out["optimized_median_ms"], out["optimized_p90_ms"] = m.group(1), m.group(2)
    if m := _SPEEDUP.search(text):
        out["speedup"] = m.group(1)
    return out


# -- the record -----------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=HERE
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason to lose a run
        return ""


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:  # noqa: BLE001
        return ""


def cuda_runtime() -> str:
    """The CUDA runtime the torch wheel bundles (`torch.version.cuda`); empty on CPU builds."""
    try:
        import torch

        return (torch.version.cuda or "") if torch.cuda.is_available() else ""
    except Exception:  # noqa: BLE001
        return ""


def driver_version() -> str:
    """The host NVIDIA driver, from nvidia-smi. The one version a rented box does
    not let you choose, and the first thing to quote when a number fails to
    reproduce on a different host."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return out.splitlines()[0].strip() if out else ""
    except Exception:  # noqa: BLE001 - no driver is a valid state on a laptop
        return ""


def record(rows: list[dict[str, Any]], path: Path = BENCHMARK_CSV) -> int:
    """Append-only. The header is written once; a different header is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    if not fresh:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != RECORD_FIELDS:
            sys.exit(f"{path} has a different header; move it aside rather than mixing layouts")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# -- main -----------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(
        description="the organizer's script, unmodified, with a candidate patched in",
        epilog="flags after `--` go to the script verbatim (e.g. -- --atol 0.002 --rtol 0.02)",
    )
    parser.add_argument("candidates", nargs="*", help="candidate names from kernels/")
    parser.add_argument("--shapes", default="dev", help="shape set name or path (default: dev)")
    parser.add_argument("--case", action="append", dest="cases", help="restrict to named case(s)")
    parser.add_argument("--record", action="store_true", help="append rows to runs/benchmark.csv")
    parser.add_argument("--notes", default="", help="free text for the recorded rows")
    parser.add_argument("--quiet", action="store_true", help="only the summary table")
    parser.add_argument("--list", action="store_true", help="candidates and shape sets")
    args = parser.parse_args(argv)

    candidates = load_candidates()
    if args.list:
        print("candidates:")
        for name, (description, _) in candidates.items():
            print(f"  {name:<14} {description}")
        print("\nshape sets:")
        for path in sorted(SHAPES_DIR.glob("*.json")):
            names = ", ".join(c.get("name", "?") for c in load_shapes(path.stem))
            print(f"  {path.stem:<14} {names}")
        print(f"\nscript  {SCRIPT}\nmd5     {script_md5()}")
        return 0
    if not args.candidates:
        parser.error("name at least one candidate, or pass --list")
    unknown = [c for c in args.candidates if c not in candidates]
    if unknown:
        parser.error(f"unknown candidate(s) {unknown}; known: {sorted(candidates)}")

    cases = load_shapes(args.shapes)
    if args.cases:
        by_name = {c["name"]: c for c in cases}
        missing = [n for n in args.cases if n not in by_name]
        if missing:
            parser.error(f"unknown case(s) {missing} in {args.shapes}; known: {sorted(by_name)}")
        cases = [by_name[n] for n in args.cases]

    module = load_script()
    md5 = script_md5()
    sha, dirty = _git("rev-parse", "--short", "HEAD"), bool(_git("status", "--porcelain"))
    gpu, cuda, driver = gpu_name(), cuda_runtime(), driver_version()

    rows: list[dict[str, Any]] = []
    for name in args.candidates:
        cls = candidates[name][1](module)
        for case in cases:
            script_args = case_argv(case) + extra
            if not args.quiet:
                print(f"\n### {name} / {case['name']}\n### {SCRIPT.name} {' '.join(script_args)}\n")
            code, text = run_case(module, cls, script_args, quiet=args.quiet)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_sha": sha, "git_dirty": dirty, "script_md5": md5,
                "gpu": gpu, "cuda": cuda, "driver": driver,
                "candidate": name, "case": case["name"],
                **{field: case.get(field, "") for field in CASE_FIELDS},
                "script_args": " ".join(extra),
                **parse(text),
                "exit_code": code, "notes": args.notes,
            }
            rows.append(row)

    print(f"\n{'candidate':<18} {'case':<14} {'verdict':<8} {'speedup':>8} {'max_abs':>10} {'base ms':>9} {'opt ms':>9}  exit")
    for row in rows:
        verdict = "PASS" if row["passed"] is True else "FAIL" if row["passed"] is False else "ERROR"
        speed = f"{row['speedup']}x" if row["speedup"] else "--"
        print(
            f"{row['candidate']:<18} {row['case']:<14} {verdict:<8} {speed:>8} "
            f"{row['max_abs'] or '--':>10} {row['baseline_median_ms'] or '--':>9} "
            f"{row['optimized_median_ms'] or '--':>9}  {row['exit_code']}"
        )
    if args.record:
        print(f"\n{record(rows)} row(s) -> {BENCHMARK_CSV}")
    return 0 if all(r["exit_code"] == 0 for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
