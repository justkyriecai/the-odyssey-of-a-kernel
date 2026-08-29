"""`python -m odyssey <command>`.

Thin: every command parses arguments, calls into a module, prints a table and
appends to `runs/`. Nothing here decides anything a reader of the code could not
find in `harness`, `ladder` or `calibrate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import ablation, calibrate, harness, ladder, official as official_mod, record, registry, roofline
from .paths import ASSETS_DIR, BENCHMARK_CSV, DISPATCH_TABLE, RUNS_DIR, SOLUTIONS_JSONL
from .shapes import Case, available_sets, load_set


# -- shared arguments ---------------------------------------------------------


def _add_measurement_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shapes", default="dev", help="shape set name or path (default: dev)")
    parser.add_argument("--case", action="append", dest="cases", help="restrict to named case(s)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--tolerance",
        choices=("strict", "loose"),
        default="strict",
        help="strict = 0.001/0.01 (docstring), loose = 0.002/0.02 (argparse). See "
        "bench/official/README.md; strict clears both.",
    )
    parser.add_argument("--atol", type=float, default=None, help="override --tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="override --tolerance")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--no-record", action="store_true", help="do not write to runs/")
    parser.add_argument("--notes", default="")


def _tolerance(args: argparse.Namespace) -> tuple[float, float]:
    atol = harness.STRICT_ATOL if args.tolerance == "strict" else harness.LOOSE_ATOL
    rtol = harness.STRICT_RTOL if args.tolerance == "strict" else harness.LOOSE_RTOL
    return (args.atol if args.atol is not None else atol,
            args.rtol if args.rtol is not None else rtol)


def _cases(args: argparse.Namespace) -> list[Case]:
    return list(load_set(args.shapes).select(args.cases))


def _eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    atol, rtol = _tolerance(args)
    return dict(
        device=args.device,
        atol=atol,
        rtol=rtol,
        trials=args.trials,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
        allow_tf32=not args.no_tf32,
        matmul_precision=args.matmul_precision,
        notes=args.notes,
    )


def _persist(results: Sequence[harness.EvalResult], args: argparse.Namespace) -> None:
    if args.no_record:
        return
    log = record.BenchmarkLog()
    written = log.append(results)
    dag = record.SolutionDAG()
    for result in results:
        dag.append_result(result)
    print(f"\n{written} row(s) -> {BENCHMARK_CSV}   (run_id {log.run_id})")
    print(f"{len(results)} node(s) -> {SOLUTIONS_JSONL}")


# -- commands -----------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    import torch

    print("=== odyssey doctor ===")
    print(f"python           {sys.version.split()[0]}")
    print(f"torch            {torch.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda runtime     {torch.version.cuda}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"  [{index}] {props.name}  sm_{props.major}{props.minor}  "
                f"{props.total_memory / 2**30:.1f} GiB  {props.multi_processor_count} SMs"
            )
        peak = roofline.peak_for(torch.cuda.get_device_name(0))
        print(f"roofline peaks   {peak.name if peak else 'no entry -- add one to DEVICE_PEAKS'}"
              f"{'' if not peak else ('  (unverified, run `odyssey peak`)' if not peak.verified else '')}")
    else:
        print("  no CUDA device. Correctness runs on CPU; every latency number needs a GPU.")

    try:
        path = official_mod.script_path()
        print(f"official script  {path}")
        print(f"                 md5 {official_mod.script_md5()}")
        official_mod.load()
        print("                 imports cleanly")
    except Exception as exc:  # noqa: BLE001
        print(f"official script  UNAVAILABLE: {exc}")
        return 1

    print(f"shape sets       {', '.join(available_sets())}")
    print(f"candidates       {', '.join(registry.names())}")
    print(f"runs dir         {RUNS_DIR}")
    print(f"dispatch table   {'present' if DISPATCH_TABLE.exists() else 'absent (baseline fallback)'}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print("candidates:")
    for candidate in registry.available():
        requires = f"  requires={list(candidate.requires)}" if candidate.requires else ""
        print(f"  {candidate.name:<14} {candidate.description}{requires}")
    print("\nshape sets:")
    for name in available_sets():
        shape_set = load_set(name)
        print(f"  {name:<14} {len(shape_set)} case(s): {', '.join(c.name for c in shape_set)}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    cases = _cases(args)
    candidates = [c.name for c in registry.resolve(args.candidates)]
    results = harness.sweep(
        cases,
        candidates,
        compile_candidate=args.compile_candidate,
        compile_mode=args.compile_mode,
        benchmark_on_failure=args.benchmark_on_failure,
        **_eval_kwargs(args),
    )
    for result in results:
        print(result.describe())
    _persist(results, args)
    return 0 if all(r.passed or r.skipped for r in results) else 2


def cmd_ladder(args: argparse.Namespace) -> int:
    cases = _cases(args)
    candidates = [c.name for c in registry.resolve(args.candidates)]
    kwargs = _eval_kwargs(args)
    kwargs.pop("device")
    everything: list[harness.EvalResult] = []
    for case in cases:
        rows = ladder.run_ladder(
            case, candidates, rungs=args.rungs, device=args.device, **kwargs
        )
        print()
        print(ladder.to_markdown(rows))
        everything.extend(rows)
    _persist(everything, args)
    return 0


def cmd_official(args: argparse.Namespace) -> int:
    """Run the organizer's `main()` verbatim with a candidate patched in."""
    candidate = registry.get(args.candidate)

    def build(module: Any) -> Any:
        # The script only ever calls `UserOptimizedTransformer(config)`, so a
        # factory is a sufficient stand-in for the class.
        return lambda config: candidate.build(official=module, config=config, case=None)

    argv = list(args.passthrough)
    if args.case:
        case = load_set(args.shapes).select([args.case]).cases[0]
        argv = case.cli_args() + argv
    print(f"# {candidate.name} via the official script")
    print(f"# argv: {' '.join(argv)}\n")
    return official_mod.run_main(build, argv)


def cmd_calibrate(args: argparse.Namespace) -> int:
    cases = _cases(args)
    decisions, results = calibrate.calibrate(
        cases,
        args.candidates,
        margin=args.margin,
        **_eval_kwargs(args),
    )
    for result in results:
        print(result.describe())
    print()
    print(calibrate.to_markdown(decisions))
    if not args.dry_run:
        path = calibrate.write_table(decisions, results)
        print(f"\ndispatch table -> {path}")
    _persist(results, args)
    return 0


def cmd_peak(args: argparse.Namespace) -> int:
    measured = roofline.measure_peak(device=args.device, dtype=args.dtype, size=args.size)
    print(json.dumps(measured, indent=2))
    path = roofline.save_measured_peak(measured, RUNS_DIR / "measured_peak.json")
    print(f"-> {path}")
    return 0


def cmd_roofline(args: argparse.Namespace) -> int:
    rows = record.BenchmarkLog().rows()
    rows = [r for r in rows if r.get("speedup") and str(r.get("accuracy_passed", "")).lower() == "true"]
    if args.case:
        rows = [r for r in rows if r["case"] in set(args.case)]
    if not rows:
        print("no passing rows in runs/benchmark.csv -- run `odyssey bench` first")
        return 1

    peak = roofline.DEVICE_PEAKS.get(args.peak) or roofline.peak_for(rows[-1].get("gpu", ""))
    if peak is None:
        print(
            f"no roofline peaks for {rows[-1].get('gpu')!r}. "
            f"Pass --peak with one of: {', '.join(roofline.DEVICE_PEAKS)}"
        )
        return 1

    points = []
    for index, row in enumerate(rows):
        case = Case(
            name=row["case"],
            batch_size=int(row["batch_size"]),
            seq_len=int(row["seq_len"]),
            d_model=int(row["d_model"]),
            num_heads=int(row["num_heads"]),
            ffn_dim=int(row["ffn_dim"]),
            num_layers=int(row["num_layers"]),
            causal=row["causal"] == "True",
            dtype=row["dtype"],
            padding_ratio=float(row["padding_ratio"]),
        )
        median = float(row["optimized_median_ms"])
        points.append(
            {
                "label": f"{row['candidate']}/{row['case']}",
                "intensity": roofline.arithmetic_intensity(case),
                "tflops": roofline.achieved_tflops(case, median),
                "round": index,
            }
        )

    out = Path(args.out or ASSETS_DIR / "roofline.svg")
    roofline.plot(points, peak, rows[-1]["dtype"], out)
    print(f"{len(points)} point(s) -> {out}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    if args.scaffold:
        created = ablation.scaffold()
        print(f"scaffolded {len(created)} arm(s) under {ablation.ABLATION_DIR}")
        for path in created:
            print(f"  {path}")
        return 0
    results = ablation.collect()
    print(ablation.to_markdown(results))
    if args.plot:
        out = Path(args.plot if isinstance(args.plot, str) else ASSETS_DIR / "ablation.svg")
        ablation.plot(results, out)
        print(f"\n-> {out}")
    return 0


def cmd_dag(args: argparse.Namespace) -> int:
    dot = record.SolutionDAG().to_dot()
    if args.out:
        Path(args.out).write_text(dot)
        print(f"-> {args.out}")
    else:
        print(dot)
    return 0


# -- parser -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odyssey",
        description="Harness for TikTok TechJam 2026 Track 3.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="environment, GPU and script check").set_defaults(func=cmd_doctor)
    sub.add_parser("list", help="candidates and shape sets").set_defaults(func=cmd_list)

    bench = sub.add_parser("bench", help="sweep candidates over a shape set")
    bench.add_argument("candidates", nargs="*", default=None)
    bench.add_argument("--compile-candidate", action="store_true")
    bench.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    bench.add_argument("--benchmark-on-failure", action="store_true")
    _add_measurement_args(bench)
    bench.set_defaults(func=cmd_bench)

    rung = sub.add_parser("ladder", help="L0 eager / L1 compile / L2 max-autotune / candidates")
    rung.add_argument("candidates", nargs="*", default=["fused-sdpa"])
    rung.add_argument("--rungs", nargs="+", default=list(ladder.DEFAULT_RUNGS))
    _add_measurement_args(rung)
    rung.set_defaults(func=cmd_ladder)

    off = sub.add_parser("official", help="run the organizer's script verbatim with a candidate")
    off.add_argument("candidate")
    off.add_argument("--shapes", default="dev")
    off.add_argument("--case", default=None, help="prepend this case's CLI flags")
    off.add_argument("passthrough", nargs="*", default=[], help="extra flags for the script")
    off.set_defaults(func=cmd_official)

    cal = sub.add_parser("calibrate", help="build runs/dispatch_table.json")
    cal.add_argument("candidates", nargs="*", default=None)
    cal.add_argument("--margin", type=float, default=calibrate.DEFAULT_MARGIN)
    cal.add_argument("--dry-run", action="store_true")
    _add_measurement_args(cal)
    cal.set_defaults(func=cmd_calibrate)

    peak = sub.add_parser("peak", help="measure this card's sustained TFLOPS and bandwidth")
    peak.add_argument("--device", default="cuda")
    peak.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    peak.add_argument("--size", type=int, default=8192)
    peak.set_defaults(func=cmd_peak)

    roof = sub.add_parser("roofline", help="plot candidates on the roofline")
    roof.add_argument("--peak", default=None, help=f"one of: {', '.join(roofline.DEVICE_PEAKS)}")
    roof.add_argument("--case", action="append")
    roof.add_argument("--out", default=None)
    roof.set_defaults(func=cmd_roofline)

    abl = sub.add_parser("ablation", help="scaffold, collect or plot the three-arm ablation")
    abl.add_argument("--scaffold", action="store_true")
    abl.add_argument("--plot", nargs="?", const=True, default=None)
    abl.set_defaults(func=cmd_ablation)

    dag = sub.add_parser("dag", help="export the search DAG as graphviz DOT")
    dag.add_argument("--out", default=None)
    dag.set_defaults(func=cmd_dag)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
