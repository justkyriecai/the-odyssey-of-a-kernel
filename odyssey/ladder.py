"""The baseline ladder.

On stage, "we made it 2.7x faster" invites exactly one question from anyone who
has done inference work: *faster than what?* The official default baseline is
eager PyTorch with TF32 already on -- and the script ships `--compile-baseline`
and `--compile-mode max-autotune`, so a much stronger opponent is one flag away.

Measuring against that opponent before anyone asks turns the question into an
answer. It also shrinks the headline number, which is the point: a net gain over
`torch.compile max-autotune` is a number nobody can take away.

    L0  eager, TF32 on                    the official default, and the denominator
    L1  torch.compile, mode=default       what a competent team gets in an afternoon
    L2  torch.compile, mode=max-autotune  the real opponent
    L3+ our candidates

All rungs are timed in one process against one input, rotating the order between
rounds so clock and thermal drift is shared rather than handed to whichever ran
first. Accuracy for every rung is measured against L0 with the official rule --
including the compiled baselines, because a `max-autotune` rung that drifts out
of tolerance is itself a finding worth having on record.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Iterable, Optional, Sequence

import torch

from . import official as official_mod
from . import registry
from .harness import (
    STRICT_ATOL,
    STRICT_RTOL,
    EvalResult,
    Timing,
    apply_backend_settings,
    check_accuracy,
    describe_device,
    time_modules,
)
from .shapes import Case

DEFAULT_RUNGS: tuple[str, ...] = ("L0", "L1", "L2")

_RUNG_LABELS = {
    "L0": "L0-eager",
    "L1": "L1-compile-default",
    "L2": "L2-compile-max-autotune",
}
_RUNG_MODES = {"L1": "default", "L2": "max-autotune"}


def run_ladder(
    case: Case,
    candidates: Sequence[str] = ("fused-sdpa",),
    *,
    rungs: Iterable[str] = DEFAULT_RUNGS,
    device: str = "auto",
    atol: float = STRICT_ATOL,
    rtol: float = STRICT_RTOL,
    trials: int = 5,
    seed: int = 1234,
    warmup: int = 20,
    repeats: int = 100,
    rounds: int = 3,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    notes: str = "",
) -> list[EvalResult]:
    official = official_mod.load()
    rungs = list(rungs)
    if "L0" not in rungs:
        rungs = ["L0", *rungs]

    resolved_device = official.resolve_device(device)
    dtype = official.resolve_dtype(case.dtype)
    config = official_mod.config_from(official, case)

    apply_backend_settings(
        resolved_device, allow_tf32=allow_tf32, matmul_precision=matmul_precision, seed=seed
    )

    reference = official.BaselineTransformer(config)
    reference = reference.to(device=resolved_device, dtype=dtype).eval()

    contenders: "OrderedDict[str, torch.nn.Module]" = OrderedDict()
    meta: dict[str, dict[str, Any]] = {}

    contenders["L0-eager"] = reference
    meta["L0-eager"] = {"compile": False, "mode": "default"}

    for rung in rungs:
        if rung == "L0":
            continue
        if rung not in _RUNG_MODES:
            raise ValueError(f"unknown rung {rung!r}; known: {sorted(_RUNG_LABELS)}")
        label = _RUNG_LABELS[rung]
        mode = _RUNG_MODES[rung]
        # A fresh copy per rung, so two compile modes never share a wrapper.
        clone = copy.deepcopy(reference).eval()
        contenders[label] = official.maybe_compile(clone, True, mode)
        meta[label] = {"compile": True, "mode": mode}

    skipped: list[EvalResult] = []
    for name in candidates:
        cand = registry.get(name)
        unmet = cand.unmet(case, resolved_device.type)
        if unmet:
            skipped.append(
                _blank_result(
                    cand.name, case, resolved_device, official, allow_tf32,
                    matmul_precision, notes, skipped="; ".join(unmet),
                )
            )
            continue
        module = cand.build(official=official, config=config, case=case)
        official.copy_model_weights(reference, module, strict=cand.strict_weight_copy)
        module = module.to(device=resolved_device, dtype=dtype).eval()
        contenders[cand.name] = module
        meta[cand.name] = {"compile": False, "mode": "default"}

    accuracies = {
        label: check_accuracy(
            reference, module, config, case,
            device=resolved_device, dtype=dtype, trials=trials,
            seed=seed, atol=atol, rtol=rtol,
        )
        for label, module in contenders.items()
    }

    x, valid_mask = official.generate_random_case(
        config=config, device=resolved_device, dtype=dtype,
        seed=seed + 100000, padding_ratio=case.padding_ratio,
        input_scale=case.input_scale,
    )
    timings = time_modules(
        contenders, x, valid_mask, device=resolved_device,
        warmup=warmup, repeats=repeats, rounds=rounds,
    )

    l0 = timings["L0-eager"]
    results = []
    for label in contenders:
        result = _blank_result(
            label, case, resolved_device, official, allow_tf32, matmul_precision, notes
        )
        result.compile_baseline = meta[label]["compile"]
        result.compile_mode = meta[label]["mode"]
        result.accuracy = accuracies[label]
        result.baseline = l0
        result.optimized = timings[label]
        results.append(result)
    return results + skipped


def _blank_result(
    name: str,
    case: Case,
    device: torch.device,
    official: Any,
    allow_tf32: bool,
    matmul_precision: str,
    notes: str,
    skipped: Optional[str] = None,
) -> EvalResult:
    return EvalResult(
        candidate=name,
        case=case,
        device=str(device),
        dtype=case.dtype,
        gpu_name=describe_device(device),
        torch_version=torch.__version__,
        script_md5=official_mod.script_md5(),
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
        notes=notes,
        skipped=skipped,
        baseline=Timing(),
        optimized=Timing(),
    )


def to_markdown(results: Sequence[EvalResult]) -> str:
    """The ladder table, ready to paste into the report."""
    if not results:
        return "_no rungs measured_"
    case = results[0].case
    lines = [
        f"### {case.name} (`{case.key}`) on {results[0].gpu_name}",
        "",
        "| Rung | Correct | median (ms) | p90 (ms) | vs L0 |",
        "|---|:--:|---:|---:|---:|",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| `{r.candidate}` | skip | -- | -- | {r.skipped} |")
            continue
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| `{r.candidate}` | {verdict} | {r.optimized.median_ms:.4f} | "
            f"{r.optimized.p90_ms:.4f} | {r.speedup:.3f}x |"
        )
    if any(r.ran and not r.passed for r in results):
        lines += [
            "",
            "> A rung marked FAIL is timed anyway, because knowing what the shortcut "
            "bought is the reason to reject it. Its speedup is not a result.",
        ]
    return "\n".join(lines)
