"""`evaluate(case, candidate) -> EvalResult`.

Same measurement the official script performs, restructured to return values.
What is imported rather than rewritten:

    generate_random_case   the inputs, including the padding mask construction
    compare_outputs        the OR-tolerance rule and the zero-bad-element verdict
    copy_model_weights     the strict state-dict copy
    warmup_model           warmup
    benchmark_once         cuda.Event timing
    TimingResult           median / mean / p90 / min
    resolve_device         device strings
    resolve_dtype          dtype strings
    maybe_compile          torch.compile wrapping

What is restructured: the two loops around those calls, so that a sweep can
collect numbers instead of reading them off stdout. `odyssey.official.run_main`
runs the script's own `main()` when the printed transcript is what you want.
"""

from __future__ import annotations

import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

import torch

from . import official as official_mod
from . import registry
from .shapes import Case

# The script's docstring and its argparse defaults disagree. These are the
# stricter pair; a candidate that clears them clears either reading.
STRICT_ATOL = 0.001
STRICT_RTOL = 0.01
# What `argparse` actually defaults to, and what the problem statement quotes.
LOOSE_ATOL = 0.002
LOOSE_RTOL = 0.02


@dataclass(frozen=True)
class Timing:
    samples_ms: tuple[float, ...] = ()

    @property
    def median_ms(self) -> float:
        return _stat(self.samples_ms, "median")

    @property
    def mean_ms(self) -> float:
        return _stat(self.samples_ms, "mean")

    @property
    def p90_ms(self) -> float:
        return _stat(self.samples_ms, "p90")

    @property
    def min_ms(self) -> float:
        return _stat(self.samples_ms, "min")

    def summary(self) -> dict[str, float]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "p90_ms": self.p90_ms,
            "min_ms": self.min_ms,
            "samples": len(self.samples_ms),
        }


def _stat(samples: tuple[float, ...], which: str) -> float:
    if not samples:
        return float("nan")
    result = official_mod.load().TimingResult(list(samples))
    return {
        "median": result.median_ms,
        "mean": result.mean_ms,
        "p90": result.p90_ms,
        "min": result.min_ms,
    }[which]


@dataclass(frozen=True)
class TrialResult:
    index: int
    passed: bool
    max_abs_error: float
    max_relative_error: float
    failed_elements: int
    total_elements: int


@dataclass(frozen=True)
class Accuracy:
    atol: float
    rtol: float
    passed: bool
    trials: tuple[TrialResult, ...]

    @property
    def failed_elements(self) -> int:
        return sum(t.failed_elements for t in self.trials)

    @property
    def total_elements(self) -> int:
        return sum(t.total_elements for t in self.trials)

    @property
    def max_abs_error(self) -> float:
        return max((t.max_abs_error for t in self.trials), default=float("nan"))

    @property
    def max_relative_error(self) -> float:
        return max((t.max_relative_error for t in self.trials), default=float("nan"))

    def headroom(self) -> dict[str, float]:
        """How much of the budget was spent. Below 1.0 means margin remains.

        Reported against `atol` alone, which is the floor the rule falls back to
        wherever `|ref|` is small -- the part of the budget that is genuinely
        scarce. `rtol` headroom is not knowable from summary statistics, since
        the worst absolute element need not be the worst relative one.
        """
        return {
            "abs_spent": self.max_abs_error / self.atol if self.atol else float("inf"),
            "rel_spent": self.max_relative_error / self.rtol if self.rtol else float("inf"),
        }


@dataclass
class EvalResult:
    candidate: str
    case: Case
    device: str
    dtype: str
    gpu_name: str
    torch_version: str
    script_md5: str
    accuracy: Optional[Accuracy] = None
    baseline: Timing = field(default_factory=Timing)
    optimized: Timing = field(default_factory=Timing)
    compile_baseline: bool = False
    compile_candidate: bool = False
    compile_mode: str = "default"
    allow_tf32: bool = True
    matmul_precision: str = "high"
    skipped: Optional[str] = None
    error: Optional[str] = None
    notes: str = ""

    @property
    def ran(self) -> bool:
        return self.skipped is None and self.error is None

    @property
    def passed(self) -> bool:
        return bool(self.accuracy and self.accuracy.passed)

    @property
    def speedup(self) -> float:
        if not self.baseline.samples_ms or not self.optimized.samples_ms:
            return float("nan")
        return self.baseline.median_ms / self.optimized.median_ms

    @property
    def tokens_per_second(self) -> float:
        if not self.optimized.samples_ms:
            return float("nan")
        return self.case.tokens * 1000.0 / self.optimized.median_ms

    def describe(self) -> str:
        if self.skipped:
            return f"{self.candidate:<18} {self.case.name:<14} SKIP  {self.skipped}"
        if self.error:
            return f"{self.candidate:<18} {self.case.name:<14} ERROR {self.error}"
        verdict = "PASS" if self.passed else "FAIL"
        acc = self.accuracy
        speed = (
            f"{self.speedup:6.3f}x" if self.baseline.samples_ms else "   --  "
        )
        return (
            f"{self.candidate:<18} {self.case.name:<14} {verdict}  {speed}  "
            f"max_abs={acc.max_abs_error:.3g} max_rel={acc.max_relative_error:.3g} "
            f"base={self.baseline.median_ms:.4f}ms opt={self.optimized.median_ms:.4f}ms"
        )


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or device.type


def apply_backend_settings(
    device: torch.device, *, allow_tf32: bool, matmul_precision: str, seed: int
) -> None:
    """The four lines the official `main()` runs before building anything."""
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def time_modules(
    modules: Mapping[str, torch.nn.Module],
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, Timing]:
    """Time N modules on one input, alternating order between rounds.

    For N=2 this is exactly `benchmark_models`' protocol: warm every module
    first, then reverse the order on odd rounds so thermal and clock drift is
    shared rather than handed to whoever runs first. The generalization to N
    keeps that property for the baseline ladder.
    """
    official = official_mod.load()
    names = list(modules)
    for name in names:
        official.warmup_model(modules[name], x, valid_mask, warmup, device)

    samples: dict[str, list[float]] = {name: [] for name in names}
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        for name in order:
            samples[name].extend(
                official.benchmark_once(modules[name], x, valid_mask, repeats, device)
            )
    return {name: Timing(tuple(values)) for name, values in samples.items()}


def check_accuracy(
    baseline: torch.nn.Module,
    optimized: torch.nn.Module,
    config: Any,
    case: Case,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    atol: float,
    rtol: float,
) -> Accuracy:
    """`run_accuracy_tests` with the prints replaced by a return value."""
    official = official_mod.load()
    results: list[TrialResult] = []

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = official.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=case.padding_ratio,
                input_scale=case.input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = official.compare_outputs(reference, candidate, rtol=rtol, atol=atol)
            results.append(
                TrialResult(
                    index=trial,
                    passed=result.passed,
                    max_abs_error=result.max_abs_error,
                    max_relative_error=result.max_relative_error,
                    failed_elements=result.failed_elements,
                    total_elements=result.total_elements,
                )
            )

    return Accuracy(
        atol=atol,
        rtol=rtol,
        passed=all(t.passed for t in results),
        trials=tuple(results),
    )


def evaluate(
    case: Case,
    candidate: str | registry.Candidate,
    *,
    device: str = "auto",
    atol: float = STRICT_ATOL,
    rtol: float = STRICT_RTOL,
    trials: int = 5,
    seed: int = 1234,
    warmup: int = 20,
    repeats: int = 100,
    rounds: int = 3,
    compile_baseline: bool = False,
    compile_candidate: bool = False,
    compile_mode: str = "default",
    allow_tf32: bool = True,
    matmul_precision: str = "high",
    benchmark_on_failure: bool = False,
    notes: str = "",
) -> EvalResult:
    official = official_mod.load()
    cand = registry.get(candidate) if isinstance(candidate, str) else candidate

    resolved_device = official.resolve_device(device)
    dtype = official.resolve_dtype(case.dtype)

    result = EvalResult(
        candidate=cand.name,
        case=case,
        device=str(resolved_device),
        dtype=case.dtype,
        gpu_name=describe_device(resolved_device),
        torch_version=torch.__version__,
        script_md5=official_mod.script_md5(),
        compile_baseline=compile_baseline,
        compile_candidate=compile_candidate,
        compile_mode=compile_mode,
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
        notes=notes,
    )

    unmet = cand.unmet(case, resolved_device.type)
    if unmet:
        result.skipped = "; ".join(unmet)
        return result

    try:
        config = official_mod.config_from(official, case)
        apply_backend_settings(
            resolved_device,
            allow_tf32=allow_tf32,
            matmul_precision=matmul_precision,
            seed=seed,
        )

        baseline = official.BaselineTransformer(config)
        optimized = cand.build(official, config, case)
        official.copy_model_weights(
            baseline, optimized, strict=cand.strict_weight_copy
        )

        baseline = baseline.to(device=resolved_device, dtype=dtype).eval()
        optimized = optimized.to(device=resolved_device, dtype=dtype).eval()

        # Compile after construction, weight copy, device transfer and eval(),
        # exactly where the official script does it.
        baseline = official.maybe_compile(baseline, compile_baseline, compile_mode)
        optimized = official.maybe_compile(optimized, compile_candidate, compile_mode)

        result.accuracy = check_accuracy(
            baseline,
            optimized,
            config,
            case,
            device=resolved_device,
            dtype=dtype,
            trials=trials,
            seed=seed,
            atol=atol,
            rtol=rtol,
        )

        if not result.accuracy.passed and not benchmark_on_failure:
            return result

        x, valid_mask = official.generate_random_case(
            config=config,
            device=resolved_device,
            dtype=dtype,
            seed=seed + 100000,
            padding_ratio=case.padding_ratio,
            input_scale=case.input_scale,
        )
        timings = time_modules(
            OrderedDict(baseline=baseline, optimized=optimized),
            x,
            valid_mask,
            device=resolved_device,
            warmup=warmup,
            repeats=repeats,
            rounds=rounds,
        )
        result.baseline = timings["baseline"]
        result.optimized = timings["optimized"]
    except Exception as exc:  # noqa: BLE001 - a failing candidate is data, not a crash
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def sweep(
    cases: Iterable[Case],
    candidates: Iterable[str | registry.Candidate],
    **kwargs: Any,
) -> list[EvalResult]:
    cases = list(cases)
    candidates = list(candidates)
    return [
        evaluate(case, candidate, **kwargs)
        for candidate in candidates
        for case in cases
    ]
