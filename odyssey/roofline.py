"""Roofline accounting and the migration plot.

The roofline is this project's best communication device, because it has a
physical ceiling. Two lines -- the card's memory bandwidth and its peak
arithmetic -- bound everything any software can do on that hardware, and
"we went from 11% of the roof to 74%" means something to a judge who has never
written a kernel. It also closes the question it invites: there is no more room
above the line.

Two honesty rules, both enforced here rather than in prose:

**Peak numbers should be measured, not quoted.** `DEVICE_PEAKS` holds vendor
spec-sheet figures so that a plot exists on day one, and every entry is marked
`verified=False`. `measure_peak()` measures what the card in front of you
actually sustains. Use the measured values in anything shown to a judge; a
roofline drawn against a marketing number is a roofline drawn against fiction.

**Arithmetic intensity needs measured DRAM traffic.** `compulsory_bytes()` is a
*lower bound* on traffic -- weights read once, activations in and out, perfect
fusion, infinite cache. Dividing FLOPs by it therefore gives an *upper bound* on
intensity, which flatters every point by pushing it right. Fine for a sanity
plot; not fine for the report. Pass real `dram__bytes.sum` from Nsight Compute
via `dram_bytes=` for the figure that goes on a slide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .shapes import Case

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass(frozen=True)
class Peak:
    name: str
    tflops: dict[str, float]
    bandwidth_gb_s: float
    verified: bool = False
    source: str = "vendor spec sheet"


# Nominal figures, dense (no structured sparsity), for bootstrapping a plot.
# Replace with `measure_peak()` output before quoting any of these.
DEVICE_PEAKS: dict[str, Peak] = {
    "rtx-4090": Peak(
        "NVIDIA GeForce RTX 4090",
        {"float32": 82.6, "tf32": 82.6, "bfloat16": 165.2, "float16": 165.2},
        1008.0,
    ),
    "a100-80gb": Peak(
        "NVIDIA A100 80GB",
        {"float32": 19.5, "tf32": 156.0, "bfloat16": 312.0, "float16": 312.0},
        2039.0,
    ),
    "h100-sxm": Peak(
        "NVIDIA H100 SXM5",
        {"float32": 67.0, "tf32": 494.7, "bfloat16": 989.4, "float16": 989.4},
        3350.0,
    ),
    "l40s": Peak(
        "NVIDIA L40S",
        {"float32": 91.6, "tf32": 91.6, "bfloat16": 362.0, "float16": 362.0},
        864.0,
    ),
}


def peak_for(gpu_name: str) -> Optional[Peak]:
    """Best-effort match from a `torch.cuda.get_device_name` string."""
    lowered = gpu_name.lower()
    for key, peak in DEVICE_PEAKS.items():
        needle = key.split("-")[0].replace("rtx", "").strip()
        if needle and needle in lowered.replace(" ", ""):
            return peak
    return None


# -- analytic accounting ------------------------------------------------------


def matmul_flops(case: Case) -> int:
    """FLOPs in the six matmuls of a layer, times layers.

    Per layer and per batch element, with S tokens and model width d:

        QKV projections   3 * 2 * S * d * d
        scores  q @ k^T       2 * S * S * d
        context probs @ v     2 * S * S * d
        out_proj              2 * S * d * d
        ffn_in                2 * S * d * ffn
        ffn_out               2 * S * ffn * d

    Elementwise work (LayerNorm, GELU, softmax, residuals) is excluded, as is
    conventional for a roofline. `elementwise_flops` counts it separately; at
    these shapes it is a rounding error against the matmuls but it is not
    a rounding error against *time*, which is exactly the point the plot makes.
    """
    b, s, d, f, layers = (
        case.batch_size,
        case.seq_len,
        case.d_model,
        case.ffn_dim,
        case.num_layers,
    )
    per_layer = 8 * b * s * d * d + 4 * b * s * s * d + 4 * b * s * d * f
    return per_layer * layers


def elementwise_flops(case: Case, *, ops_per_element: int = 8) -> int:
    """Rough count for LayerNorm, GELU, softmax and residual adds.

    `ops_per_element` is a coarse constant, not a derivation. Reported so the
    ratio to `matmul_flops` is visible, never folded into a headline number.
    """
    b, s, d, f, layers, heads = (
        case.batch_size,
        case.seq_len,
        case.d_model,
        case.ffn_dim,
        case.num_layers,
        case.num_heads,
    )
    per_layer = 2 * b * s * d + b * s * f + b * heads * s * s + 2 * b * s * d
    return (per_layer * layers + b * s * d) * ops_per_element


def parameter_count(case: Case) -> int:
    d, f, layers = case.d_model, case.ffn_dim, case.num_layers
    attention = 4 * (d * d + d)          # q, k, v, out -- all bias=True
    ffn = (d * f + f) + (f * d + d)
    norms = 2 * (2 * d)                  # two LayerNorms, weight + bias
    return layers * (attention + ffn + norms) + 2 * d  # + final_norm


def compulsory_bytes(case: Case) -> int:
    """Lower bound on DRAM traffic: weights once, activations in and out.

    Assumes perfect fusion and unlimited cache. Real traffic is higher --
    often much higher for the launch-bound shapes in this task -- so anything
    derived from this is a bound, not a measurement.
    """
    itemsize = DTYPE_BYTES[case.dtype]
    weights = parameter_count(case) * itemsize
    activations = 2 * case.batch_size * case.seq_len * case.d_model * itemsize
    return weights + activations


def arithmetic_intensity(case: Case, dram_bytes: Optional[int] = None) -> float:
    return matmul_flops(case) / float(dram_bytes or compulsory_bytes(case))


def achieved_tflops(case: Case, median_ms: float) -> float:
    if median_ms <= 0:
        return float("nan")
    return matmul_flops(case) / (median_ms * 1e-3) / 1e12


def roof(peak: Peak, intensity: float, dtype: str) -> float:
    """Attainable TFLOPS at a given intensity: min(compute roof, bandwidth ramp)."""
    compute = peak.tflops.get(dtype, max(peak.tflops.values()))
    memory = peak.bandwidth_gb_s * intensity / 1000.0
    return min(compute, memory)


def fraction_of_roof(
    case: Case, median_ms: float, peak: Peak, dram_bytes: Optional[int] = None
) -> float:
    intensity = arithmetic_intensity(case, dram_bytes)
    ceiling = roof(peak, intensity, case.dtype)
    if ceiling <= 0:
        return float("nan")
    return achieved_tflops(case, median_ms) / ceiling


def summarize(case: Case, median_ms: float, peak: Optional[Peak] = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "case": case.name,
        "params": parameter_count(case),
        "matmul_gflops": matmul_flops(case) / 1e9,
        "elementwise_gflops": elementwise_flops(case) / 1e9,
        "compulsory_mib": compulsory_bytes(case) / 2**20,
        "intensity_upper_bound": arithmetic_intensity(case),
        "achieved_tflops": achieved_tflops(case, median_ms),
    }
    if peak is not None:
        out["peak"] = peak.name
        out["peak_verified"] = peak.verified
        out["fraction_of_roof"] = fraction_of_roof(case, median_ms, peak)
    return out


# -- measured peaks -----------------------------------------------------------


def measure_peak(
    device: str = "cuda",
    dtype: str = "bfloat16",
    *,
    size: int = 8192,
    iters: int = 50,
) -> dict[str, float]:
    """Measure what this card actually sustains: big-GEMM TFLOPS and copy GB/s.

    A large square GEMM is the standard stand-in for peak arithmetic; a large
    device-to-device copy is the standard stand-in for peak bandwidth. Both
    under-report the marketing number, which is the correct direction of error
    for a ceiling you intend to claim you approached.
    """
    import torch

    dev = torch.device(device)
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    a = torch.randn(size, size, device=dev, dtype=torch_dtype)
    b = torch.randn(size, size, device=dev, dtype=torch_dtype)
    src = torch.empty(1 << 28, device=dev, dtype=torch.uint8)
    dst = torch.empty_like(src)

    def timed(fn) -> float:
        for _ in range(5):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize(dev)
            return start.elapsed_time(end) / iters / 1000.0
        import time

        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters

    gemm_s = timed(lambda: torch.mm(a, b))
    copy_s = timed(lambda: dst.copy_(src))

    return {
        "tflops": 2.0 * size**3 / gemm_s / 1e12,
        "bandwidth_gb_s": 2.0 * src.numel() / copy_s / 1e9,
        "dtype": dtype,
        "gemm_size": size,
    }


def save_measured_peak(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# -- plot ---------------------------------------------------------------------


def plot(
    points: Sequence[dict[str, Any]],
    peak: Peak,
    dtype: str,
    out_path: Path,
    *,
    title: str = "Roofline migration",
    bounded_intensity: bool = True,
) -> Path:
    """Scatter candidates on the roofline. `points` need `label`, `intensity`, `tflops`.

    Optional per-point keys: `round` (colours by search iteration) and `note`.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is needed for the roofline plot: pip install matplotlib"
        ) from exc

    compute = peak.tflops.get(dtype, max(peak.tflops.values()))
    knee = compute * 1000.0 / peak.bandwidth_gb_s

    intensities = [p["intensity"] for p in points] or [knee]
    lo = min(min(intensities) / 4, knee / 8)
    hi = max(max(intensities) * 4, knee * 8)

    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    ys = [min(compute, peak.bandwidth_gb_s * x / 1000.0) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(xs, ys, linewidth=2, color="#2b2b2b", label="attainable")
    ax.axvline(knee, linestyle=":", linewidth=1, color="#888888")
    ax.text(knee, compute * 1.05, f"  knee {knee:.0f} FLOP/byte", fontsize=8, color="#666666")

    rounds = [p.get("round", 0) for p in points]
    scatter = ax.scatter(
        [p["intensity"] for p in points],
        [p["tflops"] for p in points],
        c=rounds,
        cmap="viridis",
        s=64,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )
    for point in points:
        ax.annotate(
            point["label"],
            (point["intensity"], point["tflops"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    if len(set(rounds)) > 1:
        fig.colorbar(scatter, ax=ax, label="search round")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved TFLOP/s")
    subtitle = peak.name + ("" if peak.verified else "  (spec-sheet peaks, unverified)")
    if bounded_intensity:
        subtitle += "  ·  intensity from compulsory-traffic bound"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
