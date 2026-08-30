#!/usr/bin/env python3
"""Roofline of the dispatch-served official shapes against ceilings measured on
this card (runs/ceilings.json; a large GEMM and a large copy, never a spec
sheet).

Honesty note baked into the plot: hardware byte counters are denied on this pod
(ERR_NVGPUCTRPERM), so arithmetic intensity is ANALYTIC -- model FLOPs over a
minimum-traffic byte model (weights + activations once per op) -- not measured
DRAM traffic. The ceilings and the achieved points are measured.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path(__file__).resolve().parent.parent
ceil = json.loads((WS / "runs" / "ceilings.json").read_text())
PEAK_TFLOPS = ceil["gemm_tf32"]["tflops"]
PEAK_GBPS = ceil["copy"]["gbps"]

def model_flops(B, S, d, H, ffn, L):
    per_layer = (2 * B * S * d * 3 * d      # QKV
                 + 2 * B * H * S * S * (d // H) * 2  # scores + probs@V
                 + 2 * B * S * d * d        # out proj
                 + 2 * B * S * d * ffn * 2) # FFN in + out
    return per_layer * L

def model_bytes(B, S, d, H, ffn, L):
    # Minimum-traffic model: weights once, activations read+write once per op.
    act = B * S * d * 4
    per_layer = (4 * d * d + 2 * d * ffn) * 4 + 10 * act
    return per_layer * L

rows = list(csv.DictReader(open(WS / "runs" / "benchmark.csv")))
latest = {}
for r in rows:
    if r["candidate"] != "dispatch" or "round 2" not in r["notes"]:
        continue
    latest[r["case"]] = r

fig, ax = plt.subplots(figsize=(9, 6))
xs = [2 ** i for i in range(-2, 13)]
roof = [min(PEAK_TFLOPS, PEAK_GBPS * x / 1e3) for x in xs]
ax.plot(xs, roof, "k-", lw=2,
        label=f"measured roofs: {PEAK_TFLOPS:.0f} TF (TF32 GEMM), {PEAK_GBPS:.0f} GB/s (copy)")

for case, r in sorted(latest.items()):
    B, S, d = int(r["batch_size"]), int(r["seq_len"]), int(r["d_model"])
    H, ffn, L = int(r["num_heads"]), int(r["ffn_dim"]), int(r["num_layers"])
    fl = model_flops(B, S, d, H, ffn, L)
    by = model_bytes(B, S, d, H, ffn, L)
    t_opt = float(r["optimized_median_ms"]) / 1e3
    t_base = float(r["baseline_median_ms"]) / 1e3
    ax.plot(fl / by, fl / t_opt / 1e12, "o", ms=8, color="tab:blue")
    ax.plot(fl / by, fl / t_base / 1e12, "x", ms=6, color="tab:gray")
    ax.annotate(case, (fl / by, fl / t_opt / 1e12), textcoords="offset points",
                xytext=(6, 4), fontsize=8)

ax.plot([], [], "o", color="tab:blue", label="dispatch (served candidate), median")
ax.plot([], [], "x", color="tab:gray", label="eager baseline, median")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("analytic arithmetic intensity [FLOP/byte]  (byte counters denied on this pod)")
ax.set_ylabel("achieved [TFLOP/s]")
ax.set_title("Official shapes on RTX 6000 Ada -- measured ceilings, analytic intensity")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
out = WS / "docs" / "assets" / "roofline.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out}")
for case, r in sorted(latest.items()):
    B, S, d = int(r["batch_size"]), int(r["seq_len"]), int(r["d_model"])
    H, ffn, L = int(r["num_heads"]), int(r["ffn_dim"]), int(r["num_layers"])
    fl = model_flops(B, S, d, H, ffn, L)
    t = float(r["optimized_median_ms"]) / 1e3
    print(f"{case:<14} intensity {fl/model_bytes(B,S,d,H,ffn,L):8.1f} FLOP/B   achieved {fl/t/1e12:6.2f} TF   roof-bound {'compute' if fl/model_bytes(B,S,d,H,ffn,L) > PEAK_TFLOPS*1e3/PEAK_GBPS else 'memory'}")
