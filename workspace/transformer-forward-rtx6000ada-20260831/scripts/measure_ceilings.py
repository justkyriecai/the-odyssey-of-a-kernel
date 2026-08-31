#!/usr/bin/env python3
"""Measure this card's practical ceilings: a large TF32 GEMM and a large copy.

The roofline uses these, never spec-sheet numbers: a spec quotes boost-clock
peaks the timing protocol never sees. Median of repeated runs, CUDA events.
Writes runs/ceilings.json.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")

def timed(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), sorted(out)[int(round(0.9 * (len(out) - 1)))]

results = {"device": torch.cuda.get_device_name(0)}

# Large TF32 GEMM: 8192^3, 2*N^3 FLOPs.
n = 8192
a = torch.randn(n, n, device=dev)
b = torch.randn(n, n, device=dev)
med, p90 = timed(lambda: a @ b)
results["gemm_tf32"] = {"n": n, "median_ms": med, "p90_ms": p90,
                        "tflops": 2 * n**3 / (med / 1e3) / 1e12}

# Large fp32 IEEE GEMM (the flash-fp32 lane's ceiling).
torch.backends.cuda.matmul.allow_tf32 = False
med, p90 = timed(lambda: a @ b)
results["gemm_fp32_ieee"] = {"n": n, "median_ms": med, "p90_ms": p90,
                             "tflops": 2 * n**3 / (med / 1e3) / 1e12}
torch.backends.cuda.matmul.allow_tf32 = True

# Large device-to-device copy: 4 GiB tensor, read+write = 8 GiB moved.
src = torch.empty(1 << 30, device=dev)  # 4 GiB fp32
dst = torch.empty_like(src)
med, p90 = timed(lambda: dst.copy_(src))
results["copy"] = {"bytes_moved": 2 * src.numel() * 4, "median_ms": med, "p90_ms": p90,
                   "gbps": 2 * src.numel() * 4 / (med / 1e3) / 1e9}

out = Path(__file__).resolve().parent.parent / "runs" / "ceilings.json"
out.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
