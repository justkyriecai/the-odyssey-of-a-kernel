"""Candidate implementations of `UserOptimizedTransformer`.

Each module exposes `CANDIDATES`: name -> (description, factory), where the
factory takes the loaded benchmark script as its only argument and returns the
class to substitute for `UserOptimizedTransformer`. `verify.py` collects them
from here; nothing registers anywhere.

The lineage is deliberate and each rung has a job:

    passthrough   the baseline itself. Proves the measurement reads ~1.00x and
                  bit-identical output; any drift here is a harness bug.
    fused-safe    one QKV GEMM instead of three, no per-head .contiguous()
                  materializations, mask built once per forward instead of once
                  per layer -- and the baseline's exact fp32 softmax, so its
                  error is zero at every dtype.
    fused-sdpa    the same, handing q/k/v to scaled_dot_product_attention.
                  Faster where the backend accumulates softmax in fp32; out of
                  tolerance in bf16 where it does not. Measured, not assumed.
    graph-safe    fused-safe captured into a CUDA Graph. The default shape runs
    graph-sdpa    ~90 kernel launches per forward over 1024 tokens; if launch
                  overhead dominates, this is where it shows up.
    compiled-*    the fused body handed to torch.compile inside the candidate:
                  Inductor's fusion plus (in the -ro variants) cudagraph trees.
                  GEMMs stay on cuBLAS so the numerics track the reference.
    dispatch      the shipping layer. Per geometry, use whichever candidate a
                  calibration run proved correct and faster -- otherwise fall
                  back to the baseline path, so the answer is never slower than
                  what it replaces.
"""

from __future__ import annotations

from . import dispatch, v0_passthrough, v1_fused_attention, v2_cuda_graph, v3_compiled

CANDIDATES: dict = {}
for _module in (v0_passthrough, v1_fused_attention, v2_cuda_graph, v3_compiled, dispatch):
    CANDIDATES.update(_module.CANDIDATES)

__all__ = [
    "CANDIDATES",
    "v0_passthrough",
    "v1_fused_attention",
    "v2_cuda_graph",
    "v3_compiled",
    "dispatch",
]
