"""torch.compile applied inside the candidate, over the fused forward body.

The organizer's own example list names torch.compile as a permitted direction,
and the measured ceiling says why it matters here: Inductor turns the eager
baseline's ~116 kernels per forward into a handful, worth 2.4-10x on the
launch-bound shapes. Compiling *our* fused body hands Inductor a strictly
cheaper program than the baseline it already speeds up: one QKV GEMM per layer
instead of three, the mask built once per forward instead of once per layer.

What is compiled, and what deliberately is not:

- The compiled region is a pure function of (x, masks, packed weights). The
  fused-QKV packing and mask construction run eagerly before it, because the
  packing caches on `self` and a cache mutation inside the traced region would
  either break the graph or silently retrace every call.
- `torch.compile` wraps a bound method, not the module, so what comes back is a
  plain callable. It is cached in `__dict__` -- never a submodule, so the
  script's `load_state_dict(strict=True)` weight copy sees exactly the baseline
  parameter names.
- GEMMs stay on cuBLAS (default/reduce-overhead modes do not autotune Triton
  templates). That is a numerics decision: the eager reference runs cuBLAS
  TF32, and the measured drift of an identically-decomposed compiled program is
  ~7e-4 -- inside the official tolerance -- while max-autotune's Triton TF32
  templates drift ~5e-3 and fail.
- Reduced precision is expected to fail here and is measured anyway: the
  reference rounds scores to bf16 before its fp32 softmax, and Inductor's
  fusion keeps values in fp32 registers across that boundary; the compiled
  *baseline* already drifts 0.0625 in bf16 for exactly this reason. The safe
  graph-captured path serves those lanes.

CUDA only: on CPU these candidates fall back to the eager fused path, so a CPU
number for `compiled-*` is not a compiled number.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .v1_fused_attention import SDPA, FP32_SOFTMAX, fused_attention_class

# An official-grid sweep runs every case in one process, and each case builds a
# fresh model around the same code object. Dynamo's default of 8 cached
# compilations per code object would silently fall back to eager partway
# through such a sweep -- recording eager numbers under a compiled candidate's
# name. Raised once, at import; this is the single sanctioned piece of global
# state these candidates touch.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


@functools.lru_cache(maxsize=None)
def compiled_class(official: Any, attention_impl: str, mode: str) -> type:
    base = fused_attention_class(official, attention_impl)

    class CompiledTransformer(base):
        COMPILE_MODE = mode

        def _body(
            self,
            x: torch.Tensor,
            keep: Optional[torch.Tensor],
            is_causal: bool,
            drop: Optional[torch.Tensor],
            qkv_weights: list[torch.Tensor],
            qkv_biases: list[torch.Tensor],
        ) -> torch.Tensor:
            """The whole forward as a pure function; everything Inductor sees."""
            batch, seq_len, d_model = x.shape
            heads = self.config.num_heads
            head_dim = d_model // heads
            scale = head_dim**-0.5

            for index, layer in enumerate(self.layers):
                normed = layer.norm1(x)
                qkv = F.linear(normed, qkv_weights[index], qkv_biases[index])
                q, k, v = (
                    qkv.view(batch, seq_len, 3, heads, head_dim)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )

                context = self._attend(q, k, v, keep, is_causal, scale)
                context = context.transpose(1, 2).reshape(batch, seq_len, d_model)

                out_proj = layer.attention.out_proj
                attn_out = F.linear(context, out_proj.weight, out_proj.bias)
                if drop is not None:
                    attn_out = attn_out.masked_fill(drop, 0)
                x = x + attn_out

                x = x + layer.ffn_out(
                    F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
                )
                if drop is not None:
                    x = x.masked_fill(drop, 0)

            x = self.final_norm(x)
            if drop is not None:
                x = x.masked_fill(drop, 0)
            return x

        def _compiled_body(self, x: torch.Tensor) -> Any:
            # Plain __dict__ storage: a function is not a Module, nothing is
            # registered, the state dict stays byte-identical to the baseline's.
            key = (x.device, x.dtype, torch.is_inference_mode_enabled())
            cache = self.__dict__.setdefault("_compiled_cache", {})
            fn = cache.get(key)
            if fn is None:
                fn = torch.compile(self._body, mode=self.COMPILE_MODE, dynamic=False)
                cache[key] = fn
            return fn

        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if x.device.type != "cuda":
                return super().forward(x, valid_token_mask)

            packed = self._fused_qkv(x)
            keep, is_causal = self._keep_mask(x, valid_token_mask)
            drop = None if valid_token_mask is None else ~valid_token_mask[..., None]
            qkv_weights = [w for w, _ in packed]
            qkv_biases = [b for _, b in packed]
            return self._compiled_body(x)(x, keep, is_causal, drop, qkv_weights, qkv_biases)

    CompiledTransformer.__name__ = (
        f"Compiled{'SDPA' if attention_impl == SDPA else 'Safe'}Transformer"
    )
    return CompiledTransformer


def _factory(attention_impl: str, mode: str):
    def build(official: Any) -> type:
        return compiled_class(official, attention_impl, mode)

    return build


CANDIDATES = {
    "compiled-safe": (
        "fused body under torch.compile (default mode): Inductor-fused glue, cuBLAS "
        "GEMMs, exact fp32 softmax semantics. CUDA only; falls back to the eager "
        "fused path on CPU.",
        _factory(FP32_SOFTMAX, "default"),
    ),
    "compiled-safe-ro": (
        "the same compiled fused body with mode=reduce-overhead: cudagraph trees on "
        "top of the fusion. CUDA only; eager fused fallback on CPU.",
        _factory(FP32_SOFTMAX, "reduce-overhead"),
    ),
    "compiled-sdpa": (
        "compiled fused body with scaled_dot_product_attention: no S x S "
        "materialization plus Inductor-fused glue. Reduced precision expected to "
        "fail for SDPA's reasons; calibration decides per lane. CUDA only; eager "
        "fused-sdpa fallback on CPU.",
        _factory(SDPA, "default"),
    ),
    "compiled-sdpa-ro": (
        "compiled SDPA body with mode=reduce-overhead cudagraph trees. CUDA only; "
        "eager fused-sdpa fallback on CPU.",
        _factory(SDPA, "reduce-overhead"),
    ),
}
