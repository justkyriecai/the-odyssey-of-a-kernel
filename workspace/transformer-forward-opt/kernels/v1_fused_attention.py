"""Fused QKV projection, with the attention numerics as a named choice.

Three costs the baseline pays that neither variant here pays:

1. **Three separate QKV GEMMs.** `q_proj`, `k_proj` and `v_proj` are three
   `[d, d]` matmuls over the same input. They concatenate into one `[3d, d]`
   GEMM. The fused weight is built lazily on the first forward and cached per
   (device, dtype), so the strict state-dict copy still sees the original
   parameter names.

2. **Four whole-tensor materializations per layer.** `_split_heads` is
   `.view().transpose().contiguous()` on each of q, k and v; the attention
   output is transposed and made contiguous again. Reshaping into
   `[3, B, H, S, hd]` and unbinding gives three views with unit stride on the
   head dimension -- no copy. Only the output reshape survives.

3. **A causal mask rebuilt once per layer.** It does not depend on the layer, so
   it is built once per forward -- or skipped entirely in favour of
   `is_causal=True` when there is no padding to combine it with.

## Two variants, because the numerics are a decision

`fused-safe` reproduces the baseline's attention arithmetic exactly: scores in
the working dtype, **softmax in fp32**, probabilities cast back, context in the
working dtype. Given identical q/k/v it is bit-identical to the baseline, so its
error is zero by construction and stays zero at any dtype.

`fused-sdpa` hands the same q/k/v to `scaled_dot_product_attention`. On CUDA the
fused backends accumulate in fp32 and this is typically both faster and accurate
enough. On backends that do not -- the CPU math backend, measurably -- softmax
happens in the working dtype, and in bf16 that is worth about 2e-3 of absolute
error per layer. Six layers of residual accumulation later it clears `atol`, and
the rule tolerates zero bad elements.

That is the entire reason both exist. Which one ships is not a matter of taste:
calibration (`python kernels/dispatch.py calibrate`) measures both on the card in front of you and admits
whichever is correct *and* faster, per geometry. On a card where SDPA does the
right thing, `fused-sdpa` should win. Where it does not, it will not be admitted
at all, and the fallback is a candidate whose error is exactly zero.

GELU stays `approximate="none"` in both. The tanh approximation is worth roughly
1e-3 of relative error for a few percent of the FFN's time, and it is not
available for that price under a zero-bad-element rule.

**Padding is the trap.** With `valid_token_mask`, the baseline masks invalid
*keys* to -inf and then zeroes the output at three separate points: after
`out_proj`, after the block, and after `final_norm`. A candidate that misses any
of the three passes at `--padding-ratio 0` and fails the moment padding is on.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch
import torch.nn.functional as F

SDPA = "sdpa"
FP32_SOFTMAX = "fp32-softmax"


@functools.lru_cache(maxsize=None)
def fused_attention_class(official: Any, attention_impl: str = FP32_SOFTMAX) -> type:
    if attention_impl not in (SDPA, FP32_SOFTMAX):
        raise ValueError(f"unknown attention_impl {attention_impl!r}")

    class FusedAttentionTransformer(official.BaselineTransformer):
        ATTENTION_IMPL = attention_impl

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._qkv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
            self._qkv_key: Optional[tuple] = None

        # -- fused weights ------------------------------------------------
        def _fused_qkv(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
            # Inference tensors cannot escape inference mode, so the mode is part
            # of the cache key: a cache built under inference_mode is not
            # reusable outside it.
            key = (x.device, x.dtype, torch.is_inference_mode_enabled())
            if self._qkv_key == key and self._qkv_cache is not None:
                return self._qkv_cache

            packed = []
            for layer in self.layers:
                attn = layer.attention
                packed.append(
                    (
                        torch.cat(
                            (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight),
                            dim=0,
                        ),
                        torch.cat(
                            (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias),
                            dim=0,
                        ),
                    )
                )

            self._qkv_cache = packed
            self._qkv_key = key
            return packed

        # -- masking ------------------------------------------------------
        def _keep_mask(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
        ) -> tuple[Optional[torch.Tensor], bool]:
            """Return (bool mask where True = attend, whether to use is_causal).

            `is_causal=True` is preferred when nothing has to be combined with
            the triangle: it lets the backend skip materializing S x S.

            No row can be entirely masked out. `generate_random_case` draws
            lengths from `max(1, round(S * (1 - padding_ratio)))` upward, so key
            0 is always valid, and the causal triangle always keeps the diagonal.
            Both facts are load-bearing -- an all-masked row would make softmax
            produce NaN, and the rule fails non-finite values outright.
            """
            causal = self.config.causal
            if valid_token_mask is None:
                return None, causal

            key_mask = valid_token_mask[:, None, None, :]  # [B, 1, 1, S]
            if not causal:
                return key_mask, False

            seq_len = x.shape[1]
            triangle = torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=x.device
            ).tril()
            return triangle[None, None] & key_mask, False

        # -- attention ----------------------------------------------------
        def _attend(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            keep: Optional[torch.Tensor],
            is_causal: bool,
            scale: float,
        ) -> torch.Tensor:
            if self.ATTENTION_IMPL == SDPA:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=keep, is_causal=is_causal, scale=scale
                )

            # Baseline arithmetic, op for op.
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if is_causal:
                seq_len = q.shape[-2]
                blocked = torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=q.device
                ).triu(diagonal=1)
                scores = scores.masked_fill(blocked, float("-inf"))
            elif keep is not None:
                scores = scores.masked_fill(~keep, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            return torch.matmul(probs, v)

        # -- forward ------------------------------------------------------
        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            config = self.config
            batch, seq_len, d_model = x.shape
            heads = config.num_heads
            head_dim = d_model // heads
            scale = head_dim**-0.5

            packed = self._fused_qkv(x)
            keep, is_causal = self._keep_mask(x, valid_token_mask)
            drop = None if valid_token_mask is None else ~valid_token_mask[..., None]

            for index, layer in enumerate(self.layers):
                qkv_weight, qkv_bias = packed[index]

                normed = layer.norm1(x)
                qkv = F.linear(normed, qkv_weight, qkv_bias)
                # [B, S, 3d] -> [3, B, H, S, hd] as views; head_dim keeps stride 1.
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

    FusedAttentionTransformer.__name__ = (
        "FusedSDPATransformer" if attention_impl == SDPA else "FusedSafeTransformer"
    )
    return FusedAttentionTransformer


def build_safe(official: Any) -> type:
    return fused_attention_class(official, FP32_SOFTMAX)


def build_sdpa(official: Any) -> type:
    return fused_attention_class(official, SDPA)


CANDIDATES = {
    "fused-safe": (
        "Fused QKV, no per-head copies, baseline-exact fp32 softmax. Zero error by construction.",
        build_safe,
    ),
    "fused-sdpa": (
        "Fused QKV, no per-head copies, scaled_dot_product_attention. Faster where the backend keeps fp32 accumulation.",
        build_sdpa,
    ),
}
