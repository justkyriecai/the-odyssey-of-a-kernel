"""Online-softmax causal attention in Triton, fp32 accumulation throughout.

The stress shape (B=32, S=100000, d=1024, H=16, L=2) makes this mandatory, not
an optimization: the baseline materializes [B, H, S, S] scores -- 12.8 TB there
-- and its own causal mask alone is ~10 GB. This kernel never materializes
S x S anything: each query block streams over key/value blocks keeping a
running max and running denominator (the flash-attention recurrence), so
memory is O(S * d) and the causal triangle is an index comparison, not a
tensor.

Numerics, deliberately: scores and the softmax recurrence are computed in IEEE
fp32 (`allow_tf32=False` on both dots). The measured landscape behind that
choice: Triton TF32 GEMMs drift ~5e-3 from the cuBLAS TF32 reference (fails
the official 0.002), while the SDPA mem-efficient backend -- IEEE fp32
internally -- lands at ~1e-3 and passes. IEEE puts this kernel in the second
camp. fp32 lanes only: reduced precision falls back to the eager fused path,
whose error is zero by construction; the name says so.

Padding rides in the kernel: the evaluator's `valid_token_mask` is always a
prefix mask (positions < per-row length), so a `[B]` vector of lengths replaces
the whole [B, 1, S, S] combined mask. Key j of batch b attends iff
j < length[b], and causality adds j <= i. A non-prefix mask (impossible under
the evaluator, possible under the bare forward contract) is detected once per
forward and routed to the eager path.

CUDA only: on CPU this candidate falls back to the eager fused path, so a CPU
number for `flash-fp32` is not a flash number.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

try:  # Triton ships with the CUDA torch build; CPU-only boxes run the fallback.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from .v1_fused_attention import FP32_SOFTMAX, fused_attention_class

if HAVE_TRITON:

    @triton.jit
    def _flash_fwd(
        q_ptr, k_ptr, v_ptr, o_ptr, lengths_ptr,
        stride_qz, stride_qm, stride_qd,
        stride_kz, stride_kn, stride_kd,
        stride_vz, stride_vn, stride_vd,
        stride_oz, stride_om, stride_od,
        seq_len, scale,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_z = tl.program_id(1)  # fused batch * heads index
        batch = pid_z // HEADS

        length = tl.load(lengths_ptr + batch)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_base = q_ptr + pid_z * stride_qz
        q = tl.load(
            q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < seq_len,
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Causal rows never look past their own block; padded keys never past
        # the row length. Either bound alone caps the loop.
        if IS_CAUSAL:
            hi = tl.minimum((pid_m + 1) * BLOCK_M, length)
        else:
            hi = length

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_base = k_ptr + pid_z * stride_kz
            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k), allow_tf32=False) * scale

            keep = offs_n[None, :] < length
            if IS_CAUSAL:
                keep = keep & (offs_n[None, :] <= offs_m[:, None])
            scores = tl.where(keep, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            # Fully-masked rows keep m == -inf; exp(-inf - -inf) would be NaN.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(scores - m_safe[:, None])
            p = tl.where(keep, p, 0.0)
            rescale = tl.exp(tl.where(m_i == float("-inf"), 0.0, m_i) - m_safe)

            l_i = l_i * rescale + tl.sum(p, axis=1)
            acc = acc * rescale[:, None]

            v_base = v_ptr + pid_z * stride_vz
            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(p, v, allow_tf32=False)
            m_i = m_new

        # Rows with no visible key (an invalid query row of a padded batch)
        # produce zeros; the caller's output masking zeroes them anyway.
        denom = tl.where(l_i == 0.0, 1.0, l_i)
        out = acc / denom[:, None]

        o_base = o_ptr + pid_z * stride_oz
        tl.store(
            o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
            out,
            mask=offs_m[:, None] < seq_len,
        )


def flash_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    *,
    causal: bool,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """q, k, v: [B, H, S, hd] fp32 (any stride); lengths: [B] int32. -> [B, H, S, hd]."""
    batch, heads, seq_len, head_dim = q.shape
    q3 = q.reshape(batch * heads, seq_len, head_dim)
    k3 = k.reshape(batch * heads, seq_len, head_dim)
    v3 = v.reshape(batch * heads, seq_len, head_dim)
    out = torch.empty_like(q3, memory_format=torch.contiguous_format)

    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _flash_fwd[grid](
        q3, k3, v3, out, lengths,
        q3.stride(0), q3.stride(1), q3.stride(2),
        k3.stride(0), k3.stride(1), k3.stride(2),
        v3.stride(0), v3.stride(1), v3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        seq_len, scale,
        HEADS=heads, HEAD_DIM=head_dim, IS_CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.reshape(batch, heads, seq_len, head_dim)


@functools.lru_cache(maxsize=None)
def flash_class(official: Any) -> type:
    base = fused_attention_class(official, FP32_SOFTMAX)

    class FlashFP32Transformer(base):
        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if (
                not HAVE_TRITON
                or x.device.type != "cuda"
                or x.dtype != torch.float32
            ):
                return super().forward(x, valid_token_mask)

            batch, seq_len, d_model = x.shape
            heads = self.config.num_heads
            head_dim = d_model // heads
            if head_dim < 16:
                # tl.dot needs >=16 per dimension; narrow heads take the eager path.
                return super().forward(x, valid_token_mask)

            if valid_token_mask is None:
                lengths = torch.full(
                    (batch,), seq_len, dtype=torch.int32, device=x.device
                )
                drop = None
            else:
                lengths = valid_token_mask.sum(dim=1, dtype=torch.int32)
                # The evaluator's masks are prefix masks. Anything else exists
                # only off-harness; detect it once and take the exact path.
                positions = torch.arange(seq_len, device=x.device)
                if not torch.equal(valid_token_mask, positions[None, :] < lengths[:, None]):
                    return super().forward(x, valid_token_mask)
                drop = ~valid_token_mask[..., None]

            packed = self._fused_qkv(x)
            scale = head_dim**-0.5
            causal = self.config.causal

            import torch.nn.functional as F  # local: keep module surface small

            for index, layer in enumerate(self.layers):
                qkv_weight, qkv_bias = packed[index]
                normed = layer.norm1(x)
                qkv = F.linear(normed, qkv_weight, qkv_bias)
                q, k, v = (
                    qkv.view(batch, seq_len, 3, heads, head_dim)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )

                context = flash_attention_fp32(
                    q, k, v, lengths, causal=causal, scale=scale
                )
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

    return FlashFP32Transformer


CANDIDATES = {
    "flash-fp32": (
        "Triton online-softmax causal attention, IEEE fp32 accumulation, padding "
        "via per-row lengths -- O(S*d) memory, never materializes S x S. fp32 CUDA "
        "only; other dtypes, CPU, narrow heads (head_dim<16) and non-prefix masks "
        "fall back to the eager fused path.",
        flash_class,
    ),
}
