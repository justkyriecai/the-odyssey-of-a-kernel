#!/usr/bin/env python3
"""Shape #14 (B=32, S=100000, d=1024, H=16, L=2) -- the evidence the official
script cannot produce, labeled for what it is.

The organizer's script OOMs generating its own reference on this shape on any
hardware (the accuracy phase materializes [B, H, S, S] scores: 12.8 TB), so no
official row can exist. This harness produces the two things that can:

  1. A LATENCY measurement of a candidate's forward at S=100000, timed with
     CUDA events under the script's own protocol shape (warmup, repeated
     samples, median/p90) -- but OFF-SCRIPT: the number is ours, not the
     evaluator's, and is reported as such.
  2. An OFF-SCRIPT CORRECTNESS COMPARISON against a chunked reference. The
     reference weights, input generation, config and the element-wise judge
     all come from the vendored script (BaselineTransformer,
     generate_random_case, TransformerConfig, compare_outputs). What is new
     here is only the orchestration: attention is evaluated per batch item in
     query blocks so the [S, S] score matrix never materializes, with the same
     op sequence and fp32 softmax the reference uses. Because that
     orchestration is new numerics code, --self-check first reproduces the
     full unmodified reference at a size it CAN run (default S=2048) and
     requires exact agreement before any extrapolated claim is made.

Nothing here claims a "pass" for shape #14. The output states the comparison
verdict, the tolerance, and the label OFF-SCRIPT in the same breath.

Usage:
    python scripts/stress_100k.py --self-check          # orchestrator vs full script, S=2048
    python scripts/stress_100k.py flash-tf32            # latency + chunked comparison at S=100000
    python scripts/stress_100k.py flash-tf32 --seq-len 20000   # smaller extrapolation point
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from verify import load_script, load_candidates  # noqa: E402


def chunked_reference(model, x, valid_mask, *, q_block: int = 512):
    """The baseline's arithmetic, orchestrated so nothing S x S materializes.

    Op-for-op the sequence is BaselineTransformer.forward's: pre-LN, three
    projections, scaled scores, causal + key masking to -inf, fp32 softmax cast
    back, context, out_proj, output zeroing, FFN with exact GELU, final norm.
    All weights and the config are the script's own; only the loop over batch
    items and query blocks is new, and --self-check proves it changes nothing.
    """
    import torch.nn.functional as F

    config = model.config
    heads, d_model = config.num_heads, config.d_model
    head_dim = d_model // heads
    scale = head_dim**-0.5
    batch, seq_len, _ = x.shape
    causal = config.causal

    outs = []
    for b in range(batch):
        xb = x[b : b + 1]
        mb = valid_mask[b : b + 1]
        drop = ~mb[..., None]
        invalid_keys = ~mb[:, None, None, :]

        h = xb
        for layer in model.layers:
            normed = layer.norm1(h)
            attn = layer.attention
            q = attn.q_proj(normed).view(1, seq_len, heads, head_dim).transpose(1, 2)
            k = attn.k_proj(normed).view(1, seq_len, heads, head_dim).transpose(1, 2)
            v = attn.v_proj(normed).view(1, seq_len, heads, head_dim).transpose(1, 2)

            ctx = torch.empty_like(q)
            positions = torch.arange(seq_len, device=x.device)
            for qs in range(0, seq_len, q_block):
                qe = min(qs + q_block, seq_len)
                scores = torch.matmul(q[:, :, qs:qe], k.transpose(-2, -1)) * scale
                if causal:
                    blocked = positions[None, :] > positions[qs:qe, None]
                    scores = scores.masked_fill(blocked[None, None], float("-inf"))
                scores = scores.masked_fill(invalid_keys, float("-inf"))
                probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
                ctx[:, :, qs:qe] = torch.matmul(probs, v)
                del scores, probs

            context = ctx.transpose(1, 2).reshape(1, seq_len, d_model)
            attn_out = attn.out_proj(context).masked_fill(drop, 0)
            h = h + attn_out
            h = h + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(h)), approximate="none"))
            h = h.masked_fill(drop, 0)

        h = model.final_norm(h).masked_fill(drop, 0)
        outs.append(h)
    return torch.cat(outs, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", nargs="?", default="flash-tf32")
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--self-check", action="store_true",
                        help="verify the chunked orchestrator against the full reference at --self-check-len")
    parser.add_argument("--self-check-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--q-block", type=int, default=512)
    args = parser.parse_args()

    module = load_script()
    device = torch.device("cuda")
    seq_len = args.self_check_len if args.self_check else args.seq_len
    config = module.TransformerConfig(
        batch_size=32, seq_len=seq_len, d_model=1024,
        num_heads=16, ffn_dim=1024, num_layers=2, causal=True,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    baseline = module.BaselineTransformer(config)

    if args.self_check:
        baseline = baseline.to(device=device, dtype=torch.float32).eval()
        x, mask = module.generate_random_case(
            config=config, device=device, dtype=torch.float32,
            seed=args.seed, padding_ratio=0.0, input_scale=1.0,
        )
        with torch.inference_mode():
            direct = baseline(x, mask)
            chunked = chunked_reference(baseline, x, mask, q_block=args.q_block)
        diff = (direct - chunked).abs().max().item()
        print(f"self-check S={seq_len}: max |direct - chunked| = {diff:.3g}")
        ok = diff == 0.0
        print("orchestrator reproduces the reference EXACTLY" if ok
              else f"orchestrator drifts from the reference by {diff:.3g} -- NOT usable")
        return 0 if ok else 1

    candidates = load_candidates()
    cls = candidates[args.candidate][1](module)
    optimized = cls(config)
    module.copy_model_weights(baseline, optimized, strict=True)
    optimized = optimized.to(device=device, dtype=torch.float32).eval()
    baseline_gpu = baseline.to(device=device, dtype=torch.float32).eval()

    x, mask = module.generate_random_case(
        config=config, device=device, dtype=torch.float32,
        seed=args.seed, padding_ratio=0.0, input_scale=1.0,
    )
    print(f"shape #14 axis point: B=32 S={seq_len} d=1024 H=16 L=2 causal, fp32")
    print(f"input: {tuple(x.shape)}, {x.numel() * 4 / 2**30:.1f} GiB")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.warmup):
            optimized(x, mask)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        for i in range(args.repeats):
            starts[i].record()
            optimized(x, mask)
            ends[i].record()
        torch.cuda.synchronize()
    samples = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))]
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"OFF-SCRIPT latency ({args.candidate}): median={med:.1f} ms | "
          f"p90={p90:.1f} ms | n={args.repeats} | peak_mem={peak_gib:.1f} GiB")
    print("(no official-script row exists for this shape: its reference OOMs "
          "in the accuracy phase before any benchmark)")

    print("building chunked reference (per batch item, query blocks; may take minutes)...")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        reference = chunked_reference(baseline_gpu, x, mask, q_block=args.q_block)
        candidate_out = optimized(x, mask)
    result = module.compare_outputs(reference, candidate_out, rtol=args.rtol, atol=args.atol)
    ref_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"OFF-SCRIPT comparison vs chunked reference (judge: the script's own "
          f"compare_outputs, atol={args.atol} rtol={args.rtol}):")
    print(f"  max_abs={result.max_abs_error:.6g} max_rel={result.max_relative_error:.6g} "
          f"failed={result.failed_elements}/{result.total_elements}")
    print(f"  verdict: {'agrees within tolerance' if result.passed else 'DISAGREES'} "
          f"(OFF-SCRIPT; not an official pass claim). peak_mem={ref_peak:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
