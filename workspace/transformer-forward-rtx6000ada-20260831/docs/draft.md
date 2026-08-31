# Round 5 draft -- the reduced-precision lanes, where a 14x gap is still open

Branched from `workspace/transformer-forward-opt` onto the same card (RTX 6000
Ada, sm_89) with an empty ledger. Everything cited below is the **sibling's**
evidence and is therefore an expectation, not this round's result. Nothing here
may be quoted until it has been re-measured in this workspace.

## Round target (set against measurement, no goalpost movement)

Same framing as the sibling: beat the fastest **numerically admissible**
`torch.compile` configuration of the baseline on every measured lane, never
slower than the eager baseline anywhere. This round spends its iterations on
the lanes where that bar is currently met by the smallest margin -- the
reduced-precision lanes -- rather than on widening leads that are already
double digits.

Correctness bar unchanged: official tolerance `--atol 0.002 --rtol 0.02`, zero
bad elements, judged against the uncompiled eager reference.

## What is known, with evidence

From the sibling's 387-row `runs/benchmark.csv`. The dispatch validation over
the official fp32 grid ran 16.4x (seq-1024) down to 1.25x (wide-1024), worst
case 1.23x. That is not where the headroom is. The headroom is here:

| lane | best passing | candidate | fp32 sibling lane | gap |
|---|---|---|---|---|
| `seq-1024-bf16` | **1.18x** | `graph-safe` | 16.43x | **14x** |
| `seq-1024-fp16` | **1.18x** | `graph-safe` | 16.43x | **14x** |
| `batch-128-bf16` | 1.19x | `graph-safe` | 2.37x | 2.0x |
| `batch-128-fp16` | 1.16x | `graph-safe` | 2.37x | 2.0x |
| `heads-16-bf16` | 1.08x | `graph-safe` | 3.67x | 3.4x |
| `heads-16-fp16` | 1.05x | `graph-safe` | 3.67x | 3.5x |
| `center-bf16` | 2.57x | `graph-safe` | 4.36x | 1.7x |

**Every reduced-precision lane in the campaign is served by `graph-safe`**, the
bit-exact replay path, and by nothing else.

### Why: in bf16/fp16 the rule is bit-exactness, not accuracy

Across all 49 reduced-precision rows, the split is total and has no middle:

- Everything that **passes** records `max_abs = 0` and `max_rel = 0`.
  `passthrough`, `fused-safe`, `graph-safe`, `dispatch`.
- Everything that **fails** records `max_abs = 0.0625` (bf16) or `0.00586`--
  `0.00977` (fp16), with `max_rel` between 3.6e8 and 6.0e9.
  `compiled-*`, `fused-sdpa`, `graph-sdpa`.

Two numbers explain the whole table. `0.0625 = 2^-4` is **one bf16 ULP** at
magnitude 8--16: the failing candidates are not inaccurate, they are one
rounding step away. And `max_rel ~ 1e9` says the binding elements are ones
where `|ref|` is ~1e-11, so `rtol * |ref|` vanishes and the effective tolerance
collapses to the `atol` floor of 0.002. A single ULP of a moderately sized bf16
value is 30x that floor.

So the reference's own low-precision rounding *is* the specification. Being
more accurate than the reference fails exactly as hard as being less accurate.
This is the "rounding-point mechanism" the sibling's goal tracker recorded, now
with the mechanism named: **any transformation that changes a rounding point in
bf16/fp16 is inadmissible, no matter how small its error.**

That is why SDPA, flash and every Inductor GEMM template fail reduced
precision, and why the only survivors replay the reference's exact kernels.

### What that leaves as legal

Bit-preserving transformations, which is a narrower set than it first looks:

- Removing launch overhead. `graph-safe` does this and is the current holder.
- Fusing elementwise work **without reassociating** anything.
- Eliminating memory round-trips where the arithmetic per element is unchanged.

And what is illegal: any change to a GEMM's accumulation order (so cuBLAS calls
must stay the identical cuBLAS calls), and streaming/online softmax (flash's
rescaling changes the summation order by construction).

The open question this round exists to answer: `graph-safe` gets 2.57x at
`center-bf16`, where the work is launch overhead, but only **1.18x** at
`seq-1024-bf16`, where the work is real. Pure replay cannot help a lane that is
not launch-bound. Is there a bit-exact transformation that can?

## Ranked directions

### 1. `fused-exact` -- bit-exact fusion of the attention glue (rank 1)

At `seq-1024` the reference materializes `scores` as `[64, 4, 1024, 1024]` --
268M elements, ~537 MB in bf16 -- then writes it, reads it back for the fp32
softmax, writes `probs`, and reads that for the second GEMM. Three round trips
of a half-gigabyte tensor per layer, four layers.

The softmax itself is elementwise-over-a-row and is done in fp32 by the
reference. A full-row (not streaming) softmax that visits the row in the same
order reduces in the same order, and is therefore **bit-exact by construction**
-- unlike flash's online rescaling. So the mask, the fp32 softmax and the cast
back to bf16 can be fused into one pass over `scores` while leaving both GEMMs
as the identical cuBLAS calls the reference makes.

- Benefit: attacks the 14x gap directly, on the lane where it is largest, with
  a transformation that is legal by construction rather than by luck.
- Risk: "same order" has to be verified, not assumed -- PyTorch's softmax
  reduction order is an implementation detail and may be tiled. If it does not
  reproduce bit-exactly, the direction dies at iteration 2, cheaply, and that
  null result is itself worth recording. Also: saving memory traffic only pays
  if the lane is bandwidth-bound, which iteration 1 must establish first.
- Iterations (cap 5): (1) **the per-dtype fidelity table** -- the sibling's own
  open `task8`, never built: per operation, in each dtype, does a candidate
  implementation reproduce the reference bit-exactly? This is the map the whole
  direction navigates by, and it is cheap. (2) Profile `seq-1024-bf16` under
  `graph-safe` and establish whether it is bandwidth- or tensor-core-bound; if
  it is compute-bound at the cuBLAS roof, stop here and say so. (3) Fused
  mask+fp32-softmax+cast kernel, bit-exactness gate before any timing.
  (4) Extend the fusion to the LayerNorm/GELU/residual glue if (1) says those
  are reproducible. (5) Dispatch integration and the full reduced-precision
  sweep.

### 2. `wide-1024` -- the worst fp32 lane, against a measured roof (rank 2)

`wide-1024` (d_model = ffn = 1024) is the weakest fp32 lane at 1.25x. Unlike
the launch-bound lanes it is genuine GEMM work, so the honest question is not
"how do we speed it up" but "how much room is there above cuBLAS on this card".

- Benefit: it is the lane that sets the worst-case number, and the worst case is
  what the target is stated in terms of.
- Risk: the likely answer is "almost none", in which case the deliverable is a
  roofline showing the lane is already near the measured GEMM ceiling. That is
  a legitimate and reportable result, but it is not a speedup, which is why it
  ranks below direction 1.
- Iterations (cap 5): (1) re-measure `scripts/measure_ceilings.py` on this box
  and place `wide-1024` on it; (2) profile the lane; (3) if the gap to the roof
  is under ~15%, record the ceiling result and stop -- do not burn the
  remaining iterations proving a known negative; (4-5) reserved.

### 3. Re-establish the ledger (rank 0 -- blocks everything above)

Not an optimization direction; the precondition. `runs/` is empty by design and
no number above may be quoted until this workspace has produced it.

- `./scripts/check_gpu.sh` on the box, including the profiler permission probe.
- `./scripts/smoke.sh` -- `passthrough` at ~1.00x, zero error.
- Re-measure the eager baseline and the admissible-compile opponent, fresh
  process per case.
- Re-measure `graph-safe` on the reduced-precision lanes, since it is the
  incumbent that direction 1 must beat.

## The machine

No pod is running. Network volumes `cuda-lab` (US-CA-2) and `cuda-lab-lower`
(EU-RO-1) both survive, so the checkout, the venv and the tooling come back
with a pod attached to one of them -- `docs/runpod.md` has the procedure.
Direction 3 cannot start until a card is up; directions 1 and 2 are planning
only until then.
