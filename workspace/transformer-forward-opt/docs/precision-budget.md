# Precision Budget

The rule gives an element `max(atol, rtol*|ref|)` of room and tolerates zero
failures. This document is where every decision to spend that room, and every
decision not to, gets written down with a measurement next to it.

The point is not to demonstrate care. It is that "the technical complexity
reflects deliberate, capable decision-making" is a claim you can only support
with a record of what you chose *not* to do and why.

## What the budget actually is

Effective tolerance per element, at the strict pair `atol=0.001, rtol=0.01`:

| `\|ref\|` | Binding term | Room |
|---:|---|---:|
| 10 | `rtol` | 0.1 |
| 1 | `rtol` | 0.01 |
| 0.1 | `atol` | 0.001 |
| 0.01 | `atol` | 0.001 |
| ~0 | `atol` | 0.001 |

Above `|ref| = 0.1` the relative term is generous. Below it, everything falls
back to `atol = 0.001` and the budget is flat. **The scarce resource is
absolute error on small-magnitude outputs**, which is exactly where a
LayerNorm-heavy Pre-LN stack puts a lot of its values.

## What each dtype costs

| Format | Mantissa bits | Relative resolution | Verdict |
|---|---:|---:|---|
| FP32 | 23 | ~1e-7 | Free. The reference dtype. |
| TF32 | 10 | ~1e-3 | Already on in the baseline; not a spend. |
| BF16 | 7 (+1 implicit) | ~4e-3 | Usable with fp32 accumulation. Comfortable against `rtol`, tight against `atol`. |
| FP16 | 10 (+1) | ~5e-4 | More resolution than bf16, less range. Watch overflow in scores. |
| FP8 E4M3 | 3 | ~6e-2 | **Not used.** One tail outlier fails the whole trial. |

The FP8 line is a decision, not a limitation: the hardware supports it. Under a
zero-bad-element rule, a format with 6% relative error has no path to passing on
a tensor with millions of elements.

## Where we refuse to spend

**Softmax stays in fp32.** The reference computes
`torch.softmax(scores.float(), dim=-1)` and casts back. Softmax is low
arithmetic intensity and high precision sensitivity -- it is not where the time
is, and it is where the error compounds across six residual layers. The
reference implementation makes the same call, which is a hint worth taking.

Measured, on CPU with bf16: handing q/k/v to `scaled_dot_product_attention`
instead of reproducing the fp32 softmax costs ~2e-3 of absolute error *per
layer*. Six layers of residual accumulation later, `max_abs` is 0.031 against an
`atol` of 0.001, and the run fails. The fused-QKV projection, by contrast, is
bit-identical -- the error is entirely the softmax. That is why `fused-safe` and
`fused-sdpa` are separate named candidates rather than one implementation with a
flag: on hardware where the fused backends *do* accumulate in fp32, `fused-sdpa`
should win, and where they do not it is not admitted at all.

**LayerNorm stays in fp32.** Same reasoning: the mean and variance reductions
are exactly where reduced precision does the most damage per FLOP saved.

**GELU stays exact.** `approximate="none"`. The tanh approximation is worth a
few percent of the FFN's time and about 1e-3 of relative error, which is inside
`rtol` for large values and *outside* `atol` for small ones. Not available under
this rule.

## The table to fill in

One row per (case, candidate), from `runs/benchmark.csv`. "Spent" is
`max_abs / atol`: below 1.0 means margin remains, and how much. `verify.py`
records the `atol` and `rtol` each row was judged against, read off the
script's own `criterion:` line.

```bash
python verify.py fused-safe fused-sdpa graph-safe graph-sdpa --shapes official --record -- --atol 0.001 --rtol 0.01
python - <<'PY'
import csv
rows = list(csv.DictReader(open("runs/benchmark.csv")))
print(f"{'case':<16}{'candidate':<14}{'max_abs':>10}{'spent':>8}{'max_rel':>10}{'ok':>5}")
for r in rows:
    if not r["max_abs"] or not r["atol"]:
        continue
    print(f"{r['case']:<16}{r['candidate']:<14}{float(r['max_abs']):>10.2e}"
          f"{float(r['max_abs']) / float(r['atol']):>8.2f}{float(r['max_rel']):>10.2e}"
          f"{'Y' if r['passed'] == 'True' else 'N':>5}")
PY
```

| Case | Candidate | atol | max_abs | Spent | max_rel | Verdict |
|---|---|---:|---:|---:|---:|:--:|
| _fill from `runs/benchmark.csv`_ | | | | | | |

## Two ways to read this table wrong

**A single number is not a budget.** `max_abs` is the worst element across all
accuracy trials for one case. A candidate at 0.6 of budget on the development
shapes can still fail on a shape with a different value distribution. The row
that matters is the worst row, and it has to come from the full grid.

**`--input-scale` moves the target.** It scales the input, and therefore the
magnitude of every intermediate, and therefore which term of the OR binds. A
budget measured at `input_scale=1.0` does not transfer to `input_scale=4.0`.
Sweep it before claiming margin.
