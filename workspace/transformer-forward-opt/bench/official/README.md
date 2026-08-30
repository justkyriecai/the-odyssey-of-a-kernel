# Official Benchmark Script

`torch_transformer_benchmark.py` is the organizer-provided evaluator for TikTok
TechJam 2026 Track 3, vendored here verbatim so that every number in `runs/` is
traceable to a known file.

- Source: attachment on the Track 3 problem statement.
- Vendored: unmodified. `md5 = 21584e5923680ce0455554bd0b45bda2`.

Verify the copy has not drifted:

```bash
md5sum bench/official/torch_transformer_benchmark.py   # macOS: md5 -q
python verify.py --list                                # prints the same md5
```

Point `verify.py` at a different copy with:

```bash
export TECHJAM_BENCHMARK=/path/to/torch_transformer_benchmark.py
```

## Two things to know before trusting a number

**1. The script contradicts itself on tolerance, and the problem statement
settles it.** The module docstring says `atol=0.001, rtol=0.01`; the `argparse`
defaults say `atol=0.002, rtol=0.02`. The statement says *"the diff should be
small enough (relative error < 0.02, abs error < 0.002)"*, so the defaults are
the rule and the docstring is a leftover. `scripts/smoke.sh`, `scripts/demo.sh`
and `scripts/run_ladder.sh` all pass `-- --atol 0.002 --rtol 0.02` -- the same
values, passed explicitly so that every row in `runs/benchmark.csv` records the
tolerance it was judged at rather than inheriting a default that could move.

**2. The correctness rule is an OR, not a sum.** From `compare_outputs`:

```python
abs_ok = abs_error <= atol
rel_ok = abs_error <= rtol * ref.abs()
passed_mask = finite_mask & (abs_ok | rel_ok)
```

The effective tolerance is `max(atol, rtol * |ref|)`, which is tighter than
`torch.isclose`'s `atol + rtol * |ref|` -- the script says so in a comment, and
declines to use `isclose` on purpose. A single non-finite or out-of-tolerance
element fails the whole trial: `passed = failed_elements == 0`.

See `docs/benchmark-anatomy.md` for the full read of the script.
