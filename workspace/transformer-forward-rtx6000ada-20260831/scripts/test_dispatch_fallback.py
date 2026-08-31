#!/usr/bin/env python3
"""A dispatch table naming a candidate that no longer exists must degrade to
the baseline path -- same output, no exception. The first version of this
fallback recursed through forward at serve time; this test is the regression
gate for that failure."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from verify import load_script  # noqa: E402
from kernels.dispatch import dispatch_class, runtime_dispatch_key  # noqa: E402

module = load_script()
config = module.TransformerConfig(
    batch_size=2, seq_len=32, d_model=64, num_heads=4,
    ffn_dim=128, num_layers=2, causal=True,
)
torch.manual_seed(0)
x = torch.randn(2, 32, 64)
mask = torch.ones(2, 32, dtype=torch.bool)

baseline = module.BaselineTransformer(config).eval()
key = runtime_dispatch_key(config, x)
stale = dispatch_class(module)(config, table={key: "renamed-away-candidate"})
stale.load_state_dict(baseline.state_dict())
stale.eval()

with torch.inference_mode():
    want = baseline(x, mask)
    got = stale(x, mask)
assert torch.equal(want, got), "stale-table output differs from the baseline path"
print("dispatch stale-name fallback: baseline output, no recursion -- OK")
