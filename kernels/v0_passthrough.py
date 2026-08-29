"""The identity candidate.

Not an optimization -- a control. It runs the baseline's own forward through
the candidate slot, so it must report bit-identical output and a speedup within
noise of 1.00x. When it does not, the harness is lying and no other number in
`runs/benchmark.csv` can be trusted until that is explained.
"""

from __future__ import annotations

import functools
from typing import Any

from odyssey.registry import register


@functools.lru_cache(maxsize=None)
def _passthrough_class(official: Any) -> type:
    class PassthroughTransformer(official.BaselineTransformer):
        """Baseline forward, unchanged."""

    return PassthroughTransformer


@register(
    "passthrough",
    description="Control: baseline forward through the candidate slot. Expect 1.00x, zero error.",
    tags=("control",),
)
def build(*, official: Any, config: Any, case: Any) -> Any:
    return _passthrough_class(official)(config)
