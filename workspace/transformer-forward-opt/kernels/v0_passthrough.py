"""The identity candidate.

Not an optimization -- a control. It runs the baseline's own forward through
the candidate slot, so it must report bit-identical output and a speedup within
noise of 1.00x. When it does not, the measurement is lying and no other number
in `runs/benchmark.csv` can be trusted until that is explained.
"""

from __future__ import annotations

import functools
from typing import Any


@functools.lru_cache(maxsize=None)
def passthrough_class(official: Any) -> type:
    class PassthroughTransformer(official.BaselineTransformer):
        """Baseline forward, unchanged."""

    return PassthroughTransformer


CANDIDATES = {
    "passthrough": (
        "Control: baseline forward through the candidate slot. Expect 1.00x, zero error.",
        passthrough_class,
    ),
}
