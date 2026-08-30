"""CUDA Graph capture around another candidate.

The default case is 1024 tokens through six layers: roughly ninety kernel
launches for an amount of arithmetic a modern GPU finishes in tens of
microseconds. When launch overhead is the bottleneck, no amount of matmul
tuning helps and replaying one captured graph does. This candidate exists to
find out which regime a given shape is in -- and the answer is a per-shape
answer, which is why `dispatch` chooses between them rather than picking one.

Capture protocol, in the order it has to happen:

1. Allocate static input buffers and copy the first real input into them.
2. Warm up on a side stream. This also forces the wrapped candidate to build
   any lazy state (the fused QKV weights) *outside* the capture, since
   allocating inside a capture is what turns a graph into a crash.
3. Capture one forward.
4. On later calls, copy new data into the static buffers and replay.

The capture is keyed on (shape, dtype, device, whether a mask is present). A
new key re-captures. Values inside the mask may change freely between replays
-- the mask only feeds data-dependent arithmetic, never control flow.

Off CUDA there is nothing to capture, and the wrapped candidate runs as-is: a
CPU number for `graph-*` is the number for the candidate underneath, and says
nothing about graphs.

The output is cloned out of the static buffer by default. Returning the buffer
itself saves a copy of one activation tensor, but it aliases: two consecutive
calls would hand back the same memory, and this contest fails a candidate on a
single bad element. Set `ODYSSEY_GRAPH_ALIAS_OUTPUT=1` to measure the
difference; do not ship it without proving the caller never holds two outputs.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

import torch

from .v1_fused_attention import FP32_SOFTMAX, SDPA, fused_attention_class


def _alias_output() -> bool:
    return os.environ.get("ODYSSEY_GRAPH_ALIAS_OUTPUT", "0") == "1"


@functools.lru_cache(maxsize=None)
def cuda_graph_class(base_class: type) -> type:
    class CUDAGraphTransformer(base_class):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self._graph: Optional[torch.cuda.CUDAGraph] = None
            self._graph_key: Optional[tuple] = None
            self._static_input: Optional[torch.Tensor] = None
            self._static_mask: Optional[torch.Tensor] = None
            self._static_output: Optional[torch.Tensor] = None

        def _capture(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor], key: tuple
        ) -> None:
            self._static_input = x.detach().clone()
            self._static_mask = (
                None if valid_token_mask is None else valid_token_mask.detach().clone()
            )

            # Warm up on a side stream. Three iterations is the documented
            # minimum and is also what forces lazy weight fusion to happen here
            # rather than inside the capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    super().forward(self._static_input, self._static_mask)
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._static_output = super().forward(
                    self._static_input, self._static_mask
                )
            self._graph = graph
            self._graph_key = key

        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if x.device.type != "cuda":
                return super().forward(x, valid_token_mask)

            key = (
                tuple(x.shape),
                x.dtype,
                str(x.device),
                valid_token_mask is None,
            )
            if self._graph_key != key:
                self._capture(x, valid_token_mask, key)

            assert self._static_input is not None and self._graph is not None
            self._static_input.copy_(x)
            if self._static_mask is not None and valid_token_mask is not None:
                self._static_mask.copy_(valid_token_mask)
            self._graph.replay()

            assert self._static_output is not None
            return self._static_output if _alias_output() else self._static_output.clone()

    return CUDAGraphTransformer


def build_safe(official: Any) -> type:
    return cuda_graph_class(fused_attention_class(official, FP32_SOFTMAX))


def build_sdpa(official: Any) -> type:
    return cuda_graph_class(fused_attention_class(official, SDPA))


CANDIDATES = {
    "graph-safe": (
        "fused-safe captured into a CUDA Graph: one replay instead of ~90 launches. CUDA only.",
        build_safe,
    ),
    "graph-sdpa": ("fused-sdpa captured into a CUDA Graph. CUDA only.", build_sdpa),
}
