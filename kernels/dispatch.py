"""The shipping layer: never slower than what it replaces.

Every other candidate is a bet. This one is the policy that decides when to
take the bet. For a given geometry it uses the specialization a calibration run
proved both correct and faster; for everything else it runs the baseline path.
The floor is therefore the baseline, not the best case -- an unfamiliar shape
degrades to "no worse", never to "wrong" and never to "slower".

Two rules make that claim true rather than aspirational:

**Correctness is checked before speed.** `odyssey.calibrate` only admits a
candidate that passed the zero-bad-element check on every case in the group.

**Padding is not a dispatch axis.** The official generator always supplies a
mask -- all-ones when `--padding-ratio 0` -- so a running model cannot tell
dense from padded without a device sync, which would also break graph capture.
Rather than guess, a candidate must clear both the dense and the padded case
for a geometry before it is allowed to serve either.

Without a table this class is exactly the baseline, which is the correct
behaviour on day zero and the reason it is safe to leave wired in.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Optional

import torch

from odyssey import official as official_mod
from odyssey import registry
from odyssey.paths import DISPATCH_TABLE
from odyssey.registry import register

# Never dispatch to these: `passthrough` is a control and `dispatch` is us.
_NOT_DISPATCHABLE = frozenset({"passthrough", "dispatch"})


def load_table(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    path = Path(path or DISPATCH_TABLE)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return raw.get("entries", raw)


def runtime_dispatch_key(config: Any, x: torch.Tensor) -> str:
    batch, seq_len, _ = x.shape
    dtype = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(x.dtype, str(x.dtype))
    return (
        f"b{batch}_s{seq_len}_d{config.d_model}_h{config.num_heads}"
        f"_f{config.ffn_dim}_l{config.num_layers}"
        f"_{'causal' if config.causal else 'full'}_{dtype}"
    )


@functools.lru_cache(maxsize=None)
def dispatch_class(official: Any) -> type:
    class DispatchTransformer(official.BaselineTransformer):
        def __init__(self, config: Any, table: Optional[dict] = None) -> None:
            super().__init__(config)
            # Plain dict, so delegates never enter this module's state_dict and
            # the strict weight copy keeps working.
            self._delegates: dict[str, Any] = {}
            self._table = load_table() if table is None else table
            self._resolved: dict[str, Optional[str]] = {}

        # -- delegates ----------------------------------------------------
        def _delegate(self, name: str) -> Any:
            cached = self._delegates.get(name)
            if cached is not None:
                return cached

            candidate = registry.get(name)
            module = candidate.build(
                official=official_mod.load(), config=self.config, case=None
            )
            # Share this model's parameters by reference. The delegate's own
            # freshly-allocated layers are discarded here; that costs one CPU
            # allocation the first time a geometry is served, and nothing after.
            module.layers = self.layers
            module.final_norm = self.final_norm
            module.eval()
            self._delegates[name] = module
            return module

        def _choose(self, key: str) -> Optional[str]:
            if key in self._resolved:
                return self._resolved[key]
            entry = self._table.get(key)
            name = entry.get("candidate") if isinstance(entry, dict) else entry
            if name in _NOT_DISPATCHABLE:
                name = None
            self._resolved[key] = name
            return name

        # -- forward ------------------------------------------------------
        def forward(
            self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            name = self._choose(runtime_dispatch_key(self.config, x))
            if name is None:
                return super().forward(x, valid_token_mask)
            return self._delegate(name)(x, valid_token_mask)

    return DispatchTransformer


@register(
    "dispatch",
    description="Per-geometry dispatch to a calibrated candidate; baseline fallback otherwise.",
    tags=("ship",),
)
def build(*, official: Any, config: Any, case: Any) -> Any:
    return dispatch_class(official)(config)
