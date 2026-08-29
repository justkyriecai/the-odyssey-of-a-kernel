"""Candidate implementations, by name.

A candidate is a factory that returns a module the official script can use as
`UserOptimizedTransformer`. The contract is the script's, not ours:

  1. `forward(x, valid_token_mask=None) -> [batch, seq_len, d_model]`.
  2. Parameter names compatible with `BaselineTransformer`, because
     `copy_model_weights` loads the baseline state dict with `strict=True`.
     Subclass `BaselineTransformer` and override `forward`; derive any fused
     weights lazily on the first call, and register extra buffers with
     `persistent=False` so they stay out of the state dict.

`requires` is a hard gate, not a hint -- the harness skips a candidate whose
requirements a case does not meet, and says so, rather than reporting a number
from a silent fallback path.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

Factory = Callable[..., Any]

# Requirement tokens understood by `Candidate.unmet`.
_KNOWN_REQUIREMENTS = {"cuda", "half", "no-padding", "no-causal", "cuda-graphs"}


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Factory
    description: str = ""
    requires: tuple[str, ...] = ()
    strict_weight_copy: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def unmet(self, case: Any, device_type: str) -> list[str]:
        """Requirements this (case, device) pair fails. Empty means runnable."""
        problems = []
        for req in self.requires:
            if req in ("cuda", "cuda-graphs") and device_type != "cuda":
                problems.append(f"{req}: needs a CUDA device, got {device_type}")
            elif req == "half" and case.dtype == "float32":
                problems.append("half: needs float16 or bfloat16")
            elif req == "no-padding" and case.padding_ratio > 0:
                problems.append("no-padding: case sets padding_ratio > 0")
            elif req == "no-causal" and case.causal:
                problems.append("no-causal: case sets causal=True")
        return problems

    def build(self, official: Any, config: Any, case: Any) -> Any:
        return self.factory(official=official, config=config, case=case)


_REGISTRY: dict[str, Candidate] = {}
_KERNELS_LOADED = False


def register(
    name: str,
    *,
    description: str = "",
    requires: Iterable[str] = (),
    strict_weight_copy: bool = True,
    tags: Iterable[str] = (),
) -> Callable[[Factory], Factory]:
    requires = tuple(requires)
    unknown = set(requires) - _KNOWN_REQUIREMENTS
    if unknown:
        raise ValueError(
            f"{name}: unknown requirement(s) {sorted(unknown)}; "
            f"known: {sorted(_KNOWN_REQUIREMENTS)}"
        )

    def decorator(factory: Factory) -> Factory:
        if name in _REGISTRY:
            raise KeyError(f"candidate {name!r} is already registered")
        _REGISTRY[name] = Candidate(
            name=name,
            factory=factory,
            description=description or (factory.__doc__ or "").strip().split("\n")[0],
            requires=requires,
            strict_weight_copy=strict_weight_copy,
            tags=tuple(tags),
        )
        return factory

    return decorator


def _load_kernels() -> None:
    global _KERNELS_LOADED
    if _KERNELS_LOADED:
        return
    _KERNELS_LOADED = True
    importlib.import_module("kernels")


def get(name: str) -> Candidate:
    _load_kernels()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown candidate {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available() -> list[Candidate]:
    _load_kernels()
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def names() -> list[str]:
    _load_kernels()
    return sorted(_REGISTRY)


def resolve(patterns: Optional[Iterable[str]]) -> list[Candidate]:
    """Resolve names or `all` into candidates, preserving the given order."""
    _load_kernels()
    if not patterns:
        return available()
    out: list[Candidate] = []
    for pattern in patterns:
        if pattern == "all":
            out.extend(c for c in available() if c not in out)
        else:
            candidate = get(pattern)
            if candidate not in out:
                out.append(candidate)
    return out
