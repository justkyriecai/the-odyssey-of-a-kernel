"""The shape grid.

A `Case` is everything the official script needs to define one measurement:
the seven `TransformerConfig` fields plus the three test conditions that are
*not* shape but do change the answer -- dtype, padding ratio, input scale.

Shape sets live in `bench/shapes/*.json` so they are data, not code: the Track 3
appendix grid can be pasted in without touching the harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .paths import SHAPES_DIR

DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int = 8
    seq_len: int = 128
    d_model: int = 512
    num_heads: int = 8
    ffn_dim: int = 2048
    num_layers: int = 6
    causal: bool = False
    dtype: str = "float32"
    padding_ratio: float = 0.0
    input_scale: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if not 0.0 <= self.padding_ratio < 1.0:
            raise ValueError("padding_ratio must be in [0, 1)")
        if self.d_model % self.num_heads:
            raise ValueError(
                f"{self.name}: d_model {self.d_model} is not divisible by "
                f"num_heads {self.num_heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def key(self) -> str:
        """Stable identity for dispatch tables and CSV joins. Excludes `name`."""
        return (
            f"b{self.batch_size}_s{self.seq_len}_d{self.d_model}_h{self.num_heads}"
            f"_f{self.ffn_dim}_l{self.num_layers}"
            f"_{'causal' if self.causal else 'full'}_{self.dtype}"
            f"_p{self.padding_ratio:g}"
        )

    @property
    def dispatch_key(self) -> str:
        """Identity a running model can reconstruct from its own inputs.

        Everything in `key` except the padding ratio. A deployed model sees the
        mask but not the ratio it was drawn from, and telling a dense mask from
        a padded one costs a device sync in the hot path -- unacceptable inside
        a captured graph. So padding is not a dispatch axis: a candidate only
        enters the table for a geometry if it passed *every* case sharing this
        key, padded and dense alike. See `odyssey.calibrate`.
        """
        return (
            f"b{self.batch_size}_s{self.seq_len}_d{self.d_model}_h{self.num_heads}"
            f"_f{self.ffn_dim}_l{self.num_layers}"
            f"_{'causal' if self.causal else 'full'}_{self.dtype}"
        )

    def cli_args(self) -> list[str]:
        """The argv that reproduces this case with the untouched official script."""
        args = [
            "--batch-size", str(self.batch_size),
            "--seq-len", str(self.seq_len),
            "--d-model", str(self.d_model),
            "--heads", str(self.num_heads),
            "--ffn-dim", str(self.ffn_dim),
            "--layers", str(self.num_layers),
            "--dtype", self.dtype,
            "--padding-ratio", str(self.padding_ratio),
            "--input-scale", str(self.input_scale),
        ]
        if self.causal:
            args.append("--causal")
        return args

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_(self, **overrides: Any) -> "Case":
        return replace(self, **overrides)


@dataclass(frozen=True)
class ShapeSet:
    name: str
    description: str
    cases: tuple[Case, ...]

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def select(self, names: Optional[Iterable[str]]) -> "ShapeSet":
        if not names:
            return self
        wanted = list(names)
        by_name = {c.name: c for c in self.cases}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise KeyError(
                f"unknown case(s) {missing} in shape set {self.name!r}; "
                f"available: {sorted(by_name)}"
            )
        return ShapeSet(self.name, self.description, tuple(by_name[n] for n in wanted))


_CASE_FIELDS = {f.name for f in fields(Case)}


def load_set(name_or_path: str = "dev") -> ShapeSet:
    path = Path(name_or_path)
    if not path.exists():
        path = SHAPES_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = sorted(p.stem for p in SHAPES_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"No shape set {name_or_path!r}. Available: {available}"
        )

    raw = json.loads(path.read_text())
    defaults = raw.get("defaults", {})
    cases = []
    for entry in raw["cases"]:
        merged = {**defaults, **entry}
        unknown = set(merged) - _CASE_FIELDS
        if unknown:
            raise ValueError(f"{path}: unknown case field(s) {sorted(unknown)}")
        cases.append(Case(**merged))

    if not cases:
        raise ValueError(f"{path}: shape set is empty")
    return ShapeSet(raw.get("name", path.stem), raw.get("description", ""), tuple(cases))


def available_sets() -> list[str]:
    return sorted(p.stem for p in SHAPES_DIR.glob("*.json"))
