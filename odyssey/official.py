"""Load and drive the organizer's benchmark script.

Rule of this file: the script is read-only and authoritative. We import its
symbols; we never fork them. When a number in `runs/benchmark.csv` disagrees
with what the organizers would see, that is a bug in the wrapper, not a
difference of opinion about the metric -- which is why `run_main()` exists to
run the script's own `main()` end to end as the final gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .paths import OFFICIAL_SCRIPT

_MODULE_NAME = "techjam_official_benchmark"
_cache: dict[str, types.ModuleType] = {}


def script_path(explicit: Optional[str | Path] = None) -> Path:
    """Resolve the script: explicit argument, then $TECHJAM_BENCHMARK, then vendored."""
    raw = explicit or os.environ.get("TECHJAM_BENCHMARK")
    path = Path(raw).expanduser().resolve() if raw else OFFICIAL_SCRIPT
    if not path.exists():
        raise FileNotFoundError(
            f"Official benchmark script not found at {path}.\n"
            "Vendor it into bench/official/ or set TECHJAM_BENCHMARK."
        )
    return path


def script_md5(explicit: Optional[str | Path] = None) -> str:
    return hashlib.md5(script_path(explicit).read_bytes()).hexdigest()


def load(explicit: Optional[str | Path] = None) -> types.ModuleType:
    """Import the script as a module. Cached per resolved path."""
    path = script_path(explicit)
    key = str(path)
    if key in _cache:
        return _cache[key]

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registering before exec keeps dataclasses and pickling well-behaved.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _cache[key] = module
    return module


def run_main(
    build_optimized: Optional[Callable[[types.ModuleType], type]] = None,
    argv: Optional[Sequence[str]] = None,
    explicit: Optional[str | Path] = None,
) -> int:
    """Run the script's own `main()`, optionally with our class patched in.

    This is the gate, not the inner loop. It prints exactly what the organizers
    would see and returns the script's own exit code (0 pass, 2 accuracy fail).

    `build_optimized` receives the loaded module and returns the class to use as
    `UserOptimizedTransformer`. The substitution is undone on the way out so the
    module stays clean for anything else in the process.
    """
    module = load(explicit)
    original = module.UserOptimizedTransformer
    saved_argv = sys.argv

    try:
        if build_optimized is not None:
            module.UserOptimizedTransformer = build_optimized(module)
        sys.argv = [str(script_path(explicit)), *(argv or [])]
        return int(module.main())
    finally:
        module.UserOptimizedTransformer = original
        sys.argv = saved_argv


def config_from(module: types.ModuleType, case: Any) -> Any:
    """Build the script's own `TransformerConfig` from one of our cases."""
    config = module.TransformerConfig(
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        d_model=case.d_model,
        num_heads=case.num_heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.num_layers,
        causal=case.causal,
    )
    config.validate()
    return config
