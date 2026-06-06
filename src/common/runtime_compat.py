from __future__ import annotations

import sys
import types
from typing import Any


def ensure_triton_compat() -> None:
    """Provide a minimal triton.ops compatibility shim for bitsandbytes.

    Some Colab environments ship Triton 3.x, where `triton.ops` was removed.
    Older bitsandbytes releases still import:

    - triton.ops
    - triton.ops.matmul_perf_model

    This shim creates those modules with lightweight placeholder functions so
    imports succeed. It does not change model math; it only satisfies optional
    autotuning helpers bitsandbytes imports at module load time.
    """
    if "triton.ops.matmul_perf_model" in sys.modules:
        return

    try:
        import triton  # type: ignore
    except ImportError:
        return

    ops_module = sys.modules.get("triton.ops")
    if ops_module is None:
        ops_module = types.ModuleType("triton.ops")
        ops_module.__path__ = []  # Mark as package for submodule imports.
        sys.modules["triton.ops"] = ops_module
    elif not hasattr(ops_module, "__path__"):
        ops_module.__path__ = []

    def early_config_prune(configs: Any, *args: Any, **kwargs: Any) -> Any:
        return configs

    def estimate_matmul_time(*args: Any, **kwargs: Any) -> float:
        return 0.0

    perf_module = sys.modules.get("triton.ops.matmul_perf_model")
    if perf_module is None:
        perf_module = types.ModuleType("triton.ops.matmul_perf_model")
        perf_module.early_config_prune = early_config_prune
        perf_module.estimate_matmul_time = estimate_matmul_time
        perf_module.__all__ = ["early_config_prune", "estimate_matmul_time"]
        sys.modules["triton.ops.matmul_perf_model"] = perf_module

    ops_module.matmul_perf_model = perf_module
    if not hasattr(triton, "ops"):
        triton.ops = ops_module  # type: ignore[attr-defined]
