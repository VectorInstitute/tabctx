"""torch.compile for TabICL's attention blocks (#32, #36).

The fit profile (#31) put layer norm at ~21% of the GPU half: memory-bound
passes over [ensembles, features, rows, 128] activations, each beside a
residual add or the GELU MLP that streams the same tensor again. Compiling
each attention block's forward lets Inductor fuse those passes around the
attention kernel (which stays one kernel). Measured on an A100 at tabicl's
defaults (fp16 autocast above 1,024 rows), benchmarks/bench_compile.py:

  table       GPU half eager -> compiled   peak GiB
  3000x20     0.18 -> 0.12 s               2.0 -> 1.7
  10000x50    0.91 -> 0.52 s               9.9 -> 7.3
  40000x100   7.65 -> 4.30 s               33.1 -> 27.0
  20000x200   6.51 -> 2.92 s               31.1 -> 23.4

Accuracy and log loss are unchanged (within 0.0004 log loss on held-out
rows). In fp32 the compiled blocks match eager to 3e-5 in probability;
under fp16 autocast they differ by up to ~0.01, the same order as fp16
itself differs from fp32 (~0.1), from where rounding happens.

Shapes. Every block is compiled with dynamic shapes, so a new table
doesn't recompile -- once tabicl's ssmax log(n) is computed symbolically
inside compiled code (see tabicl.install_cached_logn): math.log on a
Python int specialized every graph to one row count, and each new table
recompiled for ~11 s. What remains is one graph per call family (stage,
cached vs uncached, size-1 dimensions, ...). Only tabicl's fp16 autocast
path (tables of 1,024+ rows) is compiled: smaller tables run in fp32 in
~0.1 s and would double the graphs. ~50 graphs compile in the backend's
warmup at replica start (~130 s per worker on an A100, workers in
parallel; Inductor's cache barely shortens it, tracing dominates). Calls
warmup doesn't cover still compile inside a request, once per worker:
tables of ~30k rows hand the blocks differently strided inputs (1-7 more
graphs, 5-25 s on an A100 for the first such fit, 10,000 x 500 one more),
and other unusual shapes may too.

Failure. Compilation errors fall back to eager (dynamo's suppress_errors),
so a missing compiler toolchain costs speed, not availability.

Settings. Both dynamo settings these blocks need (suppress_errors, and a
recompile limit above the ~50 graphs) are applied around each compiled
call, not set at install: dynamo's config overrides are per thread (a
ContextVar), and calls run on other threads than the one that installs
the blocks -- engine and worker threads, asyncio.to_thread -- which saw
dynamo's defaults: a limit of 8, so serving workers stopped compiling and
ran shapes warmup missed eagerly, and compile errors raised in tenants'
fits instead of falling back. Scoped to these blocks, they also don't
change other torch.compile users in the process.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Graphs per block, per device (stage x fp16/fp32 x cached/uncached x ...,
# ~50 measured; guards include the device, so an in-process engine over
# several GPUs needs that many per GPU): well above dynamo's default of 8,
# past which it silently runs the rest eagerly.
_RECOMPILE_LIMIT_PER_DEVICE = 256


def install_compiled_blocks(mode: str = "default") -> bool:
    """Compile tabicl's MultiheadAttentionBlock.forward for every instance
    (nn-module parameters are graph inputs, so instances share graphs).
    Idempotent; returns False (and changes nothing) if tabicl or
    torch.compile isn't available."""
    try:
        import torch
        import torch._dynamo
        from tabicl._model.layers import MultiheadAttentionBlock
    except ImportError:
        return False
    forward = MultiheadAttentionBlock.forward
    if getattr(forward, "_tabctx_compiled", False):
        return True
    compiled = with_compile_settings(
        torch.compile(forward, dynamic=True, mode=mode),
        devices=max(1, torch.cuda.device_count()),
    )

    # Only under autocast: tabicl turns fp16 autocast on for tables of
    # 1,024+ rows, where the fused kernels pay off; smaller tables (fp32,
    # ~0.1 s fits) gain little and would double the graphs to compile.
    def dispatch(self: Any, *args: Any, **kwargs: Any) -> Any:
        if torch.is_autocast_enabled("cuda"):
            return compiled(self, *args, **kwargs)
        return forward(self, *args, **kwargs)

    dispatch._tabctx_compiled = True  # type: ignore[attr-defined]
    dispatch._tabctx_eager = forward  # type: ignore[attr-defined]
    MultiheadAttentionBlock.forward = dispatch
    logger.info("tabicl attention blocks compiled (mode=%s)", mode)
    return True


def with_compile_settings(fn: Any, devices: int) -> Any:
    """fn, run under the dynamo settings these blocks need, on whichever
    thread or context calls it (see the module docstring): compile errors
    fall back to eager (warn_if_not_compiled reports it), and the recompile
    limit covers every graph on every device."""
    import torch._dynamo

    config: Any = torch._dynamo.config
    limit = max(config.recompile_limit, _RECOMPILE_LIMIT_PER_DEVICE * devices)
    return config.patch(recompile_limit=limit, suppress_errors=True)(fn)


def warn_if_not_compiled() -> bool:
    """After warmup: warn if no graph was compiled (compile errors fall back
    to eager silently, losing the speedup) or a block hit dynamo's limit.
    Returns whether compilation looks healthy."""
    from torch._dynamo.utils import counters

    graphs = counters["stats"]["unique_graphs"]
    if graphs == 0:
        logger.warning(
            "tabicl attention blocks did not compile (running eagerly): check "
            "the worker log for dynamo errors, or set TABCTX_COMPILE=0"
        )
        return False
    logger.info("tabicl attention blocks: %d graphs compiled", graphs)
    return True
