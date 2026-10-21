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
parallel; Inductor's cache barely shortens it, tracing dominates).

After warmup nothing compiles inside a request (stop_compiling_new_graphs):
a call no warmed graph fits runs eagerly. Families warmup can't foresee
remain -- tables of ~20k+ rows hand the blocks differently strided inputs,
regression predicts of 1,024+ rows take the fp16 path, and calls
overlapping on one worker produce variants -- and compiling them in
requests cost 45-200 s fits on prod's A40s. TABCTX_COMPILE_AFTER_WARMUP=1
restores compiling them.

Failure. Compilation errors fall back to eager (dynamo's suppress_errors),
so a missing compiler toolchain costs speed, not availability.

Settings. The dynamo settings these blocks need (suppress_errors, and
recompile limits above the ~50 graphs) are applied around each compiled
call as well as at install: dynamo's config overrides are per thread (a
ContextVar, as of torch 2.13), and calls run on other threads than the one
that installs the blocks -- engine and worker threads, asyncio.to_thread
-- which saw dynamo's defaults: a limit of 8. Past it dynamo compiles no
new graph of the block for the life of the process (existing ones still
run), so serving workers ran every shape warmup missed eagerly, and compile
errors raised in tenants' fits instead of falling back. Set at install too,
so a torch whose config is process-wide gets them everywhere, and every
call's patch saves and restores the same values there.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Graphs per block, per device (stage x fp16/fp32 x cached/uncached x ...,
# ~50 measured; guards include the device, so an in-process engine over
# several GPUs needs that many per GPU): well above dynamo's default of 8,
# past which it silently runs new shapes eagerly. Dynamo also caps every
# code object's graphs at accumulated_recompile_limit (256 by default), so
# that is raised to match.
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
    """fn, run under compile_settings(devices) on whichever thread or
    context calls it; also set for the installing thread (or the whole
    process, where dynamo's config isn't per thread).
    """
    import torch._dynamo

    config: Any = torch._dynamo.config
    settings = compile_settings(devices)
    for name, value in settings.items():
        setattr(config, name, value)
    return config.patch(settings)(fn)


def compile_settings(devices: int) -> dict[str, Any]:
    """The dynamo settings the compiled blocks run under (see the module
    docstring): compile errors fall back to eager (warn_if_not_compiled
    reports it), and both recompile limits cover every graph on every
    device."""
    import torch._dynamo

    config: Any = torch._dynamo.config
    limit = _RECOMPILE_LIMIT_PER_DEVICE * devices
    return {
        "recompile_limit": max(config.recompile_limit, limit),
        "accumulated_recompile_limit": max(config.accumulated_recompile_limit, limit),
        "suppress_errors": True,
    }


def stop_compiling_new_graphs() -> bool:
    """After warmup: run calls that no compiled graph fits eagerly instead
    of compiling one inside the request (torch.compiler stance
    "eager_on_recompile", process-wide). A compile costs 5-25 s per graph
    family on an A100 and ~4x that on an A40, and a request can need
    several (larger tables, other predict sizes, overlapping calls); on
    prod that was 45-200 s fits under light load, holding the device so
    other models' calls waited too (tabctx 0.14 timings_ms). Eager for a
    family warmup missed costs ~1.5-2x its compiled time instead. Returns
    False on a torch without stances (compiling continues as before)."""
    try:
        import torch
    except ImportError:
        return False
    set_stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if set_stance is None:
        return False
    set_stance("eager_on_recompile")
    logger.info("tabicl: warmup done; uncovered call shapes run eagerly from now on")
    return True


def warn_if_not_compiled() -> bool:
    """After warmup: warn if no graph was compiled (compile errors fall back
    to eager silently, losing the speedup). Returns whether compilation
    looks healthy. A recompile limit hit later shows only in the log
    (dynamo's 'hit config.recompile_limit' warning)."""
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
