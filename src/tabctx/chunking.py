"""Split a large test set into sub-batches processed sequentially against one
cached training context.

This mirrors TabPFN's own TABPFN_MAX_BATCHED_TEST_ROWS pattern (test rows are
conditionally independent given the cached context, so chunking is exact,
not an approximation) but picks the chunk size from the memory estimator
instead of a flat constant -- a context with n_train=90,000 needs a much
smaller test chunk than one with n_train=500, and a flat constant can't
express that.

This is NOT the CRUMB-style cross-request/heterogeneous-shape batching
problem (packing many different requests' tables into one GPU call) --
that's explicitly out of v1 scope. This only chunks one request's own test
set against its own single cached context.
"""

from __future__ import annotations

from tabctx.memory.estimator import MemoryEstimator

_FALLBACK_CHUNK_ROWS = 2_000


def choose_chunk_rows(
    estimator: MemoryEstimator,
    n_train: int,
    n_features: int,
    remaining_budget_bytes: int,
    min_chunk_rows: int = 1,
) -> int:
    """Largest test-row chunk size such that estimate_bytes(n_train,
    chunk_rows, n_features) fits within remaining_budget_bytes, found by
    halving from a fallback starting point. Falls back to
    _FALLBACK_CHUNK_ROWS (clamped to at least min_chunk_rows) if even the
    smallest chunk doesn't fit -- the caller (engine) is responsible for
    surfacing that as an admission failure rather than looping forever.
    """
    candidate = _FALLBACK_CHUNK_ROWS
    while candidate > min_chunk_rows:
        if (
            estimator.estimate_bytes(n_train, candidate, n_features)
            <= remaining_budget_bytes
        ):
            return candidate
        candidate //= 2
    return min_chunk_rows


def split_rows(X_test: list, chunk_rows: int) -> list[list]:
    """Split X_test into consecutive chunks of at most chunk_rows rows each,
    preserving order (so callers can reassemble predictions positionally)."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    return [X_test[i : i + chunk_rows] for i in range(0, len(X_test), chunk_rows)]
