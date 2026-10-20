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

# One call's worth of test rows at most: the engine's LIGHT_PREDICT_ROWS, so
# a light predict (one that may run beside a fit) is a single call.
MAX_CHUNK_ROWS = 2_048


def choose_chunk_rows(
    estimator: MemoryEstimator,
    n_train: int,
    n_features: int,
    remaining_budget_bytes: int,
    min_chunk_rows: int = 1,
) -> int:
    """Largest test-row chunk, halving from MAX_CHUNK_ROWS down to
    min_chunk_rows, whose predict fits remaining_budget_bytes.

    What a call of min_chunk_rows costs is, beyond those rows, the call's
    fixed cost, which no chunk size reduces. When that alone exceeds the
    budget, smaller chunks would only multiply the calls (a 100-row predict
    became 100 one-row calls, #47), so the fixed cost is taken as given and
    only the rows' cost above it is sized against the budget: a chunk never
    adds more than the budget on top of what the smallest call needs. A
    chunk that still OOMs is halved by the engine.
    """
    if min_chunk_rows < 1:
        raise ValueError("min_chunk_rows must be at least 1")
    floor_bytes = estimator.estimate_bytes(n_train, min_chunk_rows, n_features)
    allowance = remaining_budget_bytes
    if floor_bytes > remaining_budget_bytes:
        allowance = floor_bytes + remaining_budget_bytes
    candidate = MAX_CHUNK_ROWS
    while candidate > min_chunk_rows:
        if estimator.estimate_bytes(n_train, candidate, n_features) <= allowance:
            return candidate
        candidate //= 2
    return min_chunk_rows


def split_rows(X_test: list, chunk_rows: int) -> list[list]:
    """Split X_test into consecutive chunks of at most chunk_rows rows each,
    preserving order (so callers can reassemble predictions positionally)."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    return [X_test[i : i + chunk_rows] for i in range(0, len(X_test), chunk_rows)]
