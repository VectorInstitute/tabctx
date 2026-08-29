"""Same-context request coalescing for predict().

The concurrency benchmark (benchmarks/baselines/v0.5.0.json) shows
per-request latency for a small table (~200ms) dominated by fixed
per-call overhead, not GPU saturation -- so when several concurrent
requests target the SAME cached context, packing their test rows into
one backend call and splitting the results amortizes that overhead
almost for free. This is the "straightforward and worth doing
regardless" half of ROADMAP.md's Priority 3; batching across DIFFERENT
contexts is a separate, model-dependent question this module does not
attempt.

Safety argument (why this can't blow the memory budget): the engine
serializes all GPU work behind its cache lock, and engine.predict()
already chunks oversized test sets against the memory ceiling
(chunking.py). Coalescing therefore never increases the number of
concurrent GPU calls (still exactly one) -- it only reduces how many
serialized calls the same amount of work needs. The admission ceiling
derivation ("exactly one in-flight backend call") still holds.

Failure isolation: one malformed request must not poison the others in
its batch. If a *batched* call fails input validation, every member is
retried individually (unbatched), so exactly the guilty request(s) get
their 422 and innocent ones still succeed. Non-input errors (dataset
evicted, backend compute failure) affect the batch's shared context and
are propagated to all members -- they would have failed individually too.

Threading model: callers block in their own thread (the serving layer
runs sync handlers in a threadpool). The first arrival for a group key
becomes the leader, sleeps a tiny coalescing window while followers
join, then executes one engine.predict() for everyone. window_s=0
disables coalescing entirely (every request executes directly).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from tabctx.engine import TabctxEngine
from tabctx.errors import InvalidInputError
from tabctx.types import ArrayLike, PredictOutcome


@dataclass
class _Member:
    X_test: ArrayLike
    outcome: PredictOutcome | None = None
    error: Exception | None = None


@dataclass
class _Batch:
    members: list[_Member] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    closed: bool = False  # leader has taken the batch; no more joiners


class CoalescingPredictor:
    """Wraps TabctxEngine.predict with same-context request coalescing.

    Groups by (dataset_id, return_proba): rows from concurrent requests
    against one context are concatenated into a single engine.predict()
    call and the results split back per request. Correctness contract:
    every caller receives exactly the predictions (and probabilities,
    when requested) for its own rows, in its own row order.
    """

    def __init__(self, engine: TabctxEngine, window_s: float = 0.005) -> None:
        if window_s < 0:
            raise ValueError(f"window_s must be >= 0, got {window_s}")
        self._engine = engine
        self._window_s = window_s
        self._lock = threading.Lock()
        self._batches: dict[tuple[str, bool], _Batch] = {}
        # Observability: how many engine calls were saved by coalescing.
        self.batched_requests = 0
        self.engine_calls = 0

    def predict(
        self, dataset_id: str, X_test: ArrayLike, return_proba: bool = False
    ) -> PredictOutcome:
        if self._window_s == 0:
            with self._lock:
                self.engine_calls += 1
            return self._engine.predict(dataset_id, X_test, return_proba=return_proba)

        key = (dataset_id, return_proba)
        member = _Member(X_test=X_test)
        with self._lock:
            batch = self._batches.get(key)
            if batch is not None and not batch.closed:
                batch.members.append(member)
                is_leader = False
            else:
                batch = _Batch(members=[member])
                self._batches[key] = batch
                is_leader = True

        if is_leader:
            time.sleep(self._window_s)
            with self._lock:
                batch.closed = True
                if self._batches.get(key) is batch:
                    del self._batches[key]
                self.batched_requests += len(batch.members)
            try:
                self._execute(key, batch)
            finally:
                # Followers must never hang, even on a bug in the
                # execute/split path itself.
                for m in batch.members:
                    if m.outcome is None and m.error is None:
                        m.error = RuntimeError(
                            "batch leader failed before producing a result"
                        )
                batch.done.set()
        else:
            batch.done.wait()

        if member.error is not None:
            raise member.error
        assert member.outcome is not None
        return member.outcome

    def _execute(self, key: tuple[str, bool], batch: _Batch) -> None:
        dataset_id, return_proba = key
        members = batch.members
        if len(members) == 1:
            self._execute_individually(dataset_id, return_proba, members)
            return

        merged: list = []
        for m in members:
            merged.extend(m.X_test)
        try:
            with self._lock:
                self.engine_calls += 1
            outcome = self._engine.predict(
                dataset_id, merged, return_proba=return_proba
            )
        except InvalidInputError:
            # One member's malformed rows (e.g. wrong feature count) can
            # fail the whole merged call -- retry individually so only
            # the guilty member(s) see the error.
            self._execute_individually(dataset_id, return_proba, members)
            return
        except Exception as e:
            # Context-level failure (evicted, compute error): every
            # member would have hit it individually too.
            for m in members:
                m.error = e
            return

        offset = 0
        for m in members:
            n = len(m.X_test)
            m.outcome = PredictOutcome(
                predictions=outcome.predictions[offset : offset + n],
                probabilities=(
                    outcome.probabilities[offset : offset + n]
                    if return_proba and outcome.probabilities is not None
                    else None
                ),
                classes=outcome.classes if return_proba else None,
            )
            offset += n

    def _execute_individually(
        self, dataset_id: str, return_proba: bool, members: list[_Member]
    ) -> None:
        for m in members:
            try:
                with self._lock:
                    self.engine_calls += 1
                m.outcome = self._engine.predict(
                    dataset_id, m.X_test, return_proba=return_proba
                )
            except Exception as e:  # noqa: BLE001 -- delivered to the caller
                m.error = e
