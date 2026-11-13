"""Per-GPU worker processes (serve/workers.py), with real processes and the
fake backend (which never touches a GPU, so 'cuda:N' are just labels)."""

import threading

import pytest

from tabctx.errors import DatasetNotFoundError, InvalidInputError
from tabctx.multidevice import MultiDeviceEngine
from tabctx.serve.factory import ServeSettings, build_engine
from tabctx.serve.workers import ProcessShard, WorkerDiedError


def table(rows: int, label: str = "a", features: int = 2):
    return [[float(i)] * features for i in range(rows)], [label] * rows


def _settings(tmp_path=None, **kw) -> ServeSettings:
    return ServeSettings(
        backends=("fake",),
        devices=("cuda:0", "cuda:1"),
        spill_dir=str(tmp_path) if tmp_path else None,
        **kw,
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    b = build_engine(_settings(tmp_path_factory.mktemp("spill")))
    yield b
    b.close()


def test_multi_device_defaults_to_worker_processes(built):
    assert isinstance(built.engine, MultiDeviceEngine)
    assert [type(w) for w in built.worker_shards] == [ProcessShard, ProcessShard]
    assert all(w.alive for w in built.worker_shards)
    assert built.backend_names == ["fake"] and built.default == "fake"


def test_threads_when_disabled():
    b = build_engine(_settings(worker_processes=False))
    assert b.worker_shards == ()
    assert isinstance(b.engine, MultiDeviceEngine)


def test_fit_predict_round_trip_and_residency_mirror(built):
    engine = built.engine
    engine.fit(*table(10, label="x"), dataset_id="rt")
    holders = [w.device for w in built.worker_shards if w.holds("rt")]
    assert len(holders) == 1  # mirrored before fit() returned
    assert engine.predict("rt", [[0.0, 0.0]]).predictions == ["x"]
    assert engine.schema("rt") is None
    (info,) = [i for i in engine.contexts() if i.dataset_id == "rt"]
    assert info.device == holders[0] and info.location == "resident"
    assert engine.evict("rt") is True
    assert engine.evict("rt") is False
    assert not any(w.has("rt") for w in built.worker_shards)


def test_quantiles_cross_the_process_boundary(built):
    engine = built.engine
    engine.fit(
        [[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0], task="regression", dataset_id="q"
    )
    try:
        out = engine.predict("q", [[0.0], [1.0]], quantiles=[0.9, 0.1])
        # The fake's quantiles: a normal around the mean, in the order asked.
        assert len(out.quantiles) == 2
        assert out.quantiles[0][0] > out.quantiles[0][1]
    finally:
        engine.evict("q")


def test_a_restore_in_a_worker_is_counted_in_the_serving_process(tmp_path):
    """A predict's restore from the disk tier happens in the worker; its
    count must reach the serving process's timings, which is what a
    predict response's cache_outcome is read from."""
    from tabctx import timing

    one = build_engine(
        ServeSettings(
            backends=("fake",), devices=("cuda:0",), spill_dir=str(tmp_path / "cuda_0")
        )
    )
    one.engine.fit(*table(10, label="x"), dataset_id="r")
    assert one.engine._cache.evict_one() is not None  # to cuda_0/ on disk
    one.close()

    workers = build_engine(_settings(tmp_path))
    try:
        assert workers.worker_shards  # really in worker processes
        for expected in (1, 0):  # restored, then resident
            with timing.collect() as timings:
                assert workers.engine.predict("r", [[0.0, 0.0]]).predictions == ["x"]
            assert timings.counts.get("restores", 0) == expected
    finally:
        workers.close()


def test_errors_cross_the_process_boundary_with_their_type(built):
    with pytest.raises(DatasetNotFoundError):
        built.engine.predict("never-fit", [[0.0, 0.0]])
    with pytest.raises(InvalidInputError):
        built.engine.fit(*table(10), dataset_id="b", backend="nope")
    with pytest.raises(InvalidInputError):  # raised inside the worker
        built.engine.fit([[1.0], [2.0]], ["a"], dataset_id="ragged")


def test_concurrent_calls_from_many_threads(built):
    engine = built.engine
    for i in range(4):
        engine.fit(*table(10, label=f"L{i}"), dataset_id=f"c{i}")
    errors: list[Exception] = []

    def hammer(i: int) -> None:
        try:
            for _ in range(20):
                out = engine.predict(f"c{i % 4}", [[0.0, 0.0]])
                assert out.predictions == [f"L{i % 4}"]
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    # Contexts spread over both workers (placement saw both as idle).
    assert all(w.stats_nowait().n_cached_contexts >= 1 for w in built.worker_shards)


def test_stats_and_reports(built):
    built.engine.fit(*table(10), dataset_id="s")
    stats = built.engine.stats()
    assert stats.n_cached_contexts >= 1
    # Workers send their device reports on their own (the first at start).
    for w in built.worker_shards:
        for _ in range(100):
            report, age = w.latest_report()
            if report is not None:
                break
            threading.Event().wait(0.05)
        assert report["gpu_memory"] is None  # "cuda:N" are fake labels here
        assert report["spill"] is not None and age >= 0


def test_a_dead_worker_fails_calls_and_is_skipped():
    b = build_engine(_settings())
    try:
        engine = b.engine
        victim, survivor = b.worker_shards
        victim._proc.kill()
        victim._proc.join(10)
        for _ in range(200):  # the reader thread notices the closed pipe
            if not victim.alive:
                break
            threading.Event().wait(0.01)
        assert not victim.alive and "exited" in (victim.death_reason or "")
        with pytest.raises(WorkerDiedError):
            victim.fit(*table(5), dataset_id="x")
        # The router places new work on the survivor only.
        for i in range(3):
            engine.fit(*table(5), dataset_id=f"after{i}")
        assert all(survivor.holds(f"after{i}") for i in range(3))
        assert engine.predict("after0", [[0.0, 0.0]]).predictions == ["a"]
    finally:
        b.close()


def test_startup_failure_is_raised_in_the_serving_process(tmp_path):
    # The worker can't create its spill subdirectory under a regular file.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(OSError):
        build_engine(_settings(blocker))


def test_errors_that_cannot_round_trip_become_runtime_errors():
    from tabctx.serve.workers import _portable

    class TwoArgs(Exception):
        def __init__(self, a, b):
            super().__init__(f"{a}/{b}")

    portable = _portable(TwoArgs(1, 2))
    assert type(portable) is RuntimeError and "TwoArgs" in str(portable)
    assert isinstance(_portable(DatasetNotFoundError("x")), DatasetNotFoundError)


def test_a_dead_workers_host_reservations_are_returned():
    """A worker OOM-killed mid-fit never runs its release: its parent must
    return the bytes, or every later large call on the node waits forever."""
    import multiprocessing as mp

    from tabctx.memory.host import GiB, HostMemoryGuard

    ctx = mp.get_context("spawn")
    guard = HostMemoryGuard(10 * GiB, mp_context=ctx, owners=2)
    shards = ProcessShard.start_all(
        _settings(),
        ("cuda:0", "cuda:1"),
        gpu_capacity=None,
        host_guard=guard,
        mp_context=ctx,
    )
    try:
        victim = shards[1]
        guard._state[guard._block(1)] = 8 * GiB  # what it held when killed
        guard._state[guard._block(1) + 1] = 1
        victim._proc.kill()
        for _ in range(500):
            if guard.stats()["reserved_bytes"] == 0:
                break
            threading.Event().wait(0.01)
        assert guard.stats()["reserved_bytes"] == 0
        assert guard.stats()["calls_active"] == 0
    finally:
        ProcessShard.close_all(shards)


def test_workers_close_together_within_one_grace_period(monkeypatch):
    import time

    import tabctx.serve.workers as workers

    b = build_engine(
        ServeSettings(backends=("fake",), devices=("cuda:0", "cuda:1", "cuda:2"))
    )
    shards = list(b.worker_shards)
    monkeypatch.setattr(workers, "_CLOSE_GRACE_S", 1.0)
    for s in shards:  # workers that ignore the exit request
        monkeypatch.setattr(s, "_request_exit", lambda: None)
    start = time.monotonic()
    ProcessShard.close_all(shards)
    # One shared deadline (~1 s), not 1 s per worker, then killed.
    assert time.monotonic() - start < 2.5
    assert not any(s._proc.is_alive() for s in shards)


def test_concurrent_waits_all_see_a_dead_worker_exit():
    """The reader thread and close() both wait on a dying worker. Two
    waitpid calls race, and the loser used to report the worker still
    running (so close() kept its host reservations); every wait and
    liveness check now reaps under one lock, so whoever sees the exit also
    sees its exit code."""
    b = build_engine(ServeSettings(backends=("fake",), devices=("cuda:0", "cuda:1")))
    shards = list(b.worker_shards)
    shard = shards[0]
    try:
        barrier = threading.Barrier(8)
        seen: list[bool] = []

        def wait() -> None:
            barrier.wait()
            seen.append(shard._exited(5) and shard._exit_code is not None)

        threads = [threading.Thread(target=wait) for _ in range(7)]
        for t in threads:
            t.start()
        shard._proc.kill()
        barrier.wait()
        for t in threads:
            t.join()
        assert seen == [True] * 7
        assert not shard.alive
    finally:
        ProcessShard.close_all(shards)


def test_a_dead_worker_exits_while_a_grandchild_holds_its_sentinel(monkeypatch):
    """A process the worker forked inherits its sentinel's write end and
    keeps it open after the worker dies: the exit is then read without the
    sentinel, and close() still returns its host reservations."""
    import tabctx.serve.workers as workers

    b = build_engine(ServeSettings(backends=("fake",), devices=("cuda:0", "cuda:1")))
    shards = list(b.worker_shards)
    try:
        monkeypatch.setattr(workers, "wait_for_handles", lambda handles, timeout: [])
        shard = shards[0]
        shard._proc.kill()
        for _ in range(500):
            if shard._exited():
                break
            threading.Event().wait(0.01)
        assert shard._exited()
        assert shard._exit_code is not None
    finally:
        ProcessShard.close_all(shards)


def test_mp_exit_handler_is_registered_before_ours():
    """Our atexit handler must run first (atexit is LIFO), however the
    module was imported: workers.py imports multiprocessing.util itself."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, tabctx.serve.workers; "
            "print('multiprocessing.util' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "True"


def test_workers_share_the_pods_cpus_for_compiling(monkeypatch):
    import tabctx.serve.workers as workers

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(workers, "available_cpus", lambda: 30)
    workers._share_cpus(4)
    import os

    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "7"
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "2")  # deployer's wins
    workers._share_cpus(4)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "2"


def test_reports_keep_arriving_while_fits_fill_the_workers_pool(monkeypatch):
    """/readyz reads the device report each worker sends on its own. With
    more fits queued on a device than the worker has pool threads, reports
    must keep coming, or readiness fails on a healthy, busy pod."""
    import multiprocessing as mp

    import tabctx.serve.reporting as reporting
    import tabctx.serve.workers as workers
    from tabctx.backends.fake import FakeBackend

    monkeypatch.setattr(reporting, "REPORT_INTERVAL_S", 0.05)
    release = threading.Event()
    real_fit = FakeBackend.fit

    def blocked_fit(self, X, y, task):
        release.wait(30)
        return real_fit(self, X, y, task)

    monkeypatch.setattr(FakeBackend, "fit", blocked_fit)
    parent, child = mp.Pipe()
    # The worker's own loop, run on a thread so the patches apply.
    loop = threading.Thread(
        target=workers._worker_main,
        args=(
            child,
            ServeSettings(backends=("fake",), devices=("cuda:0",)),
            "cuda:0",
            1,
            None,
            None,
            0,
        ),
        daemon=True,
    )
    loop.start()
    n_fits = workers._WORKER_THREADS + 4
    try:
        assert isinstance(parent.recv(), workers._Ready)
        for i in range(n_fits):
            parent.send((i + 1, "fit", table(10), {"dataset_id": f"busy{i}"}))
        reports = 0
        while reports < 5:  # every fit is blocked: only reports can arrive
            assert parent.poll(5), "reports stopped while fits were queued"
            req_id, ok, payload, _, _ = parent.recv()
            assert (req_id, ok) == (workers._REPORT_ID, True)
            assert payload["gpu_memory"] is None
            reports += 1
    finally:
        release.set()
        fits = 0
        while fits < n_fits:  # drain, so the loop can shut down
            assert parent.poll(30)
            if parent.recv()[0] != workers._REPORT_ID:
                fits += 1
        parent.send(None)
        loop.join(30)
    assert not loop.is_alive()


def test_tasks_and_survival_times_cross_the_process_boundary(built):
    engine = built.engine
    assert "survival" in engine.backend_tasks["fake"]
    assert engine.resolve_backend_name(None, "causal") == "fake"
    engine.fit(
        [[0.0], [1.0], [2.0]],
        [[1.0, 1], [2.0, 0], [3.0, 1]],
        task="survival",
        dataset_id="surv",
    )
    try:
        out = engine.predict("surv", [[0.0]], times=[1.5, 5.0])
        assert out.survival == [[2 / 3, 0.0]]
    finally:
        engine.evict("surv")


def test_new_outputs_cross_the_process_boundary(built):
    engine = built.engine
    engine.fit([[0.0], [1.0]], [[1.0, 1], [3.0, 0]], task="survival", dataset_id="rm")
    try:
        assert engine.predict("rm", [[0.0]], rmst_horizon=2.0).rmst == [1.5]
    finally:
        engine.evict("rm")
