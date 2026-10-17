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
    assert engine.feature_names("rt") is None
    engine.evict("rt")
    assert not any(w.has("rt") for w in built.worker_shards)


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
    reports = [w.report() for w in built.worker_shards]
    assert all(r["gpu_memory"] is None for r in reports)  # "cuda:N" fake labels
    assert all(r["spill"] is not None for r in reports)


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
