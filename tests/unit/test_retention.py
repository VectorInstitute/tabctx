"""Retention limits (#103): a context with an expiry is deleted from every
tier at that time, whether or not anyone uses it again."""

import gc
import json
import threading
import time

import pytest

from tabctx.backends.fake import FakeBackend
from tabctx.cache.manager import CachedContext, ContextCacheManager
from tabctx.cache.spill import DiskSpillStore
from tabctx.engine import TabctxEngine
from tabctx.errors import DatasetNotFoundError
from tabctx.memory import A100_40GB_TABICL_CALIBRATION, PowerLawMemoryEstimator

X = [[float(i), 1.0] for i in range(6)]
Y = ["a", "b"] * 3
PAST, FUTURE = time.time() - 1, time.time() + 3600


def _rewrite_meta(store, dataset_id, edit):
    meta_path, _ = store._paths(dataset_id)
    meta = json.loads(meta_path.read_text())
    edit(meta)
    meta_path.write_text(json.dumps(meta))


def engine_with(spill_dir=None, sweep=None) -> TabctxEngine:
    estimator = PowerLawMemoryEstimator(A100_40GB_TABICL_CALIBRATION)
    spill = DiskSpillStore(spill_dir) if spill_dir else None
    return TabctxEngine(
        backend=FakeBackend(),
        cache=ContextCacheManager(estimator.ceiling_bytes(), spill_store=spill),
        estimator=estimator,
        expiry_sweep_s=sweep,
    )


def ctx(dataset_id: str, expires_at=None) -> CachedContext:
    return CachedContext(
        dataset_id, "fake", "classification", 3, 2, {"p": 1}, 10, expires_at=expires_at
    )


class TestCacheAndSpill:
    def test_expired_ids_resident_and_spilled(self, tmp_path):
        cache = ContextCacheManager(1000, spill_store=DiskSpillStore(tmp_path))
        cache.put(ctx("old-spilled", PAST))
        cache.evict_one()  # the only one: to disk
        cache.put(ctx("old-resident", PAST))
        cache.put(ctx("fresh", FUTURE))
        cache.put(ctx("forever"))
        assert cache.holds("old-resident") and not cache.holds("old-spilled")
        assert cache.expired_ids(time.time()) == ["old-resident", "old-spilled"]
        assert cache.is_expired("old-spilled", time.time())
        assert not cache.is_expired("fresh", time.time())
        assert not cache.is_expired("forever", time.time())

    def test_a_restarted_spill_tier_still_knows_expiries(self, tmp_path):
        DiskSpillStore(tmp_path).spill(ctx("old", PAST))
        DiskSpillStore(tmp_path).spill(ctx("kept", FUTURE))
        cache = ContextCacheManager(1000, spill_store=DiskSpillStore(tmp_path))
        assert cache.expired_ids(time.time()) == ["old"]

    def test_unreadable_retention_metadata_counts_as_expired(self, tmp_path):
        store = DiskSpillStore(tmp_path)
        store.spill(ctx("bad", FUTURE))
        _rewrite_meta(store, "bad", lambda m: m.update(expires_at="soon"))
        cache = ContextCacheManager(1000, spill_store=DiskSpillStore(tmp_path))
        assert cache.expired_ids(time.time()) == ["bad"]  # and startup survived

    def test_the_listing_hides_expired_and_shows_deadlines(self, tmp_path):
        cache = ContextCacheManager(1000, spill_store=DiskSpillStore(tmp_path))
        cache.put(ctx("a", FUTURE))
        cache.evict_one()
        cache.put(ctx("b"))
        cache.put(ctx("gone", PAST))
        listed = {i.dataset_id: i.expires_at for i in cache.contexts()}
        assert listed == {"a": pytest.approx(FUTURE), "b": None}

    def test_the_servers_limit_covers_existing_contexts(self, tmp_path):
        """Fit before the limit was set (no expires_at), or by a tabctx that
        didn't record fit times: both are held to it."""
        cache = ContextCacheManager(
            1000, spill_store=DiskSpillStore(tmp_path), max_age_s=10
        )
        old = ctx("old")
        old.fitted_at = time.time() - 20
        unknown = ctx("unknown")
        unknown.fitted_at = None
        recent = ctx("recent", FUTURE)
        for c in (old, unknown, recent):
            cache.put(c)
        assert cache.expired_ids(time.time()) == ["old", "unknown"]
        (info,) = cache.contexts()
        assert info.expires_at == pytest.approx(recent.fitted_at + 10)

    def test_delete_if_expired_spares_a_refit(self):
        """The sweep lists an id, then the client fits it again: the new
        context must survive the sweep's delete."""
        cache = ContextCacheManager(1000)
        cache.put(ctx("d", PAST))
        assert cache.expired_ids(time.time()) == ["d"]
        cache.put(ctx("d"))  # re-fit, no expiry
        assert cache.delete_if_expired("d", time.time()) == (None, False)
        assert cache.holds("d")


class TestEngine:
    def test_an_expired_context_is_gone_when_next_used(self):
        engine = engine_with()
        engine.fit(X, Y, dataset_id="d", expires_at=PAST)
        with pytest.raises(DatasetNotFoundError, match="retention"):
            engine.predict("d", [[0.0, 1.0]])
        assert engine.contexts() == []

    def test_a_context_within_its_limit_serves(self):
        engine = engine_with()
        engine.fit(X, Y, dataset_id="d", expires_at=FUTURE)
        assert engine.predict("d", [[0.0, 1.0]]).predictions
        assert engine.contexts()[0].expires_at == FUTURE

    def test_expire_deletes_every_tier(self, tmp_path):
        engine = engine_with(spill_dir=tmp_path)
        engine.fit(X, Y, dataset_id="spilled", expires_at=PAST)
        engine._cache.evict_one()  # to disk
        engine.fit(X, Y, dataset_id="resident", expires_at=PAST)
        engine.fit(X, Y, dataset_id="kept", expires_at=FUTURE)
        assert sorted(engine.expire()) == ["resident", "spilled"]
        assert [i.dataset_id for i in engine.contexts()] == ["kept"]
        assert list(tmp_path.iterdir()) == []  # "kept" is resident

    def test_the_sweep_deletes_what_nobody_uses(self):
        engine = engine_with(sweep=0.02)
        engine.fit(X, Y, dataset_id="d", expires_at=time.time() + 0.05)
        for _ in range(200):
            if not engine.contexts():
                break
            threading.Event().wait(0.01)
        assert engine.contexts() == []

    def test_an_expired_spilled_context_is_a_404_and_leaves_no_files(self, tmp_path):
        engine = engine_with(spill_dir=tmp_path)
        engine.fit(X, Y, dataset_id="d", expires_at=PAST)
        engine._cache.evict_one()
        with pytest.raises(DatasetNotFoundError, match="retention"):
            engine.predict("d", [[0.0, 1.0]])
        assert list(tmp_path.iterdir()) == []

    def test_one_stuck_context_doesnt_stop_the_sweep(self, monkeypatch):
        from tabctx.errors import CacheUnavailableError

        engine = engine_with()
        for name in ("a", "stuck", "z"):
            engine.fit(X, Y, dataset_id=name, expires_at=PAST)
        real = engine._cache.delete_if_expired

        def delete(dataset_id, now, settle_timeout_s=600.0):
            if dataset_id == "stuck":
                raise CacheUnavailableError("moving")
            return real(dataset_id, now, settle_timeout_s)

        monkeypatch.setattr(engine._cache, "delete_if_expired", delete)
        assert engine.expire() == ["a", "z"]

    def test_the_sweep_spares_a_refit_listed_before_it_landed(self, monkeypatch):
        engine = engine_with()
        engine.fit(X, Y, dataset_id="d")  # fresh, no expiry
        monkeypatch.setattr(engine._cache, "expired_ids", lambda now: ["d"])
        assert engine.expire() == []
        assert engine.predict("d", [[0.0, 1.0]]).predictions

    def test_the_sweep_stops_with_its_engine(self):
        """It holds the engine weakly: it neither keeps one alive nor
        outlives it."""
        engine = engine_with(sweep=0.01)
        thread = engine._expiry_thread
        assert thread.is_alive()
        del engine
        gc.collect()
        thread.join(5)
        assert not thread.is_alive()


def test_the_router_and_worker_processes_carry_the_expiry():
    from tabctx.serve.factory import ServeSettings, build_engine

    built = build_engine(
        ServeSettings(backends=("fake",), devices=("cuda:0", "cuda:1"))
    )
    try:
        built.engine.fit(X, Y, dataset_id="d", expires_at=FUTURE)
        (info,) = built.engine.contexts()
        assert info.expires_at == pytest.approx(FUTURE)
    finally:
        built.close()


class TestSettings:
    def test_max_age_from_the_environment(self, monkeypatch):
        from tabctx.serve.factory import ServeSettings

        monkeypatch.setenv("TABCTX_MAX_CONTEXT_AGE_S", "86400")
        assert ServeSettings.from_env().max_context_age_s == 86400
        monkeypatch.delenv("TABCTX_MAX_CONTEXT_AGE_S")
        assert ServeSettings.from_env().max_context_age_s is None
        for bad in ("0", "-1", "a day"):
            monkeypatch.setenv("TABCTX_MAX_CONTEXT_AGE_S", bad)
            with pytest.raises(ValueError, match="MAX_CONTEXT_AGE_S"):
                ServeSettings.from_env()


class TestHttp:
    def _app(self, monkeypatch, max_age=None):
        from tabctx.serve.factory import BACKEND_ENV_VAR
        from tabctx.serve.standalone import make_app
        from tabctx.serve.tenancy import REQUIRE_TENANT_ENV_VAR

        monkeypatch.setenv(BACKEND_ENV_VAR, "fake")
        monkeypatch.delenv(REQUIRE_TENANT_ENV_VAR, raising=False)
        if max_age is None:
            monkeypatch.delenv("TABCTX_MAX_CONTEXT_AGE_S", raising=False)
        else:
            monkeypatch.setenv("TABCTX_MAX_CONTEXT_AGE_S", str(max_age))
        return make_app()

    def _call(self, app, method, path, **kwargs):
        import asyncio

        import httpx

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.request(method, path, **kwargs)

        return asyncio.run(go())

    def _fit(self, app, **extra):
        body = {"dataset_id": "d", "train_X": X, "train_y": Y, **extra}
        return self._call(app, "POST", "/v1/tabctx/fit", json=body)

    def test_ttl_sets_the_expiry_and_the_listing_shows_it(self, monkeypatch):
        app = self._app(monkeypatch)
        before = time.time()
        expires = self._fit(app, ttl_s=60).json()["expires_at"]
        assert before + 60 <= expires <= time.time() + 60
        listed = self._call(app, "GET", "/v1/tabctx/datasets").json()["data"]
        assert listed[0]["expires_at"] == pytest.approx(expires)

    def test_no_ttl_and_no_limit_keeps_it(self, monkeypatch):
        app = self._app(monkeypatch)
        assert self._fit(app).json()["expires_at"] is None

    def test_the_servers_limit_applies_and_caps(self, monkeypatch):
        app = self._app(monkeypatch, max_age=100)
        now = time.time()
        assert self._fit(app).json()["expires_at"] == pytest.approx(now + 100, abs=5)
        assert self._fit(app, ttl_s=10).json()["expires_at"] == pytest.approx(
            now + 10, abs=5
        )
        assert self._fit(app, ttl_s=10_000).json()["expires_at"] == pytest.approx(
            now + 100, abs=5
        )

    def test_a_form_field_and_bad_values(self, monkeypatch):
        app = self._app(monkeypatch)
        form = self._call(
            app,
            "POST",
            "/v1/tabctx/fit",
            data={"dataset_id": "f", "target": "y", "ttl_s": "30"},
            files={"file": ("t.csv", b"a,y\n0,x\n1,z\n2,x\n", "text/csv")},
        )
        assert form.status_code == 200 and form.json()["expires_at"] is not None
        assert self._fit(app, ttl_s=0).status_code == 422
        assert self._fit(app, ttl_s=-5).status_code == 422

    def test_expired_is_404_on_predict(self, monkeypatch):
        app = self._app(monkeypatch)
        self._fit(app, ttl_s=0.01)
        threading.Event().wait(0.05)
        response = self._call(
            app, "POST", "/v1/tabctx/predict", json={"dataset_id": "d", "test_X": X[:1]}
        )
        assert response.status_code == 404 and "retention" in response.text


def test_the_client_sends_ttl(monkeypatch):
    import json

    from tabctx.client import TabctxClient

    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"dataset_id": "d"}).encode()

    def handler(req, timeout=None):
        body = req.data
        sent.append(b"".join(body) if not isinstance(body, bytes) else body)
        return Response()

    monkeypatch.setattr("tabctx.client._open", handler)
    client = TabctxClient("http://x")
    client.fit([[1.0]], ["a"], dataset_id="d", ttl_s=60)
    assert json.loads(sent[0])["ttl_s"] == 60
    client.fit_file(b"a,y\n1,x\n", target="y", dataset_id="d", ttl_s=60)
    assert b'name="ttl_s"\r\n\r\n60.0\r\n' in sent[1]


class TestServiceRetention:
    def test_the_schema_of_an_expired_context_goes(self):
        from tabctx.schema import TableSchema
        from tabctx.serve.schemas import SchemaCache

        cache = SchemaCache()
        with cache.fitting("old") as record:
            record(TableSchema((("red",),)), PAST)
        with cache.fitting("kept") as record:
            record(TableSchema((("blue",),)), FUTURE)
        with cache.fitting("forever") as record:
            record(TableSchema((("green",),)))
        assert cache.get("old") is None  # past its deadline: dropped on read
        with cache.fitting("old2") as record:
            record(TableSchema((("red",),)), PAST)
        assert cache.purge_expired() == 1
        assert cache.get("kept") is not None and cache.get("forever") is not None

    def test_infinite_ttl_is_refused(self, monkeypatch):
        app = TestHttp()._app(monkeypatch)
        body = {"dataset_id": "d", "train_X": X, "train_y": Y}
        for bad in ("Infinity", "1e400"):
            raw = json.dumps(body)[:-1] + f', "ttl_s": {bad}}}'
            response = TestHttp()._call(
                app,
                "POST",
                "/v1/tabctx/fit",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 422, bad

    def test_limits_show_the_policy(self, monkeypatch):
        app = TestHttp()._app(monkeypatch, max_age=3600)
        limits = TestHttp()._call(app, "GET", "/v1/tabctx/limits").json()
        assert limits["max_context_age_s"] == 3600
