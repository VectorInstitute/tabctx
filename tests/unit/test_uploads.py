"""Unit tests for the replica-local upload store (serve/uploads.py)."""

import pytest

from tabctx.errors import UploadNotFoundError, UploadTooLargeError
from tabctx.serve.uploads import UploadStore


@pytest.fixture
def store(tmp_path):
    return UploadStore(directory=tmp_path, ttl_s=3600, max_upload_bytes=1000)


class TestPutConsume:
    def test_roundtrip(self, store):
        record = store.put([b"a,b\n", b"1,2\n"])
        assert record.n_bytes == 8
        path = store.consume(record.upload_id)
        assert path.read_bytes() == b"a,b\n1,2\n"
        store.discard(path)
        assert not path.exists()

    def test_single_use(self, store):
        record = store.put([b"x"])
        store.consume(record.upload_id)
        with pytest.raises(UploadNotFoundError):
            store.consume(record.upload_id)

    def test_unknown_id(self, store):
        with pytest.raises(UploadNotFoundError):
            store.consume("nope")

    def test_streaming_writer(self, store):
        writer = store.begin()
        for chunk in (b"aa", b"bb", b"cc"):
            writer.write(chunk)
        record = writer.commit()
        assert record.n_bytes == 6
        assert store.consume(record.upload_id).read_bytes() == b"aabbcc"

    def test_abort_leaves_nothing(self, store, tmp_path):
        writer = store.begin()
        writer.write(b"partial")
        writer.abort()
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(UploadNotFoundError):
            store.consume(writer.upload_id)


class TestSizeCap:
    def test_oversized_rejected_and_cleaned(self, store, tmp_path):
        with pytest.raises(UploadTooLargeError):
            store.put([b"x" * 600, b"y" * 600])
        assert list(tmp_path.iterdir()) == []

    def test_cap_boundary_ok(self, store):
        record = store.put([b"x" * 1000])
        assert record.n_bytes == 1000


class TestTenantScoping:
    def test_wrong_tenant_sees_not_found(self, store):
        record = store.put([b"data"], tenant_id="acme")
        with pytest.raises(UploadNotFoundError):
            store.consume(record.upload_id, tenant_id="globex")
        with pytest.raises(UploadNotFoundError):
            store.consume(record.upload_id, tenant_id=None)
        # The rightful tenant still gets it (the mismatches above must
        # not have consumed or leaked it).
        assert store.consume(record.upload_id, tenant_id="acme").exists()


class TestExpiry:
    def test_expired_swept_and_gone(self, tmp_path):
        store = UploadStore(directory=tmp_path, ttl_s=0.05, max_upload_bytes=1000)
        record = store.put([b"data"])
        import time

        time.sleep(0.1)
        assert store.sweep_expired() == 1
        assert not record.path.exists()
        with pytest.raises(UploadNotFoundError):
            store.consume(record.upload_id)

    def test_stats(self, store):
        store.put([b"1234"])
        stats = store.stats()
        assert stats["n_pending_uploads"] == 1
        assert stats["pending_bytes"] == 4


class TestDefaultDirectory:
    def test_no_directory_uses_a_fresh_tempdir(self):
        store = UploadStore()
        record = store.put([b"data"])
        assert record.path.exists()
        assert store.consume(record.upload_id) == record.path


class TestIteration:
    def test_iterates_pending_records(self, store):
        a = store.put([b"a"])
        b = store.put([b"bb"])
        ids = {r.upload_id for r in store}
        assert ids == {a.upload_id, b.upload_id}
