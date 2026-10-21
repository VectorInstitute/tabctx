"""Multipart bodies read as they arrive (serve/multipart.py)."""

import asyncio
import pickle

import numpy as np
import pytest

pytest.importorskip("python_multipart")

from tabctx.errors import InvalidInputError, UploadTooLargeError
from tabctx.schema import TableSchema
from tabctx.serve.multipart import MAX_FIELD_BYTES, read_form
from tabctx.serve.schemas import SchemaCache
from tabctx.serve.tables import learn_rows
from tabctx.serve.uploads import UploadStore

BOUNDARY = "b0undary"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


def body(fields=(), files=()):
    out = b""
    for name, value in fields:
        out += (
            (
                f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            ).encode()
            + value.encode()
            + b"\r\n"
        )
    for name, data in files:
        out += (
            (
                f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; '
                'filename="t.csv"\r\n\r\n'
            ).encode()
            + data
            + b"\r\n"
        )
    return out + f"--{BOUNDARY}--\r\n".encode()


async def chunks(data, size=7):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def read(data, store, content_type=CONTENT_TYPE):
    return asyncio.run(read_form(chunks(data), content_type, store.begin))


@pytest.fixture
def store(tmp_path):
    return UploadStore(directory=tmp_path, max_upload_bytes=1000)


def test_fields_and_the_file_in_small_chunks(store, tmp_path):
    table = b"a,b\n1,2\n" * 20
    fields, writer = read(
        body([("dataset_id", "d1"), ("target", "b")], [("file", table)]), store
    )
    assert fields == {"dataset_id": "d1", "target": "b"}
    path = writer.finish()
    assert path.read_bytes() == table


def test_no_file_part(store):
    fields, writer = read(body([("dataset_id", "d")]), store)
    assert fields == {"dataset_id": "d"} and writer is None


def test_the_cap_stops_the_upload_as_bytes_arrive(store, tmp_path):
    with pytest.raises(UploadTooLargeError):
        read(body(files=[("file", b"x" * 5000)]), store)
    assert list(tmp_path.iterdir()) == []  # nothing left on disk


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (body(files=[("file", b"a"), ("file", b"b")]), "one table file"),
        (body(files=[("table", b"a")]), "as the 'file' part"),
        (body([("f", "x" * (MAX_FIELD_BYTES + 1))]), "over"),
        (body([(f"f{i}", "x") for i in range(17)]), "more than 16"),
        (body(files=[("file", b"abc")])[:-20], "ends inside a part"),
    ],
)
def test_malformed_bodies_are_rejected_and_leave_nothing(store, tmp_path, data, match):
    with pytest.raises(InvalidInputError, match=match):
        read(data, store)
    assert list(tmp_path.iterdir()) == []


def test_a_missing_boundary_is_rejected(store):
    with pytest.raises(InvalidInputError, match="boundary"):
        read(body(), store, content_type="multipart/form-data")


def test_numeric_looking_text_cells_make_a_text_column():
    # numpy alone would read "007" as 7, depending on the other columns.
    _, schema = learn_rows([[1.0, "007"], [2.0, "010"]])
    assert schema.categories == (None, ("007", "010"))


def test_schema_code_maps_are_built_once_and_not_pickled():
    schema = TableSchema((("a", "b"),))
    schema.encode([np.array(["b"], dtype=object)])
    assert "_codes" in schema.__dict__
    copy = pickle.loads(pickle.dumps(schema))
    assert "_codes" not in copy.__dict__ and copy == schema


def test_schema_cache_is_a_bounded_lru():
    cache = SchemaCache(max_entries=2)
    a, b, c = (TableSchema(((n,),)) for n in "abc")
    cache.put("a", a)
    cache.put("b", b)
    assert cache.get("a") is a  # now most recent
    cache.put("c", c)
    assert cache.get("b") is None and cache.get("a") is a and cache.get("c") is c
