"""A multipart/form-data body read as it arrives.

Text fields go to memory (few and small, both capped); the one ``file``
part goes straight to an upload writer (serve/uploads.py), which enforces
the upload size cap as bytes arrive. Nothing is spooled or copied first,
so a body over the cap stops at the cap, and a good one is written to
disk once. Disk writes run off the event loop.

Fields are strict UTF-8 (a charset parameter naming anything else is
refused) and each appears once, so a gateway that routes on them reads
exactly the values this server uses. Malformed bodies raise
InvalidInputError (a 422).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header

from tabctx.errors import InvalidInputError

# python-multipart logs every malformed body it meets; the 422 says it.
logging.getLogger("python_multipart").setLevel(logging.ERROR)

FILE_FIELD = "file"
MAX_FIELDS = 16
MAX_FIELD_BYTES = 64 * 1024


class Writer(Protocol):
    def write(self, chunk: bytes) -> None: ...
    def abort(self) -> None: ...


async def read_form(
    stream: AsyncIterator[bytes], content_type: str, begin_file: Callable[[], Writer]
) -> tuple[dict[str, str], Writer | None]:
    """(text fields, the file part's writer or None). The caller finishes
    the writer; on any error it has been aborted."""
    _, params = parse_options_header(content_type)
    charset = params.get(b"charset", b"utf-8").lower()
    if charset not in (b"utf-8", b"utf8"):
        # Fields are UTF-8, always: a proxy in front that read them in
        # another charset would route by values this server never sees.
        raise InvalidInputError(
            "multipart form fields must be UTF-8 (no other charset)"
        )
    boundary = params.get(b"boundary")
    if not boundary:
        raise InvalidInputError("multipart body without a boundary")
    form = _Form(begin_file)
    try:
        parser = MultipartParser(boundary, form.callbacks())
        async for chunk in stream:
            parser.write(chunk)
            if form.pending:
                await asyncio.to_thread(form.flush)
        parser.finalize()
        if form.in_part:
            raise InvalidInputError("the multipart body ends inside a part")
        form.flush()
    except FormParserError as e:
        if form.writer is not None:
            form.writer.abort()
        raise InvalidInputError(f"malformed multipart body: {e}") from e
    except BaseException:
        if form.writer is not None:
            form.writer.abort()
        raise
    return form.fields, form.writer


class _Form:
    def __init__(self, begin_file: Callable[[], Writer]) -> None:
        self._begin_file = begin_file
        self.fields: dict[str, str] = {}
        self.writer: Writer | None = None
        self.pending: list[bytes] = []  # file bytes parsed, not yet written
        self.in_part = False
        self.open_part = ""
        self._n_fields = 0
        self._in_file = False
        self._data = bytearray()
        self._header_name = b""
        self._header_value = b""
        self._disposition = b""

    def flush(self) -> None:
        pending, self.pending = self.pending, []
        for chunk in pending:
            self.writer.write(chunk)

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value_part,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def _part_begin(self) -> None:
        self.in_part = True
        self._data = bytearray()
        self._disposition = b""
        self._in_file = False

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _header_value_part(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._disposition = self._header_value
        self._header_name = self._header_value = b""

    def _headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        if b"name" not in options:
            raise InvalidInputError("a multipart part has no name")
        name = options[b"name"].decode("utf-8", "replace")
        self.open_part = name
        if b"filename" in options:
            if name != FILE_FIELD:
                raise InvalidInputError(
                    f"send the table as the {FILE_FIELD!r} part, not {name!r}"
                )
            if self.writer is not None:
                raise InvalidInputError("a request carries one table file")
            self.writer = self._begin_file()
            self._in_file = True
        elif name == FILE_FIELD:
            raise InvalidInputError(
                f"send the table as a file part (curl: -F {FILE_FIELD}=@train.csv)"
            )
        else:
            self._n_fields += 1
            if self._n_fields > MAX_FIELDS:
                raise InvalidInputError(f"more than {MAX_FIELDS} form fields")

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._in_file:
            self.pending.append(bytes(data[start:end]))
            return
        self._data += data[start:end]
        if len(self._data) > MAX_FIELD_BYTES:
            raise InvalidInputError(
                f"form field {self.open_part!r} is over {MAX_FIELD_BYTES} bytes"
            )

    def _part_end(self) -> None:
        if not self._in_file:
            if self.open_part in self.fields:
                # One value per field, so no reader can pick a different one.
                raise InvalidInputError(f"form field {self.open_part!r} is repeated")
            try:
                self.fields[self.open_part] = self._data.decode("utf-8")
            except UnicodeDecodeError as e:
                raise InvalidInputError(
                    f"form field {self.open_part!r} is not UTF-8 text"
                ) from e
        self.in_part = False
