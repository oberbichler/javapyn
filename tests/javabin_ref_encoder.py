"""Reference javabin (protocol v2) *encoder*, used only to build test fixtures.

This is a small, faithful re-implementation of the write-side of
``org.apache.solr.common.util.JavaBinCodec`` (Apache Solr), covering the
subset of types that occur in typical Solr query responses. It exists purely
so that the Rust decoder under test can be exercised against independently
produced, spec-accurate byte streams without needing network access to a
live Solr instance.

Do not use this for anything other than generating test fixtures.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

VERSION = 2

NULL = 0
BOOL_TRUE = 1
BOOL_FALSE = 2
BYTE = 3
SHORT = 4
DOUBLE = 5
INT = 6
LONG = 7
FLOAT = 8
DATE = 9
MAP = 10
SOLRDOC = 11
SOLRDOCLST = 12
BYTEARR = 13
ITERATOR = 14
END = 15

STR = 1 << 5
SINT = 2 << 5
SLONG = 3 << 5
ARR = 4 << 5
ORDERED_MAP = 5 << 5
NAMED_LST = 6 << 5
EXTERN_STRING = 7 << 5


class NamedList(list[tuple[str, Any]]):
    """Ordered, string-keyed, repeatable-key list (``NamedList``/``SimpleOrderedMap``)."""


@dataclass
class SolrDoc:
    fields: dict[str, Any] = field(default_factory=dict)
    children: list["SolrDoc"] = field(default_factory=list)


@dataclass
class SolrDocList:
    num_found: int
    start: int
    max_score: float | None
    num_found_exact: bool | None
    docs: list[SolrDoc]


@dataclass
class JavaDate:
    """Wraps milliseconds-since-epoch to force encoding with the ``DATE`` tag."""

    millis: int


class Encoder:
    """Stateful encoder (the extern-string cache is per-instance, like Java)."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self._strings: dict[str, int] = {}

    def encode(self, value: Any) -> bytes:
        self.buf = bytearray([VERSION])
        self._write_val(value)
        return bytes(self.buf)

    # -- low level ------------------------------------------------------

    def _write_vint(self, i: int) -> None:
        while i & ~0x7F:
            self.buf.append((i & 0x7F) | 0x80)
            i >>= 7
        self.buf.append(i & 0x7F)

    def _write_vlong(self, i: int) -> None:
        self._write_vint(i)  # identical bit-shifting logic, just wider

    def _write_tag(self, tag: int, size: int | None = None) -> None:
        if size is None:
            self.buf.append(tag)
            return
        if tag & 0xE0:
            if size < 0x1F:
                self.buf.append(tag | size)
            else:
                self.buf.append(tag | 0x1F)
                self._write_vint(size - 0x1F)
        else:
            self.buf.append(tag)
            self._write_vint(size)

    def _write_str(self, s: str) -> None:
        data = s.encode("utf-8")
        self._write_tag(STR, len(data))
        self.buf.extend(data)

    def _write_extern_string(self, s: str) -> None:
        idx = self._strings.get(s, 0)
        self._write_tag(EXTERN_STRING, idx)
        if idx == 0:
            self._write_str(s)
            self._strings[s] = len(self._strings) + 1

    # -- values -----------------------------------------------------------

    def _write_val(self, val: Any) -> None:
        if val is None:
            self.buf.append(NULL)
        elif isinstance(val, bool):
            self.buf.append(BOOL_TRUE if val else BOOL_FALSE)
        elif isinstance(val, str):
            self._write_str(val)
        elif isinstance(val, int):
            self._write_int_or_long(val)
        elif isinstance(val, float):
            self._write_float(val)
        elif isinstance(val, JavaDate):
            self.write_date_millis(val.millis)
        elif isinstance(val, bytes):
            self._write_tag(BYTEARR, len(val))
            self.buf.extend(val)
        elif isinstance(val, NamedList):
            self._write_named_list(val)
        elif isinstance(val, SolrDocList):
            self._write_solr_doc_list(val)
        elif isinstance(val, SolrDoc):
            self._write_solr_doc(val)
        elif isinstance(val, dict):
            self._write_map(val)
        elif isinstance(val, (list, tuple)):
            self._write_array(list(val))
        else:
            raise TypeError(f"unsupported value type: {type(val)!r}")

    def _write_int_or_long(self, val: int, *, force_long: bool = False) -> None:
        is_long = force_long or not (-(2**31) <= val < 2**31)
        if not is_long:
            if val > 0:
                low = val & 0x0F
                if val >= 0x0F:
                    self.buf.append(SINT | 0x10 | low)
                    self._write_vint(val >> 4)
                else:
                    self.buf.append(SINT | low)
            else:
                self.buf.append(INT)
                self.buf.extend(struct.pack(">i", val))
        else:
            if 0 <= val < 2**56:
                low = val & 0x0F
                if val >= 0x0F:
                    self.buf.append(SLONG | 0x10 | low)
                    self._write_vlong(val >> 4)
                else:
                    self.buf.append(SLONG | low)
            else:
                self.buf.append(LONG)
                self.buf.extend(struct.pack(">q", val))

    def write_long(self, val: int) -> None:
        """Force encoding ``val`` with LONG/SLONG tags (as Solr does for numFound etc.)."""
        self._write_int_or_long(val, force_long=True)

    def _write_float(self, val: float) -> None:
        self.buf.append(FLOAT)
        self.buf.extend(struct.pack(">f", val))

    def write_double(self, val: float) -> None:
        self.buf.append(DOUBLE)
        self.buf.extend(struct.pack(">d", val))

    def write_date_millis(self, millis: int) -> None:
        self.buf.append(DATE)
        self.buf.extend(struct.pack(">q", millis))

    def _write_array(self, items: list[Any]) -> None:
        self._write_tag(ARR, len(items))
        for item in items:
            self._write_val(item)

    def _write_named_list(self, nl: NamedList, tag: int = NAMED_LST) -> None:
        self._write_tag(tag, len(nl))
        for name, value in nl:
            self._write_extern_string(name)
            self._write_val(value)

    def _write_map(self, m: dict[Any, Any]) -> None:
        self._write_tag(MAP, len(m))
        for key, value in m.items():
            if isinstance(key, str):
                self._write_extern_string(key)
            else:
                self._write_val(key)
            self._write_val(value)

    def _write_solr_doc(self, doc: SolrDoc) -> None:
        self.buf.append(SOLRDOC)
        self._write_tag(ORDERED_MAP, len(doc.fields) + len(doc.children))
        for name, value in doc.fields.items():
            self._write_extern_string(name)
            self._write_val(value)
        for child in doc.children:
            self._write_solr_doc(child)

    def _write_solr_doc_list(self, dl: SolrDocList) -> None:
        self.buf.append(SOLRDOCLST)
        # header: [numFound(long), start(long), maxScore(float|null), numFoundExact(bool|null)]
        self._write_tag(ARR, 4)
        self.write_long(dl.num_found)
        self.write_long(dl.start)
        self._write_val(dl.max_score)
        self._write_val(dl.num_found_exact)
        self._write_array(dl.docs)


def encode(value: Any) -> bytes:
    """Encode a single top-level value as a javabin v2 byte string."""
    return Encoder().encode(value)
