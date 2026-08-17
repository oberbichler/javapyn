"""
An independent, pure-Python javabin *framing* scanner.

It walks a javabin byte stream tag by tag and reports where every value ends,
plus a census of which tags occur. It decodes no values and shares no code with
the Rust decoder -- which is the point: if this scanner consumes exactly the
whole response and the Rust decoder also accepts it, two independent
implementations agree on the framing of every real Solr response. A framing bug
in either one shows up as leftover or missing bytes.

Written from the write-side conventions of
``org.apache.solr.common.util.JavaBinCodec`` (the same source as
``javabin_ref_encoder.py``).
"""

from __future__ import annotations

from collections import Counter

#: Tags encoded in the top 3 bits, with a 5-bit size/value in the low bits.
SHIFTED_TAGS = {
    1: "STR",
    2: "SINT",
    3: "SLONG",
    4: "ARR",
    5: "ORDERED_MAP",
    6: "NAMED_LST",
    7: "EXTERN_STRING",
}

#: Tags that occupy a whole byte (top 3 bits zero).
PLAIN_TAGS = {
    0: "NULL",
    1: "BOOL_TRUE",
    2: "BOOL_FALSE",
    3: "BYTE",
    4: "SHORT",
    5: "DOUBLE",
    6: "INT",
    7: "LONG",
    8: "FLOAT",
    9: "DATE",
    10: "MAP",
    11: "SOLRDOC",
    12: "SOLRDOCLST",
    13: "BYTEARR",
    14: "ITERATOR",
    15: "END",
    16: "SOLRINPUTDOC",
    17: "MAP_ENTRY_ITER",
    18: "ENUM_FIELD_VALUE",
    19: "MAP_ENTRY",
    20: "UUID",
    21: "PRIMITIVE_ARR",
}

ALL_TAGS = frozenset(PLAIN_TAGS.values()) | frozenset(SHIFTED_TAGS.values())

#: Payload width in bytes for the fixed-width scalar tags.
FIXED_WIDTH = {
    "BYTE": 1,
    "SHORT": 2,
    "FLOAT": 4,
    "INT": 4,
    "DOUBLE": 8,
    "LONG": 8,
    "DATE": 8,
    "UUID": 16,
}

#: Element width per sub-tag of a PRIMITIVE_ARR.
PRIMITIVE_WIDTH = {4: 2, 5: 8, 6: 4, 7: 8, 8: 4, 1: 1, 2: 1}

EXPECTED_VERSION = 2

_END = object()


class ScanError(Exception):
    """The stream is not walkable as javabin framing."""


class Scanner:
    """One-shot framing walk over a javabin byte string."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.tags: Counter[str] = Counter()
        self.depth = 0
        self.max_depth = 0

    def _byte(self) -> int:
        try:
            value = self.data[self.pos]
        except IndexError:
            raise ScanError(f"ran past end of data at {self.pos}") from None
        self.pos += 1
        return value

    def _vint(self) -> int:
        shift = 0
        result = 0
        while True:
            byte = self._byte()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    def _size(self, tag: int) -> int:
        """The 5-bit inline size, extended by a vint once it saturates."""
        low = tag & 0x1F
        return low if low < 0x1F else 0x1F + self._vint()

    def _skip(self, count: int) -> None:
        if count < 0 or self.pos + count > len(self.data):
            raise ScanError(f"payload of {count} bytes overruns the data at {self.pos}")
        self.pos += count

    def _value(self) -> object:
        """Walk one value; returns the END sentinel for an END tag."""
        tag = self._byte()
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            shifted = tag >> 5
            if shifted:
                name = SHIFTED_TAGS[shifted]
                self.tags[name] += 1
                if name == "STR":
                    self._skip(self._size(tag))
                elif name in ("SINT", "SLONG"):
                    if tag & 0x10:  # continuation flag: a vint follows
                        self._vint()
                elif name == "ARR":
                    for _ in range(self._size(tag)):
                        self._value()
                elif name in ("ORDERED_MAP", "NAMED_LST"):
                    for _ in range(self._size(tag)):
                        self._value()  # entry name -- an extern string, or NULL
                        self._value()
                else:  # EXTERN_STRING: index 0 means "definition follows"
                    if self._size(tag) == 0:
                        self._value()
                return None

            name = PLAIN_TAGS.get(tag)
            if name is None:
                raise ScanError(f"unknown tag {tag:#04x} at {self.pos - 1}")
            self.tags[name] += 1
            if name in FIXED_WIDTH:
                self._skip(FIXED_WIDTH[name])
            elif name == "END":
                return _END
            elif name == "BYTEARR":
                self._skip(self._vint())
            elif name == "MAP":
                for _ in range(self._vint()):
                    self._value()
                    self._value()
            elif name in ("SOLRDOC", "SOLRINPUTDOC"):
                self._value()  # the field ORDERED_MAP
            elif name == "SOLRDOCLST":
                self._value()  # [numFound, start, maxScore, numFoundExact]
                self._value()  # the documents
            elif name == "ITERATOR":
                while self._value() is not _END:
                    pass
            elif name == "MAP_ENTRY_ITER":
                while self._value() is not _END:
                    self._value()
            elif name in ("MAP_ENTRY", "ENUM_FIELD_VALUE"):
                self._value()
                self._value()
            elif name == "PRIMITIVE_ARR":
                sub_tag = self._byte()
                count = self._vint()
                if sub_tag not in PRIMITIVE_WIDTH:
                    raise ScanError(f"bad PRIMITIVE_ARR sub-tag {sub_tag:#04x}")
                self._skip(count * PRIMITIVE_WIDTH[sub_tag])
            return None
        finally:
            self.depth -= 1


def scan(data: bytes) -> Counter[str]:
    """Walk ``data`` as javabin and return its tag census.

    Raises :class:`ScanError` if the framing does not walk cleanly or if the
    stream does not end exactly where the top-level value does.
    """
    if not data:
        raise ScanError("empty data")
    if data[0] != EXPECTED_VERSION:
        raise ScanError(f"version byte {data[0]}, expected {EXPECTED_VERSION}")
    scanner = Scanner(data)
    scanner.pos = 1
    scanner._value()
    if scanner.pos != len(data):
        raise ScanError(
            f"top-level value ends at {scanner.pos} but data is {len(data)} bytes"
        )
    return scanner.tags
