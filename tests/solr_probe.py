"""
HTTP helper for the live-Solr conformance tests.

Wraps the handful of Solr APIs the container tests need: creating schemas,
indexing the test corpora, and fetching the *same* query twice -- once as
``wt=javabin`` (the bytes under test) and once as ``wt=json`` (the reference
Solr itself considers correct).

Two collections are provisioned, for two different jobs:

``solr_movies``
    A bulk corpus, large enough that ``/export`` spans several network chunks.
    Drives the throughput and streaming tests.
``solr_types``
    A small corpus that puts every javabin scalar tag and every awkward value
    (numeric extremes, denormals, non-BMP text, empty and huge strings, binary
    blobs, absent fields, nested child documents) into a real Solr.

Deliberately free of pytest imports so it stays a plain, reusable client;
the container lifecycle lives in ``conftest.py``.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

COLLECTION = "solr_movies"
TYPES_COLLECTION = "solr_types"

#: Enough documents that an ``/export`` response spans more than one network
#: chunk (~210 KB), so ``StreamDecoder`` has to resume across a real HTTP chunk
#: boundary rather than a synthetic split, and repeated ``genres`` values fill
#: Solr's EXTERN_STRING table with real back-references.
DOC_COUNT = 2000

GENRES = ("Drama", "Comedy", "Sci-Fi", "Thriller")

#: The bulk corpus schema. Defined explicitly rather than via schemaless field
#: guessing because ``/export`` requires ``docValues`` on every ``fl`` and sort
#: field, and guessing would infer ``multiValued`` from whichever value it sees
#: first.
SCHEMA_FIELDS: tuple[dict[str, Any], ...] = (
    {"name": "movie_id", "type": "string"},
    {"name": "title", "type": "string"},
    {"name": "rating", "type": "pfloat"},
    {"name": "box_office", "type": "pdouble"},
    {"name": "release_year", "type": "pint"},
    {"name": "view_count", "type": "plong"},
    {"name": "is_classic", "type": "boolean"},
    {"name": "genres", "type": "string", "multiValued": True},
    {"name": "last_updated", "type": "pdate"},
)

#: All bulk fields, usable as an ``/export`` field list (every one has docValues).
EXPORT_FIELDS = ",".join(f["name"] for f in SCHEMA_FIELDS)

#: One field per javabin scalar tag, plus the shapes that stress the string and
#: container tags. ``t_text``/``t_blob``/``t_huge`` are stored-only: a text field
#: cannot have docValues, a binary field can be neither indexed nor sorted, and
#: an indexed string is capped at Lucene's 32 766-byte term limit.
TYPE_FIELDS: tuple[dict[str, Any], ...] = (
    {"name": "t_string", "type": "string"},
    {"name": "t_text", "type": "text_general", "docValues": False},
    {"name": "t_huge", "type": "string", "indexed": False, "docValues": False},
    {"name": "t_blob", "type": "binary", "indexed": False, "docValues": False},
    {"name": "t_bool", "type": "boolean"},
    {"name": "t_int", "type": "pint"},
    {"name": "t_long", "type": "plong"},
    {"name": "t_float", "type": "pfloat"},
    {"name": "t_double", "type": "pdouble"},
    {"name": "t_date", "type": "pdate"},
    # Multi-valued has to be requested per field: the plural field *types*
    # (strings, pints, ...) only carry multiValued through the _default
    # configset's dynamic fields, not through an explicitly added field.
    {"name": "t_strings", "type": "string", "multiValued": True},
    {"name": "t_ints", "type": "pint", "multiValued": True},
    {"name": "t_longs", "type": "plong", "multiValued": True},
    {"name": "t_floats", "type": "pfloat", "multiValued": True},
    {"name": "t_doubles", "type": "pdouble", "multiValued": True},
    {"name": "t_dates", "type": "pdate", "multiValued": True},
    {"name": "t_bools", "type": "boolean", "multiValued": True},
)

#: The subset of type fields that ``/export`` accepts (docValues, single or
#: multi valued, no text or binary).
TYPE_EXPORT_FIELDS = ",".join(
    f["name"] for f in TYPE_FIELDS if f["name"] not in ("t_text", "t_huge", "t_blob")
)

INT_MAX, INT_MIN = 2**31 - 1, -(2**31)
LONG_MAX, LONG_MIN = 2**63 - 1, -(2**63)
FLOAT_MAX = 3.4028234663852886e38
DOUBLE_MAX = 1.7976931348623157e308


def _bulk_documents() -> list[dict[str, Any]]:
    """A deterministic corpus covering every bulk field, with multi-valued and
    non-ASCII values.
    """
    return [
        {
            "movie_id": f"mv-{i:05d}",
            "title": f"Filmtitel {i} — äöü ✓ 日本",
            "rating": round(1.0 + (i % 100) * 0.037, 3),
            "box_office": 1234567.89 + i * 1000.5,
            "release_year": 1900 + (i % 125),
            "view_count": 10_000_000_000 + i,
            "is_classic": i % 2 == 0,
            "genres": [GENRES[i % 4], GENRES[(i + 1) % 4]],
            "last_updated": f"2024-01-{(i % 28) + 1:02d}T12:34:56Z",
        }
        for i in range(DOC_COUNT)
    ]


def _type_documents() -> list[dict[str, Any]]:
    """One document per awkward case. Every id is asserted on by name in the
    conformance tests, so keep them stable.
    """
    return [
        # Numeric extremes: the INT/LONG/FLOAT/DOUBLE tags at their limits.
        {
            "id": "extremes_max",
            "t_string": "max",
            "t_int": INT_MAX,
            "t_long": LONG_MAX,
            "t_float": FLOAT_MAX,
            "t_double": DOUBLE_MAX,
            "t_date": "9999-12-31T23:59:59.999Z",
            "t_bool": True,
        },
        {
            "id": "extremes_min",
            "t_string": "min",
            "t_int": INT_MIN,
            "t_long": LONG_MIN,
            "t_float": -FLOAT_MAX,
            "t_double": -DOUBLE_MAX,
            "t_date": "0001-01-01T00:00:00Z",
            "t_bool": False,
        },
        # Zero, negative zero, the empty string, an empty multi-valued field.
        {
            "id": "zeros",
            "t_string": "",
            "t_int": 0,
            "t_long": 0,
            "t_float": -0.0,
            "t_double": -0.0,
            "t_date": "1970-01-01T00:00:00Z",
            "t_strings": [],
        },
        # Smallest representable magnitudes.
        {
            "id": "denormals",
            "t_string": "denormals",
            "t_float": 1.401298464324817e-45,
            "t_double": 5e-324,
        },
        # SINT/SLONG inline-vs-continuation boundaries (0x0f, 0x7f, 0x3fff, ...).
        {
            "id": "int_boundaries",
            "t_string": "boundaries",
            "t_ints": [-1, 0, 1, 14, 15, 16, 127, 128, 16383, 16384, INT_MAX],
            "t_longs": [LONG_MIN, -1, 0, 1, 2**31, 2**56 - 1, 2**56, LONG_MAX],
        },
        # STR inline-size boundary (30 bytes) and its vint extension.
        {
            "id": "string_lengths",
            "t_string": "x" * 31,
            "t_strings": ["x" * n for n in (1, 30, 31, 32, 127, 128, 255, 256, 4096)],
        },
        # A string far past the inline size, and non-ASCII so bytes != chars.
        {"id": "string_huge", "t_string": "huge", "t_huge": "ä" * 200_000},
        # Non-BMP code points, ZWJ sequences, combining marks, RTL scripts.
        {
            "id": "unicode",
            "t_string": "Grüße",
            "t_huge": "𝄞 👨‍👩‍👧‍👦 مرحبا שלום 日本語 é ​ ✓",
            "t_strings": ["🎬", "🇩🇪", "áb", "\U0001f600" * 50],
        },
        # BYTEARR at zero, one, 256 and 76 800 bytes.
        {
            "id": "blob_empty",
            "t_string": "blob",
            "t_blob": base64.b64encode(b"").decode(),
        },
        {
            "id": "blob_one",
            "t_string": "blob",
            "t_blob": base64.b64encode(b"\x00").decode(),
        },
        {
            "id": "blob_256",
            "t_string": "blob",
            "t_blob": base64.b64encode(bytes(range(256))).decode(),
        },
        {
            "id": "blob_big",
            "t_string": "blob",
            "t_blob": base64.b64encode(bytes(range(256)) * 300).decode(),
        },
        # Every optional field absent.
        {"id": "all_absent", "t_string": "absent"},
        # Multi-valued with exactly one value, and with many.
        {
            "id": "multi_one",
            "t_string": "multi",
            "t_strings": ["only"],
            "t_ints": [7],
            "t_bools": [True],
            "t_floats": [1.5],
            "t_doubles": [2.5],
            "t_dates": ["2020-02-29T12:00:00.500Z"],
        },
        {
            "id": "multi_many",
            "t_string": "multi",
            "t_strings": [f"v{i}" for i in range(50)],
            "t_ints": list(range(50)),
            "t_floats": [i * 0.1 for i in range(20)],
            "t_doubles": [i * 0.1 for i in range(20)],
            "t_bools": [True, False, True],
            "t_dates": [
                "1969-12-31T23:59:59.999Z",  # negative millis
                "1970-01-01T00:00:00.001Z",
                "2038-01-19T03:14:08Z",  # past the 32-bit epoch
            ],
        },
        # A tokenised text field, so the highlighter has something to work with.
        {
            "id": "text_doc",
            "t_string": "text",
            "t_text": "the quick brown fox jumps over the lazy dog",
        },
        # Fractional seconds in both directions.
        {
            "id": "date_fractions",
            "t_string": "dates",
            "t_date": "2024-06-01T12:34:56.001Z",
            "t_dates": ["2024-06-01T12:34:56.999Z", "1900-01-01T00:00:00Z"],
        },
        # Nested child documents, two levels deep.
        {
            "id": "parent",
            "t_string": "parent",
            "t_int": 1,
            "children": [
                {
                    "id": "child_1",
                    "t_string": "child",
                    "t_int": 11,
                    "grandchildren": [
                        {"id": "grandchild_1", "t_string": "grandchild", "t_int": 111}
                    ],
                },
                {"id": "child_2", "t_string": "child", "t_int": 12},
            ],
        },
    ]


def count_documents(docs: list[dict[str, Any]]) -> int:
    """Total documents an update request creates, nesting included.

    Solr indexes a child document as a document in its own right, so the parent
    of two children with one grandchild accounts for four.
    """
    total = 0
    for doc in docs:
        total += 1
        for value in doc.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                total += count_documents(value)
    return total


class SolrProbe:
    """Thin client against the two provisioned collections."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    # -- provisioning ----------------------------------------------------

    def wait_until_ready(self, attempts: int = 120, delay: float = 1.0) -> None:
        """Poll the admin API until Solr answers. Polls HTTP rather than
        matching startup log lines, which change between releases.
        """
        last: Exception | None = None
        for _ in range(attempts):
            try:
                response = self.client.get(
                    f"{self.base_url}/admin/info/system",
                    params={"wt": "json"},
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last = exc
            time.sleep(delay)
        raise RuntimeError(f"Solr did not become ready at {self.base_url}: {last}")

    def provision(self) -> None:
        """Create both collections, their schemas, and their corpora."""
        for coll, fields, docs in (
            (COLLECTION, SCHEMA_FIELDS, _bulk_documents()),
            (TYPES_COLLECTION, TYPE_FIELDS, _type_documents()),
        ):
            self._create_collection(coll)
            self._create_schema(coll, fields)
            self._index(coll, docs)
            self._wait_for_documents(coll, count_documents(docs))

    def _create_collection(self, coll: str) -> None:
        response = self.client.get(
            f"{self.base_url}/admin/collections",
            params={
                "action": "CREATE",
                "name": coll,
                "numShards": "1",
                "replicationFactor": "1",
            },
        )
        response.raise_for_status()

    def _create_schema(self, coll: str, fields: tuple[dict[str, Any], ...]) -> None:
        payload = {
            "add-field": [
                {
                    "stored": True,
                    "indexed": True,
                    "docValues": True,
                    "multiValued": field.get("multiValued", False),
                    **field,
                }
                for field in fields
            ]
        }
        response = self.client.post(f"{self.base_url}/{coll}/schema", json=payload)
        response.raise_for_status()

    def _index(self, coll: str, docs: list[dict[str, Any]]) -> None:
        for start in range(0, len(docs), 1000):
            response = self.client.post(
                f"{self.base_url}/{coll}/update", json=docs[start : start + 1000]
            )
            response.raise_for_status()
        response = self.client.get(
            f"{self.base_url}/{coll}/update", params={"commit": "true"}
        )
        response.raise_for_status()

    def _wait_for_documents(
        self, coll: str, expected: int, attempts: int = 60, delay: float = 0.5
    ) -> None:
        """Poll until every indexed document is searchable.

        A synchronous ``commit=true`` is not quite the end of the story: on Solr 8
        the first queries after provisioning intermittently saw an empty
        collection, so waiting for the process to be up is not enough -- the data
        has to be visible too, or the earliest tests fail for reasons that have
        nothing to do with the decoder.
        """
        last = -1
        for _ in range(attempts):
            response = self.client.get(
                f"{self.base_url}/{coll}/select",
                params={"q": "*:*", "rows": "0", "wt": "json"},
            )
            if response.status_code == 200:
                last = response.json()["response"]["numFound"]
                if last >= expected:
                    return
            time.sleep(delay)
        raise RuntimeError(
            f"{coll}: only {last} of {expected} documents became searchable"
        )

    # -- dual-format fetches --------------------------------------------

    def request(
        self,
        handler: str,
        params: dict[str, Any],
        *,
        coll: str = COLLECTION,
        method: str = "GET",
    ) -> tuple[bytes, dict]:
        """Run one query twice; return (javabin bytes, wt=json dict).

        Both requests carry ``version=2`` so the echoed parameters differ only
        in ``wt``, which the comparison ignores.
        """
        javabin = self.raw(
            handler, {**params, "wt": "javabin"}, coll=coll, method=method
        )
        reference = self.raw(
            handler, {**params, "wt": "json"}, coll=coll, method=method
        )
        return javabin, json.loads(reference)

    def raw(
        self,
        handler: str,
        params: dict[str, Any],
        *,
        coll: str = COLLECTION,
        method: str = "GET",
    ) -> bytes:
        """Fetch one response body verbatim, whatever its HTTP status."""
        url = f"{self.base_url}/{coll}/{handler}"
        query = {"version": "2", **params}
        if method == "GET":
            response = self.client.get(url, params=query)
        else:
            response = self.client.post(url, data=query)
        return response.content

    def select(self, *, coll: str = COLLECTION, **params: str) -> tuple[bytes, dict]:
        """Run a ``/select`` query twice; return (javabin bytes, wt=json dict)."""
        sort = "movie_id asc" if coll == COLLECTION else "id asc"
        return self.request("select", {"q": "*:*", "sort": sort, **params}, coll=coll)

    def export(self, *, coll: str = COLLECTION, **params: str) -> tuple[bytes, dict]:
        """Run an ``/export`` query twice; return (javabin bytes, wt=json dict)."""
        fields = EXPORT_FIELDS if coll == COLLECTION else TYPE_EXPORT_FIELDS
        sort = "movie_id asc" if coll == COLLECTION else "id asc"
        return self.request(
            "export",
            {"q": "*:*", "fl": fields, "sort": sort, **params},
            coll=coll,
            method="POST",
        )

    def stream(self, expr: str, *, coll: str = COLLECTION) -> tuple[bytes, dict]:
        """Run a streaming expression twice; return (javabin bytes, wt=json dict)."""
        return self.request("stream", {"expr": expr}, coll=coll, method="POST")

    def export_chunks(
        self, *, coll: str = COLLECTION, **params: str
    ) -> Iterator[bytes]:
        """Yield the raw ``wt=javabin`` ``/export`` body as it arrives from the
        network, so decoders see real HTTP chunk boundaries.
        """
        fields = EXPORT_FIELDS if coll == COLLECTION else TYPE_EXPORT_FIELDS
        sort = "movie_id asc" if coll == COLLECTION else "id asc"
        query = {
            "q": "*:*",
            "fl": fields,
            "sort": sort,
            "wt": "javabin",
            "version": "2",
            **params,
        }
        with self.client.stream(
            "POST", f"{self.base_url}/{coll}/export", data=query
        ) as response:
            response.raise_for_status()
            yield from response.iter_bytes()
