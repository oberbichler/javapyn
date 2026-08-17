"""
Deep conformance and robustness tests against a real Apache Solr.

``test_solr_container.py`` covers the happy path of the three documented
handlers. This module is the thorough pass: every javabin scalar tag driven from
a real schema, every awkward value Solr can put on the wire, the response shapes
of the search components (facets, stats, grouping, highlighting, debug, terms,
cursors, collapse), error responses, and robustness against truncated and
corrupted real bytes.

Several tests are regression tests for defects this suite found: the rejected
null NamedList name (``test_facet_missing_bucket_decodes``), the silently
skipped ``/export`` error (``test_export_error_is_visible_to_streaming_decoders``)
and the lost infinities (``test_deserialize_json_keeps_non_finite_doubles``).

Requires Docker; deselected by default. Run with ``pytest -m solr``.
"""

import json
import math
import random

import pytest
from javabin_compare import (
    assert_json_path_matches,
    assert_matches_json,
    floats_match,
    solr_date_to_millis,
)
from javabin_scanner import ALL_TAGS, ScanError, scan
from solr_probe import (
    DOC_COUNT,
    DOUBLE_MAX,
    FLOAT_MAX,
    INT_MAX,
    INT_MIN,
    LONG_MAX,
    LONG_MIN,
    TYPES_COLLECTION,
    SolrProbe,
)

import javapyn as javabin

pytestmark = pytest.mark.solr

TYPES = {"coll": TYPES_COLLECTION}

#: Documents in solr_types, including the two nested children and one
#: grandchild, which Solr indexes as documents in their own right.
TYPES_DOC_COUNT = 21


def docs_by_id(result: dict) -> dict[str, dict]:
    return {doc["id"]: doc for doc in result["response"]["docs"]}


# -- every field type, every awkward value ------------------------------------


def test_all_field_types_match_json_reference(solr: SolrProbe) -> None:
    """Every field of every type document, compared against wt=json.

    This is the broadest single assertion in the suite: one deep comparison over
    numeric extremes, denormals, empty and huge strings, non-BMP text, binary
    blobs, absent fields and multi-valued fields, in both renderings.
    """
    data, ref = solr.select(rows="100", fl="*", **TYPES)

    result = javabin.deserialize(data)

    assert result["response"]["numFound"] == TYPES_DOC_COUNT
    assert_matches_json(result, ref, "solr_types /select fl=*")


def test_numeric_extremes(solr: SolrProbe) -> None:
    """INT/LONG/FLOAT/DOUBLE at their limits survive the round trip exactly."""
    data, _ = solr.select(q="id:extremes_*", rows="10", fl="*", **TYPES)
    docs = docs_by_id(javabin.deserialize(data))

    assert docs["extremes_max"]["t_int"] == INT_MAX
    assert docs["extremes_min"]["t_int"] == INT_MIN
    assert docs["extremes_max"]["t_long"] == LONG_MAX
    assert docs["extremes_min"]["t_long"] == LONG_MIN
    assert docs["extremes_max"]["t_double"] == DOUBLE_MAX
    assert docs["extremes_min"]["t_double"] == -DOUBLE_MAX
    assert floats_match(docs["extremes_max"]["t_float"], FLOAT_MAX)
    assert floats_match(docs["extremes_min"]["t_float"], -FLOAT_MAX)


def test_zero_and_denormal_values(solr: SolrProbe) -> None:
    """Negative zero keeps its sign; the smallest denormals keep their value."""
    data, _ = solr.select(q="id:zeros OR id:denormals", rows="10", fl="*", **TYPES)
    docs = docs_by_id(javabin.deserialize(data))

    # Solr drops an empty string rather than storing it, so the zero-length STR
    # tag is not reachable from a field; the reference-encoder tests cover it.
    assert "t_string" not in docs["zeros"]
    assert docs["zeros"]["t_int"] == 0
    assert math.copysign(1.0, docs["zeros"]["t_double"]) == -1.0
    assert math.copysign(1.0, docs["zeros"]["t_float"]) == -1.0
    assert docs["denormals"]["t_double"] == 5e-324
    assert floats_match(docs["denormals"]["t_float"], 1.401298464324817e-45)


def test_integer_tag_boundaries(solr: SolrProbe) -> None:
    """SINT/SLONG switch from an inline 4-bit value to a vint continuation; both
    sides of every boundary must decode identically."""
    data, ref = solr.select(q="id:int_boundaries", fl="*", **TYPES)
    result = javabin.deserialize(data)

    assert_matches_json(result, ref, "int boundaries")
    doc = docs_by_id(result)["int_boundaries"]
    assert doc["t_ints"] == [-1, 0, 1, 14, 15, 16, 127, 128, 16383, 16384, INT_MAX]
    assert doc["t_longs"] == [LONG_MIN, -1, 0, 1, 2**31, 2**56 - 1, 2**56, LONG_MAX]


def test_string_lengths_and_huge_string(solr: SolrProbe) -> None:
    """The STR inline-size boundary (30 bytes) and a 200 000-character value."""
    data, ref = solr.select(
        q="id:string_lengths OR id:string_huge", rows="10", fl="*", **TYPES
    )
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "string lengths")

    docs = docs_by_id(result)
    assert [len(s) for s in docs["string_lengths"]["t_strings"]] == [
        1,
        30,
        31,
        32,
        127,
        128,
        255,
        256,
        4096,
    ]
    huge = docs["string_huge"]["t_huge"]
    assert len(huge) == 200_000  # characters, not bytes: 400 000 bytes on the wire
    assert set(huge) == {"ä"}


def test_non_bmp_and_combining_unicode(solr: SolrProbe) -> None:
    """Astral-plane code points, ZWJ sequences, combining marks and RTL text."""
    data, ref = solr.select(q="id:unicode", fl="*", **TYPES)
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "unicode")

    doc = docs_by_id(result)["unicode"]
    assert "𝄞" in doc["t_huge"]
    assert "👨‍👩‍👧‍👦" in doc["t_huge"]
    assert doc["t_strings"][0] == "🎬"
    assert doc["t_strings"][3] == "\U0001f600" * 50


def test_binary_field_decodes_to_bytes(solr: SolrProbe) -> None:
    """A Solr binary field arrives as BYTEARR and must decode to bytes, matching
    the base64 the JSON writer emits."""
    data, ref = solr.select(q="id:blob_*", rows="10", fl="id,t_blob", **TYPES)
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "binary fields")

    docs = docs_by_id(result)
    # As with empty strings, Solr drops a zero-length binary value instead of
    # storing it, so a zero-length BYTEARR only occurs in the encoder tests.
    assert "t_blob" not in docs["blob_empty"]
    assert docs["blob_one"]["t_blob"] == b"\x00"
    assert docs["blob_256"]["t_blob"] == bytes(range(256))
    assert len(docs["blob_big"]["t_blob"]) == 256 * 300
    assert all(
        isinstance(doc["t_blob"], bytes)
        for name, doc in docs.items()
        if name != "blob_empty"
    )


def test_absent_fields_are_absent(solr: SolrProbe) -> None:
    """Fields with no value are omitted, not decoded as None."""
    data, ref = solr.select(q="id:all_absent", fl="*", **TYPES)
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "absent fields")

    doc = docs_by_id(result)["all_absent"]
    for field in ("t_int", "t_long", "t_float", "t_double", "t_date", "t_blob"):
        assert field not in doc


def test_dates_across_the_epoch(solr: SolrProbe) -> None:
    """DATE is signed millis: pre-1970 values are negative, and fractional
    seconds must survive."""
    data, ref = solr.select(
        q="id:date_fractions OR id:multi_many", rows="10", fl="*", **TYPES
    )
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "dates")

    docs = docs_by_id(result)
    assert docs["date_fractions"]["t_date"] == solr_date_to_millis(
        "2024-06-01T12:34:56.001Z"
    )
    assert docs["multi_many"]["t_dates"][0] < 0  # 1969-12-31T23:59:59.999Z
    assert docs["multi_many"]["t_dates"][1] == 1


def test_multivalued_cardinalities(solr: SolrProbe) -> None:
    """Multi-valued fields with one and with many values are both lists."""
    data, ref = solr.select(q="id:multi_*", rows="10", fl="*", **TYPES)
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "multi-valued")

    docs = docs_by_id(result)
    assert docs["multi_one"]["t_strings"] == ["only"]
    assert len(docs["multi_many"]["t_strings"]) == 50


def test_child_documents(solr: SolrProbe) -> None:
    """The ``[child]`` transformer returns documents nested inside documents.

    With the ``_default`` configset (which defines ``_nest_path_``) Solr returns
    children under the field they were indexed on, two levels deep here, encoded
    as an ARR of SOLRDOC. javapyn's ``_childDocuments_`` key is the *anonymous*
    child-document shape, which this schema never produces; the reference-encoder
    tests cover that one.
    """
    data, ref = solr.request(
        "select", {"q": "id:parent", "fl": "*,[child]"}, coll=TYPES_COLLECTION
    )
    result = javabin.deserialize(data)
    assert_matches_json(result, ref, "child documents")

    parent = result["response"]["docs"][0]
    assert parent["id"] == "parent"
    children = parent["children"]
    assert {child["id"] for child in children} == {"child_1", "child_2"}
    grandchildren = next(c for c in children if c["id"] == "child_1")["grandchildren"]
    assert [g["id"] for g in grandchildren] == ["grandchild_1"]
    assert grandchildren[0]["t_int"] == 111


def test_child_documents_through_every_entry_point(solr: SolrProbe) -> None:
    """A nested response must decode the same way through all five entry points.

    The streaming decoders emit one callback per *top-level* document, with the
    children nested inside it -- not one callback per document in the index. The
    columnar decoders skip the nesting field (it has no flat column type) and
    still land on the following field: regression test for a skip that consumed
    only half of a child field list and then read later columns from inside a
    child document, silently, with no error.
    """
    data, _ = solr.request(
        "select",
        {"q": "t_string:parent", "fl": "id,t_int,children,[child]", "sort": "id asc"},
        coll=TYPES_COLLECTION,
    )
    parents = javabin.deserialize(data)["response"]["docs"]
    assert len(parents) == 1
    assert len(parents[0]["children"]) == 2

    streamed: list = []
    javabin.deserialize_stream(data, streamed.append)
    assert streamed == parents

    collected: list = []
    decoder = javabin.StreamDecoder()
    for i in range(0, len(data), 7):  # small chunks: children span boundaries
        decoder.feed(data[i : i + 7], collected.append)
    decoder.finish()
    assert collected == parents
    assert decoder.count == 1

    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([("id", pa.string()), ("t_int", pa.int32())])

    batch = javabin.deserialize_arrow(data, schema)
    assert batch.to_pydict() == {"id": ["parent"], "t_int": [1]}

    arrow_decoder = javabin.ArrowStreamDecoder(schema, batch_size=2)
    batches = [
        b for i in range(0, len(data), 7) for b in arrow_decoder.feed(data[i : i + 7])
    ]
    batches.extend(arrow_decoder.finish())
    table = pa.Table.from_batches(batches, schema=schema)
    assert table.to_pydict() == {"id": ["parent"], "t_int": [1]}


def test_child_documents_explode_into_arrow_rows(solr: SolrProbe) -> None:
    """``children="explode"`` turns every nested document into a row of its own.

    The live check that matters is the linking: ``_parent_id`` must name the
    *direct* parent even though Solr writes a document's own fields around its
    child list, and the depth must reflect the real nesting -- ``grandchild_1``
    hangs off ``child_1``, not off ``parent``.
    """
    pa = pytest.importorskip("pyarrow")

    # fl=* is needed for the *second* level: naming `children` explicitly returns
    # the chapters but not their own nested field, so the grandchild would be
    # missing from the response rather than from the decoding.
    data, _ = solr.request(
        "select",
        {"q": "t_string:parent", "fl": "*,[child]", "sort": "id asc"},
        coll=TYPES_COLLECTION,
    )
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("t_int", pa.int32()),
            ("_parent_id", pa.string()),
            ("_depth", pa.int32()),
            ("_child_field", pa.string()),
        ]
    )

    exploded = javabin.deserialize_arrow(data, schema, children="explode").to_pydict()

    assert exploded["id"] == ["parent", "child_1", "grandchild_1", "child_2"]
    assert exploded["_parent_id"] == [None, "parent", "child_1", "parent"]
    assert exploded["_depth"] == [0, 1, 2, 1]
    assert exploded["_child_field"] == [None, "children", "grandchildren", "children"]
    assert exploded["t_int"] == [1, 11, 111, 12]

    # Every link must point at a document that really is in the table, and every
    # top-level row must be a document Solr returned as a hit.
    ids = set(exploded["id"])
    assert all(parent in ids for parent in exploded["_parent_id"] if parent is not None)
    top_level = [
        doc_id
        for doc_id, depth in zip(exploded["id"], exploded["_depth"])
        if depth == 0
    ]
    assert top_level == ["parent"]

    # The default still yields one row, and the streaming decoder agrees with
    # the single-shot one even when children span chunk boundaries.
    assert javabin.deserialize_arrow(data, schema).num_rows == 1

    decoder = javabin.ArrowStreamDecoder(schema, batch_size=2, children="explode")
    batches = [b for i in range(0, len(data), 7) for b in decoder.feed(data[i : i + 7])]
    batches.extend(decoder.finish())
    assert pa.Table.from_batches(batches, schema=schema).to_pydict() == exploded


def test_explode_is_a_no_op_for_flat_responses(solr: SolrProbe) -> None:
    """``/export`` already returns children as documents of their own, so the
    mode changes nothing there -- the row count stays the document count."""
    pa = pytest.importorskip("pyarrow")

    data, _ = solr.export(coll=TYPES_COLLECTION)
    schema = pa.schema([("id", pa.string()), ("_depth", pa.int32())])

    exploded = javabin.deserialize_arrow(data, schema, children="explode")

    assert exploded.num_rows == TYPES_DOC_COUNT
    assert set(exploded.column("_depth").to_pylist()) == {0}
    assert exploded.to_pydict() == javabin.deserialize_arrow(data, schema).to_pydict()


def test_export_flattens_child_documents(solr: SolrProbe) -> None:
    """``/export`` has no ``[child]`` transformer, so Solr returns children as
    documents of their own.

    Worth pinning: the same collection yields 1 document from ``/select`` with
    ``[child]`` and 21 from ``/export``, and only ``_root_`` ties a child back to
    its parent. Nothing to nest here -- but it is the behaviour callers have to
    plan for, and it must stay consistent across the entry points.
    """
    data, ref = solr.export(coll=TYPES_COLLECTION)

    docs = javabin.deserialize(data)["response"]["docs"]
    assert len(docs) == TYPES_DOC_COUNT
    assert_matches_json(javabin.deserialize(data), ref, "types export")
    assert not any("children" in doc for doc in docs)
    assert {"parent", "child_1", "child_2", "grandchild_1"} <= {d["id"] for d in docs}

    streamed: list = []
    javabin.deserialize_stream(data, streamed.append)
    assert streamed == docs


# -- search component response shapes ----------------------------------------

COMPONENTS: dict[str, dict[str, object]] = {
    "facet_field": {
        "facet": "true",
        "facet.field": "t_string",
        "facet.limit": "-1",
        "rows": "0",
    },
    # The bucket for documents missing the field has a *null* NamedList name.
    "facet_missing": {
        "facet": "true",
        "facet.field": "t_string",
        "facet.missing": "true",
        "facet.limit": "-1",
        "rows": "0",
    },
    "facet_range": {
        "facet": "true",
        "facet.range": "t_int",
        "f.t_int.facet.range.start": "-10",
        "f.t_int.facet.range.end": "200",
        "f.t_int.facet.range.gap": "50",
        "facet.range.other": "all",
        "rows": "0",
    },
    "facet_query": {"facet": "true", "facet.query": "t_bool:true", "rows": "0"},
    "facet_pivot": {
        "facet": "true",
        "facet.pivot": "t_bool,t_string",
        "facet.pivot.mincount": "0",
        "rows": "0",
    },
    "facet_interval": {
        "facet": "true",
        "facet.interval": "t_int",
        "f.t_int.facet.interval.set": "[0,100]",
        "rows": "0",
    },
    "json_facet": {
        "json.facet": (
            '{terms:{type:terms,field:t_string,limit:3,facet:{avg:"avg(t_int)"}},'
            'q:{type:query,q:"*:*",facet:{s:"sum(t_double)"}}}'
        ),
        "rows": "0",
    },
    "stats": {
        "stats": "true",
        "stats.field": "t_int",
        "rows": "0",
    },
    "stats_percentiles": {
        "stats": "true",
        "stats.field": "{!calcdistinct=true percentiles='1,50,99'}t_int",
        "rows": "0",
    },
    "grouping": {
        "group": "true",
        "group.field": "t_string",
        "group.limit": "3",
        "group.ngroups": "true",
    },
    "grouping_simple": {
        "group": "true",
        "group.field": "t_string",
        "group.format": "simple",
    },
    "grouping_func": {"group": "true", "group.func": "add(t_int,1)"},
    "debug_all": {"debug": "all", "rows": "3"},
    "cursor_mark": {"cursorMark": "*", "rows": "5"},
    "collapse_expand": {
        "fq": "{!collapse field=t_string nullPolicy=expand}",
        "expand": "true",
    },
    "highlighting": {
        "q": "t_text:quick",
        "hl": "true",
        "hl.fl": "t_text",
        "rows": "5",
    },
    "score_and_functions": {
        "fl": "id,score,add(t_int,1),mul(t_double,2)",
        "rows": "5",
    },
    "rows_zero": {"rows": "0"},
    "empty_result": {"q": "id:no_such_document"},
}


#: Components whose section order is not stable even between two requests to the
#: same Solr with the same writer, because it comes out of an unordered Java map.
#: On Solr 8 the collapse component's ``expanded`` section reorders per request.
UNORDERED_COMPONENTS = frozenset({"collapse_expand"})


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_response_shape_matches_json(solr: SolrProbe, name: str) -> None:
    """Each search component builds its own NamedList/Map/SolrDocumentList shape;
    every one must decode to exactly what wt=json reports."""
    params = {"q": "*:*", "sort": "id asc", **COMPONENTS[name]}
    data, ref = solr.request("select", params, coll=TYPES_COLLECTION)

    assert_matches_json(
        javabin.deserialize(data),
        ref,
        name,
        check_order=name not in UNORDERED_COMPONENTS,
    )


def test_terms_component(solr: SolrProbe) -> None:
    data, ref = solr.request(
        "terms",
        {"terms": "true", "terms.fl": "t_string", "terms.limit": "-1"},
        coll=TYPES_COLLECTION,
    )
    assert_matches_json(javabin.deserialize(data), ref, "terms")


@pytest.mark.parametrize(
    "expr_name,expr",
    [
        (
            "search",
            'search(solr_types, q="*:*", fl="id,t_int", sort="id asc", rows=50)',
        ),
        (
            "rollup",
            'rollup(sort(search(solr_types, q="*:*", fl="t_string,t_int", '
            'sort="t_string asc", qt="/export"), by="t_string asc"), over="t_string", '
            "count(*), sum(t_int), avg(t_int), min(t_int), max(t_int))",
        ),
        (
            "facet",
            'facet(solr_types, q="*:*", buckets="t_string", '
            'bucketSorts="count(*) desc", bucketSizeLimit=10, count(*), avg(t_int))',
        ),
        (
            "stats",
            'stats(solr_types, q="*:*", count(*), sum(t_int), min(t_int), max(t_double))',
        ),
        ("tuple", 'tuple(a=1, b="x", c=1.5)'),
        ("echo", 'echo("hello world")'),
        (
            "empty",
            'search(solr_types, q="id:nope", fl="id", sort="id asc", qt="/export")',
        ),
        (
            "expression_error",
            'search(solr_types, q="*:*", fl="no_such_field_zz", '
            'sort="no_such_field_zz asc", qt="/export")',
        ),
    ],
)
def test_streaming_expression_shapes(
    solr: SolrProbe, expr_name: str, expr: str
) -> None:
    """Streaming expressions return tuples, buckets, stats and error markers
    through the same ITERATOR encoding; all of them must match wt=json."""
    data, ref = solr.stream(expr, coll=TYPES_COLLECTION)
    assert_matches_json(javabin.deserialize(data), ref, f"stream {expr_name}")


ERROR_CASES: dict[str, dict[str, object]] = {
    "bad_query_syntax": {"q": ":::"},
    "unknown_field": {"q": "no_such_field:1"},
    "bad_sort_field": {"q": "*:*", "sort": "no_such_field asc"},
    "bad_json_facet": {"q": "*:*", "json.facet": "{broken"},
}


@pytest.mark.parametrize("name", sorted(ERROR_CASES))
def test_error_responses_decode(solr: SolrProbe, name: str) -> None:
    """A Solr error is a javabin response too (HTTP 400 with an error
    NamedList); it must decode and match wt=json."""
    data, ref = solr.request("select", ERROR_CASES[name], coll=TYPES_COLLECTION)

    result = javabin.deserialize(data)
    assert result["responseHeader"]["status"] == 400
    assert "error" in result
    assert_matches_json(result, ref, name)


# -- framing cross-check against an independent scanner ----------------------


def collect_corpus(solr: SolrProbe) -> dict[str, bytes]:
    """Real javabin responses covering the shapes this suite exercises."""
    corpus: dict[str, bytes] = {}
    for name, params in COMPONENTS.items():
        corpus[f"select_{name}"] = solr.raw(
            "select",
            {"q": "*:*", "sort": "id asc", "wt": "javabin", **params},
            coll=TYPES_COLLECTION,
        )
    for name, params in ERROR_CASES.items():
        corpus[f"error_{name}"] = solr.raw(
            "select", {**params, "wt": "javabin"}, coll=TYPES_COLLECTION
        )
    corpus["select_all_types"] = solr.raw(
        "select",
        {"q": "*:*", "rows": "100", "fl": "*", "sort": "id asc", "wt": "javabin"},
        coll=TYPES_COLLECTION,
    )
    corpus["select_children"] = solr.raw(
        "select",
        {"q": "id:parent", "fl": "*,[child]", "wt": "javabin"},
        coll=TYPES_COLLECTION,
    )
    corpus["export_types"] = solr.export(coll=TYPES_COLLECTION)[0]
    corpus["export_movies"] = solr.export()[0]
    corpus["export_empty"] = solr.export(q="movie_id:nope")[0]
    corpus["stream_search"] = solr.stream(
        'search(solr_types, q="*:*", fl="id,t_int", sort="id asc", rows=50)',
        coll=TYPES_COLLECTION,
    )[0]
    corpus["schema"] = solr.raw("schema", {"wt": "javabin"}, coll=TYPES_COLLECTION)
    corpus["luke"] = solr.raw(
        "admin/luke", {"show": "all", "wt": "javabin"}, coll=TYPES_COLLECTION
    )
    # Handlers come and go between Solr majors; a 404 answers with an HTML page
    # rather than javabin, which is not this suite's business.
    return {
        name: data
        for name, data in corpus.items()
        if data[:1] == b"\x02"  # the javabin v2 version byte
    }


@pytest.fixture(scope="module")
def corpus(solr: SolrProbe) -> dict[str, bytes]:
    return collect_corpus(solr)


def test_independent_scanner_agrees_on_framing(corpus: dict[str, bytes]) -> None:
    """An independent pure-Python framing scanner must consume every response
    exactly, ending where the top-level value ends.

    Two implementations written from the same spec agreeing on the byte extent of
    every value in every real response is a much stronger statement than either
    one accepting the input alone: a framing bug in the Rust reader (a wrong
    size, a missed vint continuation) would leave bytes over here.
    """
    for name, data in corpus.items():
        javabin.deserialize(data)  # must decode ...
        try:
            scan(data)  # ... and must frame identically
        except ScanError as exc:
            pytest.fail(f"{name}: independent scanner disagrees: {exc}")


def test_live_responses_exercise_the_expected_tags(corpus: dict[str, bytes]) -> None:
    """Assert what the live suite actually covers, so shrinking coverage is a
    test failure rather than a silent regression.

    The tags Solr never emits in a query response -- BYTE, SHORT, MAP_ENTRY,
    ENUM_FIELD_VALUE, PRIMITIVE_ARR, SOLRINPUTDOC, UUID -- stay covered by the
    Rust unit tests and the reference encoder only. Notably a Solr UUIDField
    serialises as a plain STR, so javabin's UUID tag is not reachable this way.
    """
    seen: set[str] = set()
    for data in corpus.values():
        seen |= set(scan(data))

    required = {
        "NULL",
        "BOOL_TRUE",
        "BOOL_FALSE",
        "DOUBLE",
        "INT",
        "LONG",
        "FLOAT",
        "DATE",
        "MAP",
        "SOLRDOC",
        "SOLRDOCLST",
        "BYTEARR",
        "ITERATOR",
        "END",
        "MAP_ENTRY_ITER",
        "STR",
        "SINT",
        "SLONG",
        "ARR",
        "ORDERED_MAP",
        "NAMED_LST",
        "EXTERN_STRING",
    }
    assert required <= seen, f"live coverage lost tags: {sorted(required - seen)}"
    assert seen <= ALL_TAGS


# -- robustness against damaged real bytes -----------------------------------


def test_every_truncation_of_a_real_response_raises_value_error(
    corpus: dict[str, bytes],
) -> None:
    """Every prefix of every real response must raise ValueError -- never a
    panic, a wrong exception type, or a silent partial result.

    Truncation is what a dropped connection looks like, so this is the failure
    mode most likely to happen in production.
    """
    checked = 0
    for name, data in corpus.items():
        cuts = (
            range(1, len(data))
            if len(data) <= 2000
            else sorted(random.Random(len(data)).sample(range(1, len(data)), 200))
        )
        # What StreamDecoder yields from the whole response is the reference for
        # what a prefix of it may yield: grouped and expanded responses carry
        # documents that are not under response.docs at all.
        stream_reference: list = []
        reference_decoder = javabin.StreamDecoder()
        reference_decoder.feed(data, stream_reference.append)
        reference_decoder.finish()
        for cut in cuts:
            prefix = data[:cut]
            checked += 1
            for label, call in (
                ("deserialize", lambda p=prefix: javabin.deserialize(p)),
                ("deserialize_json", lambda p=prefix: javabin.deserialize_json(p)),
                (
                    "deserialize_stream",
                    lambda p=prefix: javabin.deserialize_stream(p, lambda d: None),
                ),
            ):
                try:
                    call()
                except ValueError:
                    continue
                except Exception as exc:  # noqa: BLE001 - the point of the test
                    pytest.fail(
                        f"{name}: {label} raised {type(exc).__name__} on a "
                        f"{cut}-byte prefix: {exc}"
                    )
                pytest.fail(f"{name}: {label} accepted a {cut}-byte prefix")

            # StreamDecoder is fed incrementally, so a prefix that already
            # contains the whole document sequence may legitimately complete;
            # what it must never do is invent or lose documents.
            collected: list = []
            decoder = javabin.StreamDecoder()
            try:
                decoder.feed(prefix, collected.append)
                decoder.finish()
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"{name}: StreamDecoder raised {type(exc).__name__} on a "
                    f"{cut}-byte prefix: {exc}"
                )
            assert collected == stream_reference[: len(collected)], (
                f"{name}: StreamDecoder produced non-prefix documents from a "
                f"{cut}-byte prefix"
            )
    assert checked > 2000


def test_corrupted_real_responses_never_crash(corpus: dict[str, bytes]) -> None:
    """Random byte mutations of real responses must raise ``ValueError`` or
    decode cleanly -- nothing else.

    The documented contract is that malformed javabin raises ``ValueError``, so
    any other exception type is a failure: a Rust panic surfacing as
    ``pyo3_runtime.PanicException``, or the ``TypeError`` that used to escape when
    a mutation turned a map key into a container (see
    ``test_decoder.py::test_unhashable_map_key_raises_value_error``). A segfault
    would take the whole test session down with it.
    """
    rng = random.Random(20240817)
    mutations = 0
    for name, data in corpus.items():
        for _ in range(60):
            buf = bytearray(data)
            for _ in range(rng.randint(1, 4)):
                buf[rng.randrange(len(buf))] = rng.randrange(256)
            mutated = bytes(buf)
            mutations += 1
            for label, call in (
                ("deserialize", lambda m=mutated: javabin.deserialize(m)),
                ("deserialize_json", lambda m=mutated: javabin.deserialize_json(m)),
                (
                    "deserialize_stream",
                    lambda m=mutated: javabin.deserialize_stream(m, lambda d: None),
                ),
            ):
                try:
                    call()
                except (ValueError, RecursionError, MemoryError):
                    pass
                except Exception as exc:  # noqa: BLE001 - the point of the test
                    pytest.fail(
                        f"{name}: {label} raised {type(exc).__name__} on a mutated "
                        f"response: {exc}"
                    )
    assert mutations > 1000


def test_decoders_agree_on_every_real_response(corpus: dict[str, bytes]) -> None:
    """deserialize, deserialize_json, deserialize_stream and StreamDecoder must
    agree on every real response."""
    for name, data in corpus.items():
        obj = javabin.deserialize(data)
        # Decoding is deterministic (NaN never compares equal to itself).
        assert _has_nan(obj) or javabin.deserialize(data) == obj, name

        assert_json_path_matches(obj, json.loads(javabin.deserialize_json(data)), name)

        docs = _docs_of(obj)
        if docs is None:
            continue
        streamed: list = []
        javabin.deserialize_stream(data, streamed.append)
        assert streamed == docs, f"{name}: deserialize_stream disagrees"

        collected: list = []
        decoder = javabin.StreamDecoder()
        for i in range(0, len(data), 997):
            decoder.feed(data[i : i + 997], collected.append)
        decoder.finish()
        assert collected == docs, f"{name}: StreamDecoder disagrees"


def _docs_of(obj: object) -> list | None:
    """The document list of a /select, /export or /stream response, if it has one."""
    if not isinstance(obj, dict):
        return None
    for envelope_key in ("response", "result-set"):
        envelope = obj.get(envelope_key)
        if isinstance(envelope, dict):
            docs = envelope.get("docs")
            if isinstance(docs, list):
                return docs
    return None


def _has_nan(obj: object) -> bool:
    if isinstance(obj, float):
        return math.isnan(obj)
    if isinstance(obj, dict):
        return any(_has_nan(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nan(v) for v in obj)
    return False


# -- defects this suite found -------------------------------------------------


def test_facet_missing_bucket_decodes(solr: SolrProbe) -> None:
    """``facet.missing=true`` adds a bucket whose NamedList *name* is null.

    Solr writes the name as a NULL tag and its own reader tolerates it, so the
    bucket lands under the ``None`` key. Regression test for a defect that made
    any faceted query with ``facet.missing`` undecodable on Solr 8.11, 9.10 and
    10.0, for string and numeric fields, with both facet methods.
    """
    data, ref = solr.request(
        "select",
        {
            "q": "*:*",
            "rows": "0",
            "facet": "true",
            "facet.field": "t_string",
            "facet.missing": "true",
            "facet.limit": "-1",
        },
        coll=TYPES_COLLECTION,
    )

    result = javabin.deserialize(data)
    counts = result["facet_counts"]["facet_fields"]["t_string"]
    # The reference renders the NamedList flat, ending with [..., null, <count>].
    ref_counts = ref["facet_counts"]["facet_fields"]["t_string"]
    assert ref_counts[-2] is None
    assert counts[None] == ref_counts[-1]


def test_export_error_is_visible_to_streaming_decoders(solr: SolrProbe) -> None:
    """A failed ``/export`` must not look like an empty one.

    Without a sort parameter Solr answers HTTP 400 with
    ``response.docs[0].EXCEPTION``, encoding ``response`` as a plain MAP holding
    an ARR instead of the MAP_ENTRY_ITER + ITERATOR shape of a successful export.
    Regression test for a defect where every streaming path skipped that shape,
    so a caller that did not check the HTTP status could not tell a failed export
    from an empty result.
    """
    data = solr.raw(
        "export",
        {"q": "*:*", "fl": "id", "wt": "javabin"},
        coll=TYPES_COLLECTION,
        method="POST",
    )

    full = javabin.deserialize(data)
    assert "EXCEPTION" in full["response"]["docs"][0]

    streamed: list = []
    envelope = javabin.deserialize_stream(data, streamed.append)
    assert streamed == full["response"]["docs"]
    assert envelope["response"]["docs"] == []

    collected: list = []
    decoder = javabin.StreamDecoder()
    decoder.feed(data, collected.append)
    decoder.finish()
    assert collected == full["response"]["docs"]
    assert decoder.count == 1


def test_deserialize_json_keeps_non_finite_doubles(solr: SolrProbe) -> None:
    """``deserialize_json`` must not lose infinities.

    Reachable from ordinary queries: summing the double extremes overflows in the
    stats component, and a function query over them does too. Regression test for
    a defect where serde_json rendered them as ``null``, so the JSON path
    disagreed with ``deserialize`` despite promising the same shape. Both now
    agree with Solr, which writes the string ``"Infinity"``.
    """
    data, ref = solr.request(
        "select",
        {"q": "*:*", "rows": "0", "stats": "true", "stats.field": "t_double"},
        coll=TYPES_COLLECTION,
    )

    stats = javabin.deserialize(data)["stats"]["stats_fields"]["t_double"]
    assert math.isinf(stats["sumOfSquares"])
    assert ref["stats"]["stats_fields"]["t_double"]["sumOfSquares"] == "Infinity"

    via_json = json.loads(javabin.deserialize_json(data))
    assert via_json["stats"]["stats_fields"]["t_double"]["sumOfSquares"] == "Infinity"


# -- scale --------------------------------------------------------------------


def test_bulk_export_arrow_and_streaming_agree(solr: SolrProbe) -> None:
    """At 2 000 documents the object, streaming and columnar paths must produce
    the same values from the same real export."""
    pa = pytest.importorskip("pyarrow")

    data, _ = solr.export()
    docs = javabin.deserialize(data)["response"]["docs"]
    assert len(docs) == DOC_COUNT

    schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("view_count", pa.int64()),
            ("is_classic", pa.bool_()),
        ]
    )
    batch = javabin.deserialize_arrow(data, schema)
    assert batch.column("movie_id").to_pylist() == [d["movie_id"] for d in docs]
    assert batch.column("view_count").to_pylist() == [d["view_count"] for d in docs]
    assert batch.column("is_classic").to_pylist() == [d["is_classic"] for d in docs]

    decoder = javabin.ArrowStreamDecoder(schema, batch_size=256)
    batches = [b for chunk in solr.export_chunks() for b in decoder.feed(chunk)]
    batches.extend(decoder.finish())
    table = pa.Table.from_batches(batches, schema=schema)
    assert table.column("movie_id").to_pylist() == [d["movie_id"] for d in docs]
