"""
Conformance tests against a real Apache Solr in a container.

Every test here fetches the *same* query twice -- ``wt=javabin`` and ``wt=json``
-- and asserts the decoded javabin equals what Solr itself says the result is.
The reference is therefore Solr's own encoder, not our reading of the spec:
these are the only tests in the repo that would catch a shared blind spot
between ``javabin_ref_encoder.py`` and the decoder, or an encoder change in a
new Solr release.

They complement, not replace, the fixture tests. What only a live Solr gives us:
``/export`` with actual documents (the committed fixture is a 68-byte *empty*
result), a response large enough to cross real HTTP chunk boundaries, and an
EXTERN_STRING table filled by the real encoder over thousands of documents.

Requires Docker; deselected by default. Run with ``pytest -m solr``.
"""

import json

import pytest
from javabin_compare import assert_docs_match, assert_json_path_matches
from solr_probe import COLLECTION, DOC_COUNT, EXPORT_FIELDS, SolrProbe

import javapyn as javabin

pytestmark = pytest.mark.solr

SELECT_ROWS = "50"

STREAM_EXPR = (
    f'search({COLLECTION}, q="*:*", fl="movie_id,rating,genres,box_office", '
    'sort="movie_id asc", qt="/export")'
)


def test_select_matches_json_reference(solr: SolrProbe) -> None:
    """/select with fl=* -- every field of every document, plus the
    SolrDocumentList metadata."""
    data, ref = solr.select(rows=SELECT_ROWS, fl="*")

    result = javabin.deserialize(data)
    response = result["response"]
    ref_response = ref["response"]

    assert response["numFound"] == ref_response["numFound"] == DOC_COUNT
    assert response["start"] == ref_response["start"]
    assert response["numFoundExact"] == ref_response["numFoundExact"]
    # javabin's SolrDocumentList always carries a maxScore slot; wt=json omits
    # the key entirely when sorting by a field. Both mean "no score".
    assert response.get("maxScore") == ref_response.get("maxScore")

    assert len(response["docs"]) == int(SELECT_ROWS)
    assert_docs_match(response["docs"], ref_response["docs"])

    assert result["responseHeader"]["status"] == ref["responseHeader"]["status"]


def test_export_with_documents_matches_json_reference(solr: SolrProbe) -> None:
    """/export over a non-empty result set.

    The committed ``solr_movies_export.bin`` fixture is an empty result, so this
    is the first time the real encoder's MAP_ENTRY_ITER envelope wrapping a
    non-empty ITERATOR of documents is exercised.
    """
    data, ref = solr.export()

    result = javabin.deserialize(data)
    docs = result["response"]["docs"]

    # Guard against passing on an empty result -- exactly how the fixture-based
    # export coverage was silently vacuous.
    assert len(docs) == DOC_COUNT
    assert docs[0]["movie_id"] == "mv-00000"
    assert len(data) > 100_000

    assert result["response"]["numFound"] == ref["response"]["numFound"]
    assert_docs_match(docs, ref["response"]["docs"])


def test_stream_expression_matches_json_reference(solr: SolrProbe) -> None:
    """/stream -- result-set/ITERATOR shape plus the synthetic EOF marker."""
    data, ref = solr.stream(STREAM_EXPR)

    result = javabin.deserialize(data)
    assert set(result.keys()) == {"result-set"}

    docs = result["result-set"]["docs"]
    ref_docs = ref["result-set"]["docs"]
    assert len(docs) == len(ref_docs) == DOC_COUNT + 1  # docs + EOF marker

    *data_docs, eof = docs
    *ref_data_docs, _ = ref_docs
    assert eof["EOF"] is True
    # RESPONSE_TIME is per-request, so it is only checked structurally.
    assert "RESPONSE_TIME" in eof

    assert_docs_match(data_docs, ref_data_docs)


@pytest.mark.parametrize("handler", ["select", "export", "stream"])
def test_deserialize_json_matches_deserialize(solr: SolrProbe, handler: str) -> None:
    """The direct-to-JSON path agrees with the object path on real bytes from
    all three handlers."""
    if handler == "select":
        data, _ = solr.select(rows=SELECT_ROWS, fl="*")
    elif handler == "export":
        data, _ = solr.export()
    else:
        data, _ = solr.stream(STREAM_EXPR)

    assert_json_path_matches(
        javabin.deserialize(data), json.loads(javabin.deserialize_json(data)), handler
    )


def test_deserialize_stream_matches_full_decode(solr: SolrProbe) -> None:
    """Streaming a real /export yields the same documents as a full decode, and
    keeps the envelope metadata."""
    data, _ = solr.export()

    full_docs = javabin.deserialize(data)["response"]["docs"]

    streamed: list = []
    envelope = javabin.deserialize_stream(data, streamed.append)

    assert streamed == full_docs
    assert envelope["response"]["numFound"] == DOC_COUNT
    assert envelope["response"]["docs"] == []


def test_stream_decoder_over_real_http_chunks(solr: SolrProbe) -> None:
    """Feed the /export body to StreamDecoder exactly as the network delivers
    it.

    The fixture tests only ever chunk synthetically (byte-by-byte and random
    splits). This asserts the decoder resumes correctly across the boundaries a
    real HTTP response actually produces -- and that the body is big enough for
    there to be more than one.
    """
    data, _ = solr.export()
    expected = javabin.deserialize(data)["response"]["docs"]

    got: list = []
    decoder = javabin.StreamDecoder()
    chunks = 0
    for chunk in solr.export_chunks():
        chunks += 1
        decoder.feed(chunk, got.append)
    decoder.finish()

    assert chunks > 1, "response arrived in a single chunk; no boundary was tested"
    assert decoder.count == DOC_COUNT
    assert got == expected


def test_deserialize_arrow_matches_json_reference(solr: SolrProbe) -> None:
    """The columnar path decodes real encoder bytes into the expected Arrow
    columns."""
    pa = pytest.importorskip("pyarrow")

    data, ref = solr.export()

    schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("title", pa.string()),
            ("rating", pa.float32()),
            ("box_office", pa.float64()),
            ("release_year", pa.int32()),
            ("view_count", pa.int64()),
            ("is_classic", pa.bool_()),
            ("genres", pa.list_(pa.string())),
            ("last_updated", pa.timestamp("ms")),
        ]
    )

    batch = javabin.deserialize_arrow(data, schema)

    assert batch.num_rows == DOC_COUNT
    assert batch.schema.names == EXPORT_FIELDS.split(",")

    columns = batch.to_pydict()
    ref_docs = ref["response"]["docs"]
    for field in ("movie_id", "title", "view_count", "is_classic", "genres"):
        assert columns[field] == [doc[field] for doc in ref_docs], field


def test_arrow_stream_decoder_over_real_http_chunks(solr: SolrProbe) -> None:
    """Batches assembled from real network chunks cover every document exactly
    once."""
    pa = pytest.importorskip("pyarrow")

    schema = pa.schema([("movie_id", pa.string()), ("rating", pa.float32())])
    decoder = javabin.ArrowStreamDecoder(schema, batch_size=256)

    batches = []
    for chunk in solr.export_chunks():
        batches.extend(decoder.feed(chunk))
    batches.extend(decoder.finish())

    table = pa.Table.from_batches(batches, schema=schema)
    assert table.num_rows == DOC_COUNT
    assert table.column("movie_id").to_pylist()[0] == "mv-00000"
    assert table.column("movie_id").to_pylist()[-1] == f"mv-{DOC_COUNT - 1:05d}"
