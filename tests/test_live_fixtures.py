"""
Regression tests against real-world modeled ``javabin`` byte fixtures captured,
alongside the equivalent ``wt=json`` response.

These tests replay the generated ``wt=javabin`` byte responses captured from
several Solr collections (movie and studio datasets) and assert field-by-field
equality against a ``wt=json`` response captured in the same request batch.
"""

import json
import random as _random
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import javapyn as javabin

FIXTURES = Path(__file__).parent / "fixtures"

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _solr_date_to_millis(iso: str) -> int:
    """Parse a Solr ``tdate`` JSON string (``YYYY-MM-DDTHH:MM:SSZ``, optionally
    with fractional seconds) into milliseconds since the Unix epoch, matching
    the javabin ``DATE`` tag's representation.
    """
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in iso else "%Y-%m-%dT%H:%M:%SZ"
    dt = datetime.strptime(iso, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _load_fixture(name: str) -> tuple[bytes, dict]:
    data = (FIXTURES / f"{name}.bin").read_bytes()
    ref = json.loads((FIXTURES / f"{name}.json").read_text())
    return data, ref


def _assert_docs_match(got_docs: list[dict], ref_docs: list[dict]) -> None:
    assert len(got_docs) == len(ref_docs)
    for got, expected in zip(got_docs, ref_docs):
        for key, expected_value in expected.items():
            got_value = got.get(key)
            if isinstance(expected_value, str) and _ISO_DATE_RE.match(expected_value):
                assert got_value == _solr_date_to_millis(expected_value), key
            else:
                assert got_value == expected_value, key


def test_live_sample_matches_json_reference() -> None:
    data, ref = _load_fixture("solr_movies_deterministic")

    result = javabin.deserialize(data)

    ref_response = ref["response"]
    response = result["response"]

    assert response["numFound"] == ref_response["numFound"]
    assert response["start"] == ref_response["start"]
    assert response["maxScore"] == ref_response["maxScore"]
    assert response["numFoundExact"] == ref_response["numFoundExact"]
    _assert_docs_match(response["docs"], ref_response["docs"])

    # responseHeader (a plain NamedList) should also match, modulo QTime
    # (query timing) and wt (deliberately different between the two requests).
    assert result["responseHeader"]["status"] == ref["responseHeader"]["status"]
    got_params = dict(result["responseHeader"]["params"])
    ref_params = dict(ref["responseHeader"]["params"])
    got_params.pop("wt", None)
    ref_params.pop("wt", None)
    assert got_params == ref_params


def test_live_sample_deserialize_json_matches_deserialize() -> None:
    data, _ = _load_fixture("solr_movies_deterministic")

    assert json.loads(javabin.deserialize_json(data)) == javabin.deserialize(data)


def test_live_unsorted_sample_decodes_without_error() -> None:
    """
    A second, larger live sample used purely as a smoke test that decoding a substantially
    different real response doesn't raise.
    """
    data, ref = _load_fixture("solr_movies_sample")

    result = javabin.deserialize(data)

    assert result["response"]["numFound"] == ref["response"]["numFound"]
    assert len(result["response"]["docs"]) == len(ref["response"]["docs"])
    for doc in result["response"]["docs"]:
        assert "movie_id" in doc


def test_solr_movies_all_fields_matches_json_reference() -> None:
    """SOLR_MOVIES: every schema field, covering bool/int/tdate/string."""
    data, ref = _load_fixture("solr_movies_all_fields")
    result = javabin.deserialize(data)
    _assert_docs_match(result["response"]["docs"], ref["response"]["docs"])


def test_solr_studios_all_fields_matches_json_reference() -> None:
    """SOLR_STUDIOS: every schema field, several bool fields per document."""
    data, ref = _load_fixture("solr_studios_all_fields")
    result = javabin.deserialize(data)
    _assert_docs_match(result["response"]["docs"], ref["response"]["docs"])


def test_solr_reviewers_all_fields_matches_json_reference() -> None:
    """SOLR_REVIEWERS: every schema field."""
    data, ref = _load_fixture("solr_reviewers_all_fields")
    result = javabin.deserialize(data)
    _assert_docs_match(result["response"]["docs"], ref["response"]["docs"])


def test_stream_endpoint_result_set() -> None:
    """
    Solr's /stream (streaming expression) endpoint uses a different top-level
    shape than /select: a ``result-set`` NamedList wrapping a ``docs`` list
    that is javabin-encoded as an ITERATOR (unknown length, END-terminated)
    and ends with a synthetic ``{"EOF": true, "RESPONSE_TIME": ...}`` marker.
    """
    data, ref = _load_fixture("solr_movies_stream")
    result = javabin.deserialize(data)

    assert set(result.keys()) == {"result-set"}
    docs = result["result-set"]["docs"]
    ref_docs = ref["result-set"]["docs"]
    assert len(docs) == len(ref_docs)

    # Last entry is the EOF marker; RESPONSE_TIME is inherently per-request,
    # so compare it structurally only.
    *data_docs, eof = docs
    *ref_data_docs, ref_eof = ref_docs
    assert eof["EOF"] is True
    assert "RESPONSE_TIME" in eof

    _assert_docs_match(data_docs, ref_data_docs)


def test_export_endpoint_result_set() -> None:
    """
    Solr's /export handler encodes its response with the streaming
    ``MAP_ENTRY_ITER`` tag (a map of unknown length, END-terminated) rather
    than the fixed-size ``NamedList``/``SolrDocumentList`` used by /select,
    and represents ``docs`` as an ITERATOR. This fixture is an (empty)
    ``/export`` result — decoding it exercises the MAP_ENTRY_ITER + empty
    ITERATOR paths and must match the ``wt=json`` reference exactly.
    """
    data, ref = _load_fixture("solr_movies_export")
    result = javabin.deserialize(data)

    assert result == ref
    assert result["response"]["docs"] == ref["response"]["docs"]


def _docs_of(result: dict) -> list:
    """Extract the docs list from either a /select or /stream shaped result."""
    if "response" in result:
        return result["response"]["docs"]
    return result["result-set"]["docs"]


def test_deserialize_stream_matches_full_select() -> None:
    """Streaming a /select response yields exactly the same docs as a full
    decode, and the returned envelope keeps metadata but an empty docs list."""
    data, _ = _load_fixture("solr_movies_sample")

    full_docs = _docs_of(javabin.deserialize(data))

    streamed: list = []
    env = javabin.deserialize_stream(data, streamed.append)

    assert streamed == full_docs
    assert (
        env["response"]["numFound"] == javabin.deserialize(data)["response"]["numFound"]
    )
    assert env["response"]["docs"] == []


def test_deserialize_stream_matches_full_stream_endpoint() -> None:
    """Streaming a /stream (result-set/ITERATOR) response yields the same docs
    (including the trailing EOF marker) as a full decode."""
    data, _ = _load_fixture("solr_movies_stream")

    full_docs = _docs_of(javabin.deserialize(data))

    streamed: list = []
    javabin.deserialize_stream(data, streamed.append)

    assert streamed == full_docs
    assert streamed[-1]["EOF"] is True


def test_deserialize_stream_all_fields() -> None:
    """Streaming works for every-field documents across collections."""
    for name in (
        "solr_movies_all_fields",
        "solr_studios_all_fields",
        "solr_reviewers_all_fields",
    ):
        data, _ = _load_fixture(name)
        full_docs = _docs_of(javabin.deserialize(data))
        streamed: list = []
        javabin.deserialize_stream(data, streamed.append)
        assert streamed == full_docs, name


def test_deserialize_stream_callback_exception_propagates() -> None:
    """An exception raised inside the callback propagates unchanged."""
    data, _ = _load_fixture("solr_movies_sample")

    class Boom(Exception):
        pass

    def cb(_doc: object) -> None:
        raise Boom

    with pytest.raises(Boom):
        javabin.deserialize_stream(data, cb)


def _feed_chunked(data: bytes, chunk_sizes) -> list:
    """Feed `data` to a StreamDecoder in the given chunk sizes, collecting docs."""
    got: list = []
    dec = javabin.StreamDecoder()
    i = 0
    for n in chunk_sizes:
        dec.feed(data[i : i + n], got.append)
        i += n
    if i < len(data):
        dec.feed(data[i:], got.append)
    dec.finish()
    return got


@pytest.mark.parametrize(
    "name",
    [
        "solr_movies_sample",
        "solr_movies_stream",
        "solr_movies_export",
        "solr_movies_all_fields",
        "solr_studios_all_fields",
        "solr_reviewers_all_fields",
    ],
)
def test_stream_decoder_byte_by_byte_matches_full(name: str) -> None:
    """Feeding one byte at a time (worst-case chunking) yields exactly the same
    documents as the whole-buffer streaming decode."""
    data, _ = _load_fixture(name)

    ref: list = []
    javabin.deserialize_stream(data, ref.append)

    got = _feed_chunked(data, [1] * len(data))
    assert got == ref
    assert javabin.StreamDecoder is not None


@pytest.mark.parametrize("name", ["solr_movies_all_fields", "solr_movies_stream"])
def test_stream_decoder_random_chunking(name: str) -> None:
    data, _ = _load_fixture(name)
    ref: list = []
    javabin.deserialize_stream(data, ref.append)

    for seed in range(15):
        rng = _random.Random(seed)
        sizes = []
        remaining = len(data)
        while remaining > 0:
            n = min(remaining, rng.randint(1, 97))
            sizes.append(n)
            remaining -= n
        assert _feed_chunked(data, sizes) == ref, f"seed {seed}"


def test_stream_decoder_count_and_callback_error() -> None:
    data, _ = _load_fixture("solr_movies_all_fields")

    # count attribute
    dec = javabin.StreamDecoder()
    dec.feed(data, lambda d: None)
    dec.finish()
    assert dec.count == 5

    # callback exception propagates
    class Boom(Exception):
        pass

    dec = javabin.StreamDecoder()
    with pytest.raises(Boom):
        dec.feed(data, lambda d: (_ for _ in ()).throw(Boom()))
