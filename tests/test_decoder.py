"""
Round-trip tests for the javabin decoder against an independent reference
encoder (``javabin_ref_encoder``), built directly from the Apache Solr
``JavaBinCodec`` write-side logic.

These tests don't require network access to a live Solr instance: they
construct a realistic response shape modeled on the ``SOLR_MOVIES`` collection
schema (string/int/long/float/bool/date fields, multi-valued arrays, null
fields, and a nested child document) and verify that ``javabin.deserialize``
and ``javabin.deserialize_json`` reproduce the original structure.
"""

import json

import pytest
from javabin_ref_encoder import NamedList, SolrDoc, SolrDocList, encode

import javapyn as javabin


def test_scalars_roundtrip() -> None:
    assert javabin.deserialize(encode(None)) is None
    assert javabin.deserialize(encode(True)) is True
    assert javabin.deserialize(encode(False)) is False
    assert javabin.deserialize(encode("hello")) == "hello"
    assert javabin.deserialize(encode("")) == ""
    assert javabin.deserialize(encode(0)) == 0
    assert javabin.deserialize(encode(5)) == 5
    assert javabin.deserialize(encode(-5)) == -5
    assert javabin.deserialize(encode(1000)) == 1000
    assert javabin.deserialize(encode(2**40)) == 2**40
    assert javabin.deserialize(encode(-(2**40))) == -(2**40)
    assert javabin.deserialize(encode(1.5)) == pytest.approx(1.5)
    assert javabin.deserialize(encode(b"\x00\x01\xff")) == b"\x00\x01\xff"


def test_unicode_string_roundtrip() -> None:
    s = "Straße äöü \u00e9 \U0001f600"
    assert javabin.deserialize(encode(s)) == s


def test_array_roundtrip() -> None:
    assert javabin.deserialize(encode([1, 2, 3])) == [1, 2, 3]
    assert javabin.deserialize(encode(["a", "b", "a"])) == ["a", "b", "a"]
    assert javabin.deserialize(encode([])) == []


def test_named_list_roundtrip() -> None:
    nl = NamedList([("status", 0), ("QTime", 12)])
    assert javabin.deserialize(encode(nl)) == {"status": 0, "QTime": 12}


def test_generic_map_roundtrip() -> None:
    m = {"a": 1, "b": [1, 2], "c": None}
    assert javabin.deserialize(encode(m)) == m


def test_repeated_field_names_use_string_cache() -> None:
    # Field names repeated across many docs exercise the EXTERN_STRING cache.
    docs = [SolrDoc(fields={"title": f"M{i}", "rating": 5.0 + i}) for i in range(5)]
    dl = SolrDocList(
        num_found=len(docs), start=0, max_score=None, num_found_exact=True, docs=docs
    )
    result = javabin.deserialize(encode(dl))

    assert result["numFound"] == len(docs)
    assert result["start"] == 0
    assert result["maxScore"] is None
    assert result["numFoundExact"] is True
    assert [d["title"] for d in result["docs"]] == ["M0", "M1", "M2", "M3", "M4"]
    assert [d["rating"] for d in result["docs"]] == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_solr_movies_like_response_roundtrip() -> None:
    """
    Build a response shaped like a real ``SOLR_MOVIES`` ``/query`` result:
    a top-level NamedList with ``responseHeader`` and ``response``
    (a SolrDocumentList), documents covering every relevant Solr field type,
    including a multi-valued field and a nested child document.
    """
    doc = SolrDoc(
        fields={
            "movie_id": "movie-101",
            "release_id": 123456789012,  # long
            "title": "Inception Trivia - Spinning Top 😀 äöü",  # string
            "rating": 8.8,  # float
            "runtime_minutes": 148,  # int
            "is_classic": False,  # bool
            "is_blockbuster": True,
            "last_updated": None,  # date, unset
            "genres": ["Action", "Sci-Fi", "Adventure"],  # multiValued string
            "description": "Spinning Top 😀 äöü",  # non-ASCII string
        },
        children=[
            SolrDoc(
                fields={
                    "movie_id": "movie-101-trivia-1",
                    "title": "Inception Trivia - Spinning Top",
                }
            ),
        ],
    )

    response_header = NamedList(
        [
            ("status", 0),
            ("QTime", 5),
        ]
    )

    response = SolrDocList(
        num_found=1,
        start=0,
        max_score=None,
        num_found_exact=True,
        docs=[doc],
    )

    top = NamedList(
        [
            ("responseHeader", response_header),
            ("response", response),
        ]
    )

    data = encode(top)
    result = javabin.deserialize(data)

    assert result["responseHeader"] == {"status": 0, "QTime": 5}

    resp = result["response"]
    assert resp["numFound"] == 1
    assert resp["start"] == 0
    assert resp["maxScore"] is None
    assert resp["numFoundExact"] is True
    assert len(resp["docs"]) == 1

    d = resp["docs"][0]
    assert d["movie_id"] == "movie-101"
    assert d["release_id"] == 123456789012
    assert d["title"] == "Inception Trivia - Spinning Top 😀 äöü"
    assert d["rating"] == pytest.approx(8.8)
    assert d["runtime_minutes"] == 148
    assert d["is_classic"] is False
    assert d["is_blockbuster"] is True
    assert d["last_updated"] is None
    assert d["genres"] == ["Action", "Sci-Fi", "Adventure"]
    assert d["description"] == "Spinning Top 😀 äöü"
    assert d["_childDocuments_"] == [
        {"movie_id": "movie-101-trivia-1", "title": "Inception Trivia - Spinning Top"}
    ]

    # deserialize_json must produce an equivalent structure.
    json_result = json.loads(javabin.deserialize_json(data))
    assert json_result == result


def test_date_and_double_roundtrip() -> None:
    from javabin_ref_encoder import Encoder

    enc = Encoder()
    enc.buf = bytearray([2])
    enc.write_date_millis(1_700_000_000_000)
    assert javabin.deserialize(bytes(enc.buf)) == 1_700_000_000_000

    enc2 = Encoder()
    enc2.buf = bytearray([2])
    enc2.write_double(3.14159265)
    assert javabin.deserialize(bytes(enc2.buf)) == pytest.approx(3.14159265)


def test_invalid_version_raises() -> None:
    with pytest.raises(ValueError, match="version"):
        javabin.deserialize(bytes([1, 0]))


def test_truncated_data_raises() -> None:
    with pytest.raises(ValueError):
        javabin.deserialize(bytes([2, 6, 0, 0]))  # INT tag but only 2 of 4 body bytes


def test_real_solr_movies_field_values_roundtrip() -> None:
    """
    Round-trip test using simulated movie field values modeled on open data:
    - ``_version_``: a real Lucene document version, a genuinely large long
      (> 2**56) that must use the full-width ``LONG`` tag rather than the
      compact ``SLONG`` encoding.
    - ``last_updated``: a movie release/update date.
    - ``genres``: a multi-valued string field.
    """
    from javabin_ref_encoder import JavaDate

    docs = [
        SolrDoc(
            fields={
                "movie_id": "movie-101",
                "release_id": "movie-101#1",
                "_version_": 1870516012295651331,
                "title": "Inception",
                "rating": 8.8,
                "runtime_minutes": 148,
                "release_year": 2010,
                "is_classic": False,
                "last_updated": JavaDate(1_517_270_400_000),  # 2018-01-30T00:00:00Z
                "genres": ["Action", "Sci-Fi", "Adventure"],
            }
        ),
        SolrDoc(
            fields={
                "movie_id": "movie-102",
                "release_id": "movie-102#1",
                "_version_": 1870516012295651330,
                "title": "The Matrix",
                "rating": 8.7,
                "runtime_minutes": 136,
                "release_year": 1999,
                "is_classic": True,
                "last_updated": JavaDate(1_321_488_000_000),  # 2011-11-17T00:00:00Z
                "genres": ["Action", "Sci-Fi"],
            }
        ),
    ]

    response = SolrDocList(
        num_found=2, start=0, max_score=None, num_found_exact=True, docs=docs
    )
    data = encode(response)
    result = javabin.deserialize(data)

    assert result["numFound"] == 2
    assert len(result["docs"]) == 2

    first = result["docs"][0]
    assert first["movie_id"] == "movie-101"
    assert first["_version_"] == 1870516012295651331
    assert first["rating"] == pytest.approx(8.8)
    assert first["last_updated"] == 1_517_270_400_000
    assert first["genres"] == ["Action", "Sci-Fi", "Adventure"]

    second = result["docs"][1]
    assert second["_version_"] == 1870516012295651330
    assert second["genres"] == ["Action", "Sci-Fi"]
    assert second["release_year"] == 1999
    assert second["last_updated"] == 1_321_488_000_000
