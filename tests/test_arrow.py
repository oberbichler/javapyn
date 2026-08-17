"""
Tests for the Arrow output path (``deserialize_arrow`` and
``ArrowStreamDecoder``), verified against the same live ``wt=json`` reference
fixtures used by the other decoders.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

import javapyn as javabin

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> tuple[bytes, dict]:
    data = (FIXTURES / f"{name}.bin").read_bytes()
    ref = json.loads((FIXTURES / f"{name}.json").read_text())
    return data, ref


def _docs(ref: dict) -> list:
    if "response" in ref:
        return ref["response"]["docs"]
    return ref["result-set"]["docs"]


def _date_ms(iso: str) -> int:
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in iso else "%Y-%m-%dT%H:%M:%SZ"
    dt = datetime.strptime(iso, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# A schema covering the field types present in solr_movies_deterministic.
MOVIE_SCHEMA = pa.schema(
    [
        ("movie_id", pa.string()),
        ("title", pa.string()),
        ("rating", pa.float32()),
        ("runtime_minutes", pa.int32()),
        ("release_year", pa.int32()),
        ("_version_", pa.int64()),
        ("last_updated", pa.timestamp("ms")),
        ("genres", pa.list_(pa.string())),
    ]
)


def _assert_batch_matches_json(batch, ref_docs, schema) -> None:
    assert batch.num_rows == len(ref_docs)
    cols = batch.to_pydict()
    for i, doc in enumerate(ref_docs):
        for field in schema:
            name, typ = field.name, field.type
            got = cols[name][i]
            expected = doc.get(name)
            if expected is None:
                assert got is None, f"row {i} col {name}: expected null, got {got!r}"
            elif pa.types.is_timestamp(typ):
                assert got == datetime.fromtimestamp(
                    _date_ms(expected) / 1000, tz=timezone.utc
                ).replace(tzinfo=None), f"row {i} col {name}"
            elif pa.types.is_float32(typ):
                assert abs(got - expected) < 1e-6, f"row {i} col {name}"
            else:
                assert got == expected, f"row {i} col {name}: {got!r} != {expected!r}"


def test_deserialize_arrow_matches_json():
    data, ref = _load("solr_movies_deterministic")
    batch = javabin.deserialize_arrow(data, MOVIE_SCHEMA)
    assert isinstance(batch, pa.RecordBatch)
    _assert_batch_matches_json(batch, _docs(ref), MOVIE_SCHEMA)


def test_deserialize_arrow_to_polars():
    pl = pytest.importorskip("polars")
    data, ref = _load("solr_movies_deterministic")
    batch = javabin.deserialize_arrow(data, MOVIE_SCHEMA)
    df = pl.from_arrow(batch)
    assert df.height == len(_docs(ref))
    assert df["title"].to_list() == [d["title"] for d in _docs(ref)]
    # multi-valued list column round-trips
    assert df["genres"].to_list()[0] == _docs(ref)[0]["genres"]


def test_deserialize_arrow_stream_endpoint():
    # /stream result-set with a trailing EOF marker: the EOF doc has fields
    # not in our schema (EOF/RESPONSE_TIME) and is simply an extra row of
    # mostly-nulls; the real docs must decode correctly.
    data, ref = _load("solr_movies_stream")
    schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("title", pa.string()),
            ("rating", pa.float32()),
            ("runtime_minutes", pa.int32()),
        ]
    )
    batch = javabin.deserialize_arrow(data, schema)
    ref_docs = _docs(ref)
    assert batch.num_rows == len(ref_docs)
    cols = batch.to_pydict()
    # last row is the EOF marker -> all schema columns null there
    assert cols["movie_id"][-1] is None
    # the real data rows match
    for i, doc in enumerate(ref_docs[:-1]):
        assert cols["movie_id"][i] == doc["movie_id"]
        assert cols["title"][i] == doc["title"]


@pytest.mark.parametrize("name", ["solr_movies_all_fields", "solr_studios_all_fields"])
def test_deserialize_arrow_subset_fields(name):
    # Requesting a subset of fields still works; each column matches JSON.
    data, ref = _load(name)
    docs = _docs(ref)
    # pick a couple of common typed fields
    common = {}
    for doc in docs:
        for k, v in doc.items():
            common.setdefault(k, type(v))
    # build a small schema of string+int fields that appear
    fields = []
    for k, t in common.items():
        if t is str:
            fields.append((k, pa.string()))
        elif t is int:
            fields.append((k, pa.int64()))
        elif t is bool:
            fields.append((k, pa.bool_()))
        if len(fields) >= 4:
            break
    schema = pa.schema(fields)
    batch = javabin.deserialize_arrow(data, schema)
    cols = batch.to_pydict()
    for i, doc in enumerate(docs):
        for name_, _ in fields:
            assert cols[name_][i] == doc.get(name_), f"{name_} row {i}"


def test_arrow_stream_decoder_matches_single_shot():
    data, _ = _load("solr_movies_deterministic")
    reference = javabin.deserialize_arrow(data, MOVIE_SCHEMA)

    # byte-by-byte
    dec = javabin.ArrowStreamDecoder(MOVIE_SCHEMA, batch_size=2)
    batches = []
    for i in range(len(data)):
        batches.extend(dec.feed(data[i : i + 1]))
    batches.extend(dec.finish())
    table = pa.Table.from_batches(batches, schema=MOVIE_SCHEMA)
    assert table.num_rows == reference.num_rows
    assert table.to_pydict() == pa.Table.from_batches([reference]).to_pydict()


def test_arrow_stream_decoder_random_chunking():
    import random

    data, _ = _load("solr_movies_all_fields")
    schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("_version_", pa.int64()),
            ("release_year", pa.int32()),
        ]
    )
    reference = javabin.deserialize_arrow(data, schema).to_pydict()

    for seed in range(10):
        rng = random.Random(seed)
        dec = javabin.ArrowStreamDecoder(schema, batch_size=3)
        batches = []
        i = 0
        while i < len(data):
            n = rng.randint(1, 40)
            batches.extend(dec.feed(data[i : i + n]))
            i += n
        batches.extend(dec.finish())
        table = pa.Table.from_batches(batches, schema=schema)
        assert table.to_pydict() == reference, f"seed {seed}"


def test_arrow_stream_decoder_batching():
    data, ref = _load("solr_movies_all_fields")
    n = len(_docs(ref))
    schema = pa.schema([("movie_id", pa.string())])
    dec = javabin.ArrowStreamDecoder(schema, batch_size=2)
    batches = dec.feed(data)
    batches.extend(dec.finish())
    sizes = [b.num_rows for b in batches]
    assert sum(sizes) == n
    assert all(s <= 2 for s in sizes)


def test_arrow_child_document_errors():
    # A doc with child documents cannot be flattened -> ValueError.
    # Build synthetically using the reference encoder.
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from javabin_ref_encoder import NamedList, SolrDoc, SolrDocList, encode

    doc = SolrDoc(
        fields={"movie_id": "p"}, children=[SolrDoc(fields={"movie_id": "c"})]
    )
    msg = encode(NamedList([("response", SolrDocList(1, 0, None, True, [doc]))]))
    schema = pa.schema([("movie_id", pa.string())])
    with pytest.raises(ValueError, match="child document"):
        javabin.deserialize_arrow(msg, schema)


def test_arrow_skips_named_child_field():
    """A named child field has no flat column, so leaving it out of the schema
    skips it -- and the field *after* it must still land in the right column.

    Regression test: skipping a child field used to consume only half of the
    child's field list, so later columns were read from inside the child document
    and silently got wrong values.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from javabin_ref_encoder import NamedList, SolrDoc, SolrDocList, encode

    doc = SolrDoc(
        fields={
            "movie_id": "p",
            # Named nesting: what [child] returns from a schema with _nest_path_.
            "chapters": [
                SolrDoc(fields={"movie_id": "c1", "rating": 1.0}),
                SolrDoc(fields={"movie_id": "c2", "rating": 2.0}),
            ],
            "rating": 8.5,
        }
    )
    msg = encode(NamedList([("response", SolrDocList(1, 0, None, True, [doc]))]))
    schema = pa.schema([("movie_id", pa.string()), ("rating", pa.float32())])

    batch = javabin.deserialize_arrow(msg, schema)

    assert batch.to_pydict() == {"movie_id": ["p"], "rating": [8.5]}


def _nested_message():
    """A book with two chapters and one section, three levels deep.

    Each document's ``id`` is written *after* its child list, which is the
    ordering that only works if the parent key is resolved after staging.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from javabin_ref_encoder import NamedList, SolrDoc, SolrDocList, encode

    section = SolrDoc(fields={"id": "s1", "title": "Section"})
    chapters = [
        SolrDoc(fields={"title": "Chapter 1", "sections": [section], "id": "c1"}),
        SolrDoc(fields={"title": "Chapter 2", "id": "c2"}),
    ]
    book = SolrDoc(fields={"title": "Book", "chapters": chapters, "id": "b1"})
    return encode(NamedList([("response", SolrDocList(1, 0, None, True, [book]))]))


NESTED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("title", pa.string()),
        ("_parent_id", pa.string()),
        ("_depth", pa.int32()),
        ("_child_field", pa.string()),
    ]
)


def test_arrow_explode_emits_one_row_per_document():
    batch = javabin.deserialize_arrow(
        _nested_message(), NESTED_SCHEMA, children="explode"
    )

    assert batch.to_pydict() == {
        # Pre-order: a document, then its children.
        "id": ["b1", "c1", "s1", "c2"],
        "title": ["Book", "Chapter 1", "Section", "Chapter 2"],
        "_parent_id": [None, "b1", "c1", "b1"],
        "_depth": [0, 1, 2, 1],
        "_child_field": [None, "chapters", "sections", "chapters"],
    }


def test_arrow_skip_is_the_default():
    batch = javabin.deserialize_arrow(_nested_message(), NESTED_SCHEMA)

    assert batch.to_pydict() == {
        "id": ["b1"],
        "title": ["Book"],
        "_parent_id": [None],
        "_depth": [0],
        "_child_field": [None],
    }


def test_arrow_explode_metadata_columns_are_renameable():
    schema = pa.schema(
        [("id", pa.string()), ("parent", pa.string()), ("level", pa.int64())]
    )

    batch = javabin.deserialize_arrow(
        _nested_message(),
        schema,
        children="explode",
        child_columns={"parent_id": "parent", "depth": "level"},
    )

    assert batch.to_pydict() == {
        "id": ["b1", "c1", "s1", "c2"],
        "parent": [None, "b1", "c1", "b1"],
        "level": [0, 1, 2, 1],
    }


def test_arrow_stream_decoder_explodes_across_chunk_boundaries():
    data = _nested_message()

    for chunk in (1, 5, len(data)):
        decoder = javabin.ArrowStreamDecoder(
            NESTED_SCHEMA, batch_size=2, children="explode"
        )
        batches = [
            b
            for i in range(0, len(data), chunk)
            for b in decoder.feed(data[i : i + chunk])
        ]
        batches.extend(decoder.finish())
        table = pa.Table.from_batches(batches, schema=NESTED_SCHEMA)

        assert table.column("id").to_pylist() == ["b1", "c1", "s1", "c2"], chunk
        assert table.column("_parent_id").to_pylist() == [None, "b1", "c1", "b1"], chunk


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"children": "nope"}, 'must be "skip" or "explode"'),
        (
            {"children": "explode", "child_columns": {"nope": "x"}},
            "unknown child_columns role",
        ),
    ],
)
def test_arrow_child_arguments_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        javabin.deserialize_arrow(_nested_message(), NESTED_SCHEMA, **kwargs)


def test_arrow_type_mismatch_errors():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from javabin_ref_encoder import NamedList, SolrDoc, SolrDocList, encode

    doc = SolrDoc(fields={"rating_float": "a string"})
    msg = encode(NamedList([("response", SolrDocList(1, 0, None, True, [doc]))]))
    schema = pa.schema([("rating_float", pa.int32())])  # but value is a string
    with pytest.raises(ValueError):
        javabin.deserialize_arrow(msg, schema)
