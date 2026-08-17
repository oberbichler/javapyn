# javapyn

[![PyPI](https://img.shields.io/pypi/v/javapyn.svg)](https://pypi.org/project/javapyn/)

Fast Python deserializer for Apache Solr's javabin (protocol version 2) wire format ([org.apache.solr.common.util.JavaBinCodec](https://github.com/apache/solr/blob/main/solr/solrj/src/java/org/apache/solr/common/util/JavaBinCodec.java)), written in Rust.

## Why

Solr's default `wt=json` response format is convenient but slower and larger on the wire than `wt=javabin`, Solr's compact binary protocol. This package implements a from-scratch decoder for that format in Rust (no Java/JVM dependency) and exposes it to Python as `javapyn`.

## Usage

```python
import httpx
import javapyn

response = httpx.get(
    "https://solr.example.com/solr/solr_movies/select",
    params={"q": "*:*", "rows": 10, "wt": "javabin", "version": "2"},
)

data = javapyn.deserialize(response.content)
# data["response"]["docs"] -> list[dict]

json_text = javapyn.deserialize_json(response.content)
```

`deserialize` returns native Python objects: dict, list, str, int, float, bool, bytes, None. SolrDocumentList values (the response key of a query result) are shaped like Solr's own `wt=json` response: `{"numFound", "start", "maxScore", "numFoundExact", "docs"}`. Child documents (Solr's nested/child-document feature) appear under a `"_childDocuments_"` key on the parent document.

`deserialize_json` does the same decoding but serializes directly to a JSON string via `serde_json`, skipping the Python object construction step. Where JSON cannot express what javabin can, the output matches Solr's own `wt=json`: a binary field becomes a base64 string, a non-finite float becomes `"Infinity"`/`"-Infinity"`/`"NaN"`, and a null NamedList name becomes the empty-string key.

### Supported endpoints

All three of Solr's response-producing handlers are supported, despite using different javabin encodings under the hood (all verified against live wt=json references):

- select: the standard query handler. Top-level NamedList with a SolrDocumentList under response.
- export: the export handler (docValues fields only). Encodes its response with the streaming MAP_ENTRY_ITER tag and docs as an ITERATOR (unknown length, END-terminated) rather than fixed-size containers. Decodes to the same `{"responseHeader": ..., "response": {"numFound", "docs"}}` shape as its JSON form.
- stream (streaming expressions, e.g. search, rollup): top-level `{"result-set": {"docs": [...]}}`, where docs is an ITERATOR ending with a synthetic `{"EOF": true, "RESPONSE_TIME": ...}` marker.

```python
# /stream example
resp = httpx.post(
    "https://solr.example.com/solr/solr_movies/stream",
    data={
        "expr": 'search(solr_movies, q="*:*", fl="movie_id,rating", sort="movie_id asc", qt="/export")',
        "wt": "javabin",
        "version": "2",
    },
)
docs = javapyn.deserialize(resp.content)["result-set"]["docs"]
# docs[-1] == {"EOF": True, "RESPONSE_TIME": <ms>}
```

## Memory and streaming

`deserialize` and `deserialize_json` require the entire response body to already be in memory as data. For large export or stream result sets, use `deserialize_stream`, which hands each document to a callback and drops it immediately, keeping peak memory at roughly one document:

```python
import javapyn

# process one document at a time; nothing accumulates
def handle(doc):
    pass

envelope = javapyn.deserialize_stream(response.content, handle)
# envelope still has metadata: envelope["response"]["numFound"], etc.
# (its "docs" list is empty — the docs went to the callback)
```

Note that `deserialize_stream` streams the decoded objects but still holds the whole raw byte buffer. For a genuinely large export (many GB) even the byte buffer is too big. Use `StreamDecoder` instead, which is fed network chunks and holds only one document's worth of bytes at a time:

```python
import javapyn

dec = javapyn.StreamDecoder()
with client.stream("POST", f"{collection}/export",
                   data={"q": "*:*", "fl": "...", "sort": "movie_id asc",
                         "wt": "javabin", "version": "2"}) as resp:
    for chunk in resp.iter_bytes():
        dec.feed(chunk, handle_doc)
dec.finish()
print(dec.count, "documents")
```

`StreamDecoder` decodes the small response envelope, then emits each complete document to the callback as soon as its bytes have arrived, dropping consumed bytes. It resumes only at document boundaries (documents are self-contained), so it works for arbitrary network chunk sizes.

## Performance and Benchmarks

We measured performance on a modern system (CPython 3.14, Apple Silicon M1 Pro). The benchmarks are fully reproducible using the scripts folder.

### Benchmark 1: Streaming Deserialization

This benchmark stream-processes 20_000_000 movie records (ca. 1.43 GB javabin, equivalent to ca. 4.5 GB raw Solr JSON) on the fly. It pipes raw bytes directly into javapyn decoders:

| Deserialization Method      | Total Time | Speed (docs/sec) | Memory Overhead | Performance characteristics                |
| :-------------------------- | :--------: | :--------------: | :-------------: | :----------------------------------------- |
| StreamDecoder (Objects)     |  50.03 s   |     399,780      | flat (~523 MB)  | Streaming Python dictionary structures     |
| ArrowStreamDecoder (Polars) |  49.33 s   |     405,409      | flat (~1005 MB) | Streaming zero-copy columnar Arrow batches |

Memory stays flat throughout the entire run since consumed bytes and processed records are immediately discarded.

### Benchmark 2: In-Memory Deserialization (1,000,000 Records)

This benchmark parses 1_000_000 movie records (JSON = 113.38 MB, javabin = 71.42 MB, representing a 2.3x smaller wire payload):

| Deserialization Method             | Total Time | Memory Overhead | Speed Comparison              |
| :--------------------------------- | :--------: | :-------------: | :---------------------------- |
| json.loads (std-lib JSON)          |  0.6539 s  |    522.0 MB     | baseline speed                |
| orjson.loads (Rust JSON)           |  0.3540 s  |    647.2 MB     | 1.8x faster                   |
| javapyn.deserialize (Rust)         |  0.3302 s  |    395.0 MB     | 2.0x faster with 25% less RAM |
| javapyn.deserialize_arrow (Polars) |  0.1694 s  |    100.2 MB     | 4.0x faster with 5x less RAM  |

## Arrow / DataFrame output

When the destination is a DataFrame, decoding into per-row Python objects and then re-parsing them is wasteful. `deserialize_arrow` decodes straight into columnar Arrow arrays (zero Python objects per value) and hands back a single `pyarrow.RecordBatch` over the Arrow C Data Interface (zero-copy). Requires the arrow extra (`pip install javapyn[arrow]` or `uv add "javapyn[arrow]"`, which pulls in pyarrow).

```python
import pyarrow as pa
import polars as pl
import javapyn

# One field per column; type per the Solr schema.
schema = pa.schema([
    ("movie_id", pa.string()),
    ("title", pa.string()),
    ("rating", pa.float32()),
    ("last_updated", pa.timestamp("ms")),
    ("genres", pa.list_(pa.string())),
])

batch = javapyn.deserialize_arrow(response.content, schema)
df = pl.from_arrow(batch)
```

Supported column types: `int32`, `int64`, `float32`, `float64`, `bool`, `string`, `binary`, `timestamp('ms')`, and list of any of those for multi-valued fields. Fields absent from a document become nulls; fields not in the schema are skipped.

For a multi-GB export, use `ArrowStreamDecoder` to get batches as chunks arrive, so neither the bytes nor the decoded columns are ever fully in memory:

```python
schema = pa.schema([("title", pa.string()), ("rating", pa.float32())])
dec = javapyn.ArrowStreamDecoder(schema, batch_size=65536)
batches = []
with client.stream("POST", f"{collection}/export",
                   data={"q": "*:*", "fl": "title,rating",
                         "sort": "movie_id asc", "wt": "javabin", "version": "2"}) as r:
    for chunk in r.iter_bytes():
        batches.extend(dec.feed(chunk))
batches.extend(dec.finish())
df = pl.from_arrow(pa.Table.from_batches(batches, schema=schema))
```

## Loading large collections fast

When bulk-reading a big collection, how you page matters far more than the decoder. Performance on a large dataset:

| method                              |    throughput    | note                         |
| :---------------------------------- | :--------------: | :--------------------------- |
| select + cursorMark (serial)        |  ~3-13k docs/s   | one request per 10k-doc page |
| select + cursorMark, 10-way sharded |   ~7.8k docs/s   | one cursor per shard, async  |
| export (single request)             | ~130-260k docs/s | whole result set streamed    |
| export, 10-way sharded              |   ~800k docs/s   | one export per shard         |

export is 10-20x faster than cursorMark paging. It streams the entire matching set in a single request and its javabin encoding decodes cleanly.

Caveats for export:

- docValues only: every field in fl and the sort field must have docValues.
- A sort is required.
- Use POST for wide field lists.

```python
with client.stream("POST", f"{collection}/export",
                   data={"q": "*:*", "fl": "movie_id,rating,release_year",
                         "sort": "movie_id asc", "wt": "javabin", "version": "2"}) as r:
    javapyn.deserialize_stream(r.read(), handle_doc)
```

## Development

```sh
uv venv
uv sync --group dev
uvx maturin develop --release --uv
uv run pytest tests/
cargo test --release --lib
```

### Notes on `cargo test`

`cargo test` embeds a real Python interpreter (via PyO3's `auto-initialize` feature, enabled under `dev-dependencies` only — the actual extension module doesn't link `libpython` at all; see `Cargo.toml`). PyO3 picks whichever Python it finds first: an active virtualenv, then `python`, then `python3` on `PATH`.

On macOS, if the Xcode Command Line Tools are installed, `python3` on `PATH` may resolve ahead of your venv to a stub `Python3.framework` that has no actual `libpython3.x.dylib`, which fails with a linker error like:

```
ld: library 'python3.9' not found
```

If you hit this, point PyO3 explicitly at the project's own uv-managed venv:

```sh
PYO3_PYTHON=$(pwd)/.venv/bin/python3 cargo test --release --lib
```

This is a local `PATH`-ordering quirk only, not something `cargo test` needs in general: CI (`.github/workflows/ci.yml`) doesn't set `PYO3_PYTHON` and doesn't hit this, because its `test-rust` job runs `actions/setup-python` first, which puts a working interpreter at the front of `PATH` before a system Python stub can shadow it.

Use `--release` for cargo test and maturin develop.

## Testing strategy

The decoder is validated four ways:

1. Rust unit tests hand-construct byte sequences for each tag type per the JavaBinCodec spec and assert the decoded result. Fast-path tests assert it decodes identically to the safe path. Reference-leak and error-path cleanup of the unsafe ffi code are checked from Python.
2. Python integration tests use an independent reference encoder (tests/javabin_ref_encoder.py) to build realistic response shapes (modeled on the movie Solr collection schema) and verify round-tripping.
3. Live-modeled fixture tests (tests/test_live_fixtures.py) replay real-world modeled wt=javabin byte responses captured from several Solr movie and studio collections and assert field-by-field equality against a wt=json response.
4. Live conformance tests (tests/test_solr_container.py) start a real Solr in a container and compare against bytes from Solr's own JavaBinCodec.

All Solr movie and studio collections in int2 have been verified this way with fl=\*. Supported types include string, int, long, float, bool, tdate (plus multi-valued arrays).

### Live Solr conformance tests

The first three layers all rest on our own reading of the javabin spec — the reference encoder shares any blind spot with the decoder, and the fixtures are frozen captures. The container tests make Apache Solr itself the reference: each one issues the same query twice, as `wt=javabin` and as `wt=json`, and asserts the decoded bytes equal what Solr says the result is (`tests/javabin_compare.py` knows every legitimate difference between the two renderings — dates as millis, float32 rounding, binary as base64, non-finite doubles as strings, NamedLists rendered flat — and treats everything else as a defect).

Two collections are provisioned (`tests/solr_probe.py`): `solr_movies` with 2 000 documents for scale and streaming, and `solr_types` with one document per awkward case. Coverage:

- **Every field type**: string, text, binary, boolean, pint, plong, pfloat, pdouble, pdate, and multi-valued variants of each — so INT/LONG/FLOAT/DOUBLE/DATE/BYTEARR come from Solr's encoder, not ours.
- **Awkward values**: `Integer`/`Long` limits, ±`Float`/`Double` max, denormals, negative zero, the SINT/SLONG inline-vs-vint boundaries, the STR inline-size boundary, a 200 000-character string, non-BMP and ZWJ Unicode, dates before 1970 and with fractional seconds, absent fields, nested child documents two levels deep.
- **Component response shapes**: field/range/query/pivot/interval facets, JSON facets, stats with percentiles, all four grouping modes, highlighting, debug, terms, cursor paging, collapse/expand, function-query fields, and error responses (HTTP 400 is a javabin response too).
- **All three handlers**: `/select`, `/export` (including non-empty exports), and eight `/stream` expression types.
- **Framing cross-check**: `tests/javabin_scanner.py` is an independent pure-Python scanner that walks the tag stream without touching the Rust decoder. Every real response must be consumed exactly, so two independent implementations agree on where every value ends. It also asserts which tags the live suite actually exercises, so shrinking coverage becomes a test failure.
- **Robustness**: every prefix of every real response must raise `ValueError` (a truncated response is what a dropped connection looks like), and thousands of random byte mutations must never crash the interpreter or raise an undocumented exception type.

They need Docker and are deselected by default, so the normal `uv run pytest` stays fast and Docker-free:

```sh
uv run pytest -m solr
```

The Solr image is pinned (`solr:9.10.1`); override it to check another release. CI runs one release per supported major, since a javabin change in a new Solr has to surface here:

```sh
JAVAPYN_SOLR_IMAGE=solr:8.11.4 uv run pytest -m solr
JAVAPYN_SOLR_IMAGE=solr:10.0.0 uv run pytest -m solr
```

Without a reachable Docker daemon the tests skip rather than fail. With [Colima](https://github.com/abiosoft/colima) instead of Docker Desktop, Testcontainers cannot bind-mount Colima's socket path into its reaper container; point it at the in-VM socket:

```sh
TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock uv run pytest -m solr
```

A Solr container started the same way also serves as a fixture source — `scripts/fetch_sample.py --base-url http://localhost:<port>/solr --collection solr_movies` captures new fixtures from it — so adding fixtures no longer requires access to an internal Solr.

### Defects this suite found

Five, all fixed and covered by regression tests. They are listed here because each one changes what you get back:

| Defect | Effect |
| :----- | :----- |
| A `NamedList` entry with a null name was rejected | Any `facet.missing=true` response was undecodable (`ValueError: expected string`). Solr writes a NULL name for the missing bucket and its own reader accepts it (`(String) readVal(...)`). Reproduced on Solr 8, 9 and 10, for string and numeric fields, with both facet methods. Such an entry now lands under the `None` key. |
| A failed `/export` was invisible to the streaming decoders | `deserialize` returned the `EXCEPTION` document, but `deserialize_stream`, `StreamDecoder` and `deserialize_arrow` reported zero documents and raised nothing, so a failed export was indistinguishable from an empty one. The error response encodes `response` as a MAP holding a plain ARR rather than the MAP_ENTRY_ITER + ITERATOR of a successful export; all envelope walkers now know that shape. |
| `StreamDecoder.finish()` accepted a truncated stream | It returned successfully whenever the envelope had not been parsed, so feeding a lone version byte reported a clean, empty result. It now reports truncation unless the document sequence actually terminated. |
| `deserialize_json` lost non-finite doubles | ±Infinity and NaN became `null`. Both now render as `"Infinity"`/`"-Infinity"`/`"NaN"`, as Solr's own writer does. Reachable from `stats.field` over large doubles and from overflowing function queries. |
| A container as a map key raised `TypeError` | Malformed javabin is documented to raise `ValueError`; a corrupted `MAP_ENTRY_ITER` key let CPython's `TypeError: unhashable type` escape instead. Found by mutation-fuzzing real responses. |

One deliberate output change came with them: `deserialize_json` now renders a binary field as base64 rather than an array of integers, matching `wt=json`.

Tags that live Solr never emits in a query response — `BYTE`, `SHORT`, `MAP_ENTRY`, `ENUM_FIELD_VALUE`, `PRIMITIVE_ARR`, `SOLRINPUTDOC`, `UUID` — stay covered by the Rust unit tests and the reference encoder alone. Notably a Solr `UUIDField` serialises as a plain `STR`, so javabin's `UUID` tag (which this decoder rejects) is not reachable through a schema.

To capture new live-modeled fixtures, use a small script (scripts/fetch_sample.py):

```sh
uv run python scripts/fetch_sample.py \
    --base-url https://solr.example.com/solr \
    --collection solr_movies \
    --query "last_updated:[* TO *] AND is_blockbuster:true" \
    --fields "movie_id,title,rating,genres,_version_,last_updated,is_classic,release_id,runtime_minutes" \
    --rows 5 \
    --out /path/to/javapyn/tests/fixtures/solr_movies_deterministic
```

## Format reference

Implemented directly from the [Apache Solr JavaBinCodec source](https://github.com/apache/solr/blob/main/solr/solrj/src/java/org/apache/solr/common/util/JavaBinCodec.java), supporting: `NULL`, `BOOL_TRUE`/`FALSE`, `BYTE`, `SHORT`, `DOUBLE`, `INT`, `LONG`, `FLOAT`, `DATE`, `MAP`, `SOLRDOC`, `SOLRDOCLST`, `BYTEARR`, `ITERATOR`, `END`, `SOLRINPUTDOC`, `MAP_ENTRY_ITER`, `ENUM_FIELD_VALUE`, `MAP_ENTRY`, `PRIMITIVE_ARR`, `STR`, `SINT`, `SLONG`, `ARR`, `ORDERED_MAP`, `NAMED_LST`, `EXTERN_STRING`.
