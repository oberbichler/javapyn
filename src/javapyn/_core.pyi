from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pyarrow

def deserialize(data: bytes) -> Any:
    """
    Deserialize a Solr ``javabin`` (protocol version 2) byte string into
    native Python objects.

    Parameters
    ----------
    data
        The raw javabin-encoded response body (e.g. the body of an HTTP
        response requested with ``wt=javabin``).

    Returns
    -------
    Any
        The decoded value. Solr query responses typically decode to a
        ``dict`` (the top-level ``NamedList``), with a ``"response"`` key
        holding a ``dict`` shaped like ``{"numFound", "start", "maxScore",
        "numFoundExact", "docs"}``.

    Raises
    ------
    ValueError
        If ``data`` is not valid javabin (wrong version byte, truncated,
        invalid UTF-8, or an unknown tag byte), including the byte offset at
        which decoding failed where available.
    """

def deserialize_json(data: bytes) -> str:
    """
    Deserialize a Solr ``javabin`` byte string directly into a JSON string,
    without constructing intermediate Python objects.

    Parameters
    ----------
    data
        The raw javabin-encoded response body.

    Returns
    -------
    str
        The decoded value serialized as JSON, using the same shape as
        :func:`deserialize`.

    Raises
    ------
    ValueError
        If ``data`` is not valid javabin.
    """

def deserialize_stream(data: bytes, callback: Callable[[Any], object]) -> Any:
    """
    Deserialize a Solr ``javabin`` response, streaming each document to
    ``callback`` instead of collecting them all.

    For every document in the result's ``docs`` sequence (from ``/select``,
    ``/export`` or ``/stream``), ``callback(doc)`` is invoked and the document
    is then released, keeping peak memory at roughly one document rather than
    the whole result set.

    Note the full response body must still be in memory as ``data``; this
    streams the *decoded objects*, not the raw byte download.

    Parameters
    ----------
    data
        The raw javabin-encoded response body.
    callback
        Called once per document; its return value is ignored.

    Returns
    -------
    Any
        The response envelope (same shape as :func:`deserialize` but with an
        empty ``docs`` list), so metadata such as ``numFound`` remains
        available.

    Raises
    ------
    ValueError
        If ``data`` is not valid javabin. Any exception raised by ``callback``
        propagates out unchanged.
    """

class StreamDecoder:
    """
    Incremental, chunk-fed streaming decoder for large ``/export`` responses.

    Feed network chunks with :meth:`feed`; every complete document is passed to
    the callback and released, so the internal byte buffer stays at roughly one
    document regardless of the total response size (unlike
    :func:`deserialize_stream`, which needs the whole body in memory).

    Example
    -------
    ::

        dec = javapyn.StreamDecoder()
        with client.stream("POST", url, data=params) as resp:
            for chunk in resp.iter_bytes():
                dec.feed(chunk, handle_doc)
        dec.finish()
        print(dec.count, "documents")
    """

    def __init__(self) -> None: ...
    def feed(self, chunk: bytes, callback: Callable[[Any], object]) -> None:
        """Feed one chunk of bytes; complete documents are passed to
        ``callback``. Exceptions raised by ``callback`` propagate unchanged."""

    def finish(self) -> None:
        """Signal end of input. Raises ``ValueError`` if the stream ended
        mid-document or before the document sequence completed."""

    @property
    def count(self) -> int:
        """Total number of documents streamed so far."""

def deserialize_arrow(data: bytes, schema: "pyarrow.Schema") -> "pyarrow.RecordBatch":
    """
    Deserialize a Solr ``javabin`` response directly into an Arrow
    ``RecordBatch``, given the target Arrow schema.

    Every document becomes a row; every schema field becomes a typed column.
    Absent fields become nulls. No per-value Python object is built — the whole
    batch is handed back as one zero-copy Arrow structure — making this the
    fastest way to load a tabular result into a DataFrame
    (``polars.from_arrow`` / ``pandas``).

    Supported column types: int32, int64, float32, float64, bool, string
    (utf8), binary, timestamp('ms'), and ``list_`` of any of those for
    multi-valued fields. ``/stream`` encodes integers/floats more widely, so
    an int32 column also accepts a long that fits, and a float32 column also
    accepts a double.

    Raises
    ------
    ValueError
        If ``data`` is not valid javabin, a value doesn't fit its column type,
        the schema uses an unsupported type, or a document has child documents.
    """

class ArrowStreamDecoder:
    """
    Incremental, chunk-fed decoder that yields Arrow ``RecordBatch`` objects of
    ``batch_size`` rows as they complete. Combines flat-memory chunk streaming
    with columnar, object-free decoding — the most efficient way to load a huge
    ``/export`` into a DataFrame without holding the whole thing in memory.

    Example
    -------
    ::

        import pyarrow as pa, polars as pl
        schema = pa.schema([("title", pa.string()), ("rating", pa.float32())])
        dec = javapyn.ArrowStreamDecoder(schema, batch_size=65536)
        batches = []
        with client.stream("POST", url, data=params) as resp:
            for chunk in resp.iter_bytes():
                batches.extend(dec.feed(chunk))
        batches.extend(dec.finish())
        df = pl.from_arrow(pa.Table.from_batches(batches, schema=schema))
    """

    def __init__(self, schema: "pyarrow.Schema", batch_size: int = 65536) -> None: ...
    def feed(self, chunk: bytes) -> list["pyarrow.RecordBatch"]:
        """Feed one chunk; returns any ``RecordBatch`` objects completed as a
        result (possibly empty)."""

    def finish(self) -> list["pyarrow.RecordBatch"]:
        """Signal end of input; returns the final batch list. Raises
        ``ValueError`` on truncation."""

    def schema(self) -> "pyarrow.Schema":
        """The Arrow schema this decoder produces."""
