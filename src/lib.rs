//! Python bindings (via PyO3) for the javabin decoder.

mod decoder;
mod py_decoder;
mod py_decoder_fast;
mod reader;
mod stream_decoder;
mod value;

mod arrow_decoder;

use arrow::pyarrow::{PyArrowType, ToPyArrow};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Python-visible error raised for javabin decode failures.
fn decode_error_to_py(err: reader::DecodeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
mod _core {
    use super::*;

    /// Deserialize a Solr `javabin` (protocol version 2) byte string into
    /// native Python objects.
    ///
    /// Decodes directly into Python objects in a single pass (see
    /// `py_decoder` module docs); field names repeated across sibling
    /// documents (the common case in a `SolrDocumentList`) are interned so
    /// repeat occurrences are a cheap reference instead of a fresh string.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The raw javabin-encoded response body (e.g. the body of an HTTP
    ///     response requested with ``wt=javabin``).
    ///
    /// Returns
    /// -------
    /// Any
    ///     The decoded value. Solr query responses typically decode to a
    ///     ``dict`` (the top-level ``NamedList``), with a ``"response"`` key
    ///     holding a ``dict`` shaped like ``{"numFound", "start", "maxScore",
    ///     "numFoundExact", "docs"}``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``data`` is not valid javabin (wrong version byte, truncated,
    ///     invalid UTF-8, or an unknown tag byte), including the byte offset
    ///     at which decoding failed where available.
    #[pyfunction]
    fn deserialize<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        py_decoder_fast::decode(py, data)
    }

    /// Deserialize a Solr `javabin` response, streaming each document to a
    /// callback instead of collecting them all.
    ///
    /// For every document in the result's ``docs`` sequence (whether from
    /// ``/select``, ``/export`` or ``/stream``), ``callback(doc)`` is invoked
    /// and the document is then released. This keeps peak memory at roughly
    /// one document rather than the entire result set, which matters for
    /// large exports.
    ///
    /// Note the whole response body must still be in memory as ``data`` — this
    /// streams the *decoded objects*, not the raw byte download.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The raw javabin-encoded response body.
    /// callback : Callable[[Any], object]
    ///     Called once per document; its return value is ignored.
    ///
    /// Returns
    /// -------
    /// Any
    ///     The response envelope (same shape as :func:`deserialize`, but with
    ///     an empty ``docs`` list), so metadata such as ``numFound`` remains
    ///     available.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``data`` is not valid javabin. Any exception raised by
    ///     ``callback`` propagates out unchanged.
    #[pyfunction]
    fn deserialize_stream<'py>(
        py: Python<'py>,
        data: &[u8],
        callback: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py_decoder_fast::decode_stream(py, data, callback)
    }

    /// Safe-PyO3 reference implementation of :func:`deserialize`, kept for
    /// differential testing/benchmarking against the ``pyo3::ffi`` fast path.
    #[pyfunction]
    fn _deserialize_safe<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        py_decoder::decode(py, data)
    }

    /// Deserialize a Solr `javabin` byte string directly into a JSON string,
    /// without constructing intermediate Python objects.
    ///
    /// This is faster than ``deserialize`` followed by ``json.dumps`` for
    /// use cases that only need the JSON text (e.g. handing the result to
    /// another JSON-based tool or writing it to a file).
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The raw javabin-encoded response body.
    ///
    /// Returns
    /// -------
    /// str
    ///     The decoded value serialized as JSON, using the same shape as
    ///     ``deserialize`` (dicts for maps/documents, Solr-JSON-compatible
    ///     shape for document lists).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``data`` is not valid javabin.
    #[pyfunction]
    fn deserialize_json(data: &[u8]) -> PyResult<String> {
        let value = decoder::decode(data).map_err(decode_error_to_py)?;
        serde_json::to_string(&value.to_json())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize to JSON: {e}")))
    }

    /// Benchmark helper: decode into the intermediate Rust `Value` tree and
    /// return only the total number of scalar+container nodes, doing zero
    /// Python object construction. Lets us isolate pure byte-parsing cost
    /// from PyO3/CPython object-creation cost.
    #[pyfunction]
    fn _bench_parse_only(data: &[u8]) -> PyResult<usize> {
        let value = decoder::decode(data).map_err(decode_error_to_py)?;
        Ok(value::count_nodes(&value))
    }

    /// Deserialize a Solr `javabin` response directly into an Arrow
    /// ``RecordBatch``, given the target Arrow schema.
    ///
    /// Every document becomes a row; every schema field becomes a typed
    /// column. Absent fields become nulls. This skips per-value Python object
    /// construction entirely — the whole batch is handed back as a single
    /// zero-copy Arrow structure — so it is the fastest way to load a large
    /// tabular result into a DataFrame (``pl.from_arrow`` / ``pd.from_arrow``).
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     The raw javabin-encoded response body (``/select``, ``/export`` or
    ///     ``/stream``).
    /// schema : pyarrow.Schema
    ///     One field per desired column. Supported field types: int32, int64,
    ///     float32, float64, bool, string (utf8), binary, timestamp('ms'), and
    ///     ``list_`` of any of those for multi-valued Solr fields.
    ///
    /// Returns
    /// -------
    /// pyarrow.RecordBatch
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``data`` is not valid javabin, a value doesn't fit its column
    ///     type, the schema uses an unsupported type, or a document has child
    ///     documents (which a flat table can't represent).
    #[pyfunction]
    fn deserialize_arrow(
        py: Python<'_>,
        data: &[u8],
        schema: PyArrowType<arrow::datatypes::Schema>,
    ) -> PyResult<Py<PyAny>> {
        use std::sync::Arc;
        let schema = Arc::new(schema.0);
        let batch = py
            .detach(|| {
                let mut dec = arrow_decoder::ArrowDecoder::new(schema)?;
                dec.decode_response(data)?;
                dec.finish_batch()
            })
            .map_err(decode_error_to_py)?;
        Ok(batch.to_pyarrow(py)?.unbind())
    }

    #[pymodule_export]
    use super::StreamDecoder;

    #[pymodule_export]
    use super::ArrowStreamDecoder;
}

/// Incremental, chunk-fed streaming decoder for large `/export` responses.
///
/// Feed network chunks with :meth:`feed`; every complete document is passed
/// to the callback and released, so the internal byte buffer stays at roughly
/// one document regardless of the total response size (unlike
/// ``deserialize_stream``, which needs the whole body in memory). Call
/// :meth:`finish` when the stream ends to detect truncation.
///
/// Example
/// -------
/// ::
///
///     dec = javapyn.StreamDecoder()
///     with client.stream("POST", url, data=params) as resp:
///         for chunk in resp.iter_bytes():
///             dec.feed(chunk, handle_doc)
///     dec.finish()
///     print(dec.count, "documents")
#[pyclass(unsendable)]
struct StreamDecoder {
    state: stream_decoder::StreamState,
}

#[pymethods]
impl StreamDecoder {
    #[new]
    fn new() -> Self {
        Self {
            state: stream_decoder::StreamState::new(),
        }
    }

    /// Feed one chunk of bytes. Any documents now complete are decoded and
    /// passed to ``callback`` (a callable taking one argument). Exceptions
    /// raised by ``callback`` propagate unchanged.
    fn feed(&mut self, py: Python<'_>, chunk: &[u8], callback: &Bound<'_, PyAny>) -> PyResult<()> {
        self.state.feed(py, chunk, callback.as_ptr())
    }

    /// Signal end of input. Raises ``ValueError`` if the stream ended
    /// mid-document or before the document sequence completed.
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        self.state.finish(py)
    }

    /// Total number of documents streamed so far.
    #[getter]
    fn count(&self) -> u64 {
        self.state.count()
    }
}

/// Incremental, chunk-fed decoder that yields Arrow ``RecordBatch`` objects.
///
/// Combines the flat-memory chunk streaming of :class:`StreamDecoder` with the
/// columnar, object-free decoding of :func:`deserialize_arrow`: feed network
/// chunks, and get back full ``pyarrow.RecordBatch`` batches of ``batch_size``
/// rows as they complete. This is the most efficient way to load a huge
/// ``/export`` into a DataFrame without ever holding the whole thing in memory.
///
/// Example
/// -------
/// ::
///
///     import pyarrow as pa, polars as pl
///     schema = pa.schema([("title", pa.string()), ("rating", pa.float32())])
///     dec = javapyn.ArrowStreamDecoder(schema, batch_size=65536)
///     batches = []
///     with client.stream("POST", url, data=params) as resp:
///         for chunk in resp.iter_bytes():
///             batches.extend(dec.feed(chunk))
///     batches.extend(dec.finish())
///     df = pl.from_arrow(pa.Table.from_batches(batches, schema=schema))
#[pyclass(unsendable)]
struct ArrowStreamDecoder {
    state: arrow_decoder::ArrowStreamState,
}

#[pymethods]
impl ArrowStreamDecoder {
    #[new]
    #[pyo3(signature = (schema, batch_size = 65536))]
    fn new(schema: PyArrowType<arrow::datatypes::Schema>, batch_size: usize) -> PyResult<Self> {
        let schema = std::sync::Arc::new(schema.0);
        let state =
            arrow_decoder::ArrowStreamState::new(schema, batch_size).map_err(decode_error_to_py)?;
        Ok(Self { state })
    }

    /// Feed one chunk of bytes. Returns a list of any ``RecordBatch`` objects
    /// completed as a result (possibly empty).
    fn feed(&mut self, py: Python<'_>, chunk: &[u8]) -> PyResult<Vec<Py<PyAny>>> {
        let batches = py
            .detach(|| self.state.feed(chunk))
            .map_err(decode_error_to_py)?;
        batches
            .into_iter()
            .map(|b| Ok(b.to_pyarrow(py)?.unbind()))
            .collect()
    }

    /// Signal end of input; returns the final ``RecordBatch`` list. Raises
    /// ``ValueError`` on truncation.
    fn finish(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let batches = py
            .detach(|| self.state.finish())
            .map_err(decode_error_to_py)?;
        batches
            .into_iter()
            .map(|b| Ok(b.to_pyarrow(py)?.unbind()))
            .collect()
    }

    /// The Arrow schema this decoder produces.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        let schema: &arrow::datatypes::Schema = &self.state.schema();
        Ok(schema.to_pyarrow(py)?.unbind())
    }
}
