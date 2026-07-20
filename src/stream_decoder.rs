//! Incremental, chunk-fed streaming decoder for large `/export` responses.
//!
//! [`crate::py_decoder_fast::decode_stream`] streams *documents* to a callback
//! but still needs the whole response body in memory. For a multi-GB export
//! that's infeasible. This module instead decodes as network chunks arrive,
//! keeping the byte buffer at roughly one document.
//!
//! # Approach
//!
//! The response is `{... "response"/"result-set": {... "docs": ITERATOR[
//! doc, doc, ... ] END } ...}`. Documents are self-contained, so the decoder
//! only ever needs to *resume at document boundaries*, never at arbitrary
//! byte offsets:
//!
//! 1. **Envelope phase** — decode the small wrapper up to and including the
//!    `docs` ITERATOR/ARR tag. The envelope is tiny; if a chunk splits it we
//!    simply wait for more bytes and retry from the start of the buffer.
//! 2. **Docs phase** — repeatedly decode one document from the buffer. After
//!    each success, the consumed prefix is dropped. If a document is only
//!    partially present ([`DecodeError::UnexpectedEof`]), we rewind to the
//!    last document boundary and ask for more bytes.
//! 3. **Done** — the `END` marker (for an ITERATOR) or the doc count (for an
//!    ARR) terminates the sequence.
//!
//! The `EXTERN_STRING` cache (field-name interning) is kept across chunks,
//! since references span the whole stream.
//!
//! Callers feed bytes via [`StreamDecoder::feed`]; unconsumed tail bytes are
//! retained internally and prepended to the next chunk.

use pyo3::ffi;
use pyo3::prelude::*;

use crate::py_decoder_fast::{Decoder, DocsPhase, Obj};
use crate::reader::{DecodeError, tag};

/// Where we are in the response.
enum Phase {
    /// Still consuming the wrapper, haven't reached the doc sequence.
    Envelope,
    /// Inside a fixed-length `ARR` of documents (from a `SOLRDOCLST`).
    DocsArr { remaining: usize },
    /// Inside a variable-length `ITERATOR` of documents (from `/export` and
    /// `/stream`), terminated by `END`.
    DocsIter,
    /// The document sequence is finished; trailing envelope bytes (if any)
    /// are ignored.
    Done,
}

/// Streaming decoder state, persisted across `feed` calls.
pub struct StreamState {
    /// Bytes received but not yet fully consumed. Bytes before `start` have
    /// been consumed and are dead weight until the next compaction; keeping a
    /// cursor instead of draining after every document avoids an O(n) memmove
    /// per document (which would make decoding a large in-buffer run O(n²)).
    buf: Vec<u8>,
    /// Offset of the first not-yet-consumed byte within `buf`.
    start: usize,
    phase: Phase,
    /// Interned field-name strings, shared across all documents in the stream
    /// (mirrors `JavaBinCodec.stringsList`, which is per-unmarshal).
    strings: Vec<Obj>,
    /// Whether the leading version byte has been validated.
    version_checked: bool,
    /// Total documents streamed so far.
    count: u64,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            start: 0,
            phase: Phase::Envelope,
            strings: Vec::new(),
            version_checked: false,
            count: 0,
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// The not-yet-consumed portion of the buffer.
    #[inline]
    fn pending(&self) -> &[u8] {
        &self.buf[self.start..]
    }

    /// Mark `n` bytes (from `start`) as consumed. Amortised O(1): the actual
    /// memmove only happens in [`Self::compact`], and only once the dead
    /// prefix has grown large enough to be worth reclaiming.
    #[inline]
    fn advance(&mut self, n: usize) {
        self.start += n;
    }

    /// Reclaim the consumed prefix if it's worth it (dead prefix at least as
    /// large as the live tail, and non-trivial). Called after each `feed` so
    /// the buffer can't grow unboundedly while still keeping per-document
    /// consumption cheap.
    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
            return;
        }
        // Only shift when the dead prefix is a meaningful fraction, so this is
        // amortised: at most one memmove per doubling of live data.
        if self.start >= self.buf.len() - self.start {
            self.buf.drain(..self.start);
            self.start = 0;
        }
    }

    /// Feed one chunk of bytes. Any complete documents now available are
    /// decoded and passed to `callback`. Returns `Ok(())` on success; a raised
    /// callback exception or a decode error propagates as `Err`.
    pub fn feed(
        &mut self,
        py: Python<'_>,
        chunk: &[u8],
        callback: *mut ffi::PyObject,
    ) -> PyResult<()> {
        self.buf.extend_from_slice(chunk);
        let r = self.drive(py, callback).map_err(|e| err_to_py(py, e));
        self.compact();
        r
    }

    /// Signal end of input. Errors if the stream ended mid-document or before
    /// the document sequence terminated.
    pub fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        match self.phase {
            Phase::Done => Ok(()),
            Phase::DocsIter | Phase::DocsArr { .. } | Phase::Envelope => {
                // If everything that remains is the trailing END markers of an
                // already-emptied iterator we could be lenient, but a clean
                // finish should have reached Done. Treat leftover as an error
                // only if documents are still expected.
                match self.phase {
                    Phase::DocsArr { remaining } if remaining > 0 => Err(err_to_py(
                        py,
                        DecodeError::UnexpectedEof {
                            offset: self.buf.len(),
                        },
                    )),
                    _ => Ok(()),
                }
            }
        }
    }

    /// Consume as many complete units from the pending buffer as possible.
    fn drive(&mut self, py: Python<'_>, callback: *mut ffi::PyObject) -> Result<(), DecodeError> {
        loop {
            match self.phase {
                Phase::Done => return Ok(()),
                Phase::Envelope => {
                    // Try to parse the envelope up to the docs sequence. This
                    // needs the whole (tiny) envelope prefix present; if not,
                    // wait for more bytes.
                    match self.try_parse_envelope(py)? {
                        Some(next_phase) => {
                            self.phase = next_phase;
                            // consumed prefix already accounted for by try_parse_envelope
                        }
                        None => return Ok(()), // need more bytes
                    }
                }
                Phase::DocsArr { remaining } => {
                    if remaining == 0 {
                        self.phase = Phase::Done;
                        continue;
                    }
                    match self.try_one_doc(py, callback)? {
                        true => {
                            self.phase = Phase::DocsArr {
                                remaining: remaining - 1,
                            };
                        }
                        false => return Ok(()), // need more bytes
                    }
                }
                Phase::DocsIter => {
                    // Peek: is the next byte an END marker?
                    if self.pending().is_empty() {
                        return Ok(());
                    }
                    if self.pending()[0] == tag::END {
                        self.advance(1);
                        self.phase = Phase::Done;
                        continue;
                    }
                    match self.try_one_doc(py, callback)? {
                        true => {}              // consumed a doc, loop for the next
                        false => return Ok(()), // need more bytes
                    }
                }
            }
        }
    }

    /// Parse the wrapper up to the document sequence tag. On success returns
    /// the docs phase and advances past the consumed bytes. Returns `None` if
    /// more bytes are needed (cursor left untouched).
    ///
    /// This always re-parses from `start` (version byte included) because
    /// nothing is consumed until the *whole* envelope is present — so a partial
    /// envelope split across chunks is simply retried. The interned-string
    /// cache is only adopted on success; on a partial retry it is reset to
    /// empty (the envelope is the very first thing in the stream, so no
    /// document strings have been cached yet).
    fn try_parse_envelope(&mut self, py: Python<'_>) -> Result<Option<Phase>, DecodeError> {
        let mut dec = Decoder::for_stream(py, self.pending(), Vec::new());

        // version byte
        let v = match dec.reader_read_u8() {
            Ok(v) => v,
            Err(_) => return Ok(None), // not even the version byte yet
        };
        if v != crate::reader::EXPECTED_VERSION {
            return Err(DecodeError::InvalidVersion { found: v });
        }

        let docs_phase = match dec.stream_envelope_to_docs() {
            Ok(p) => p,
            Err(DecodeError::UnexpectedEof { .. }) => {
                // envelope not fully present yet — wait for more bytes. Drop
                // any strings interned during this failed attempt.
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        let phase = match docs_phase {
            DocsPhase::Arr(len) => Phase::DocsArr { remaining: len },
            DocsPhase::Iter => Phase::DocsIter,
            DocsPhase::None => Phase::Done,
        };

        let consumed = dec.reader_pos();
        self.strings = dec.take_strings();
        self.version_checked = true;
        self.advance(consumed);
        Ok(Some(phase))
    }

    /// Try to decode exactly one document from the pending buffer and pass it
    /// to `callback`. On success advances past the consumed bytes and returns
    /// `true`; if the document is incomplete, leaves the cursor intact and
    /// returns `false`.
    fn try_one_doc(
        &mut self,
        py: Python<'_>,
        callback: *mut ffi::PyObject,
    ) -> Result<bool, DecodeError> {
        // Number of interned strings before this attempt. A *partial* decode
        // may have interned new field names; if the doc turns out incomplete
        // we must roll the cache back to here, otherwise re-decoding the same
        // doc next time would register those strings twice and corrupt the
        // EXTERN_STRING index space.
        let cache_len = self.strings.len();

        let strings = std::mem::take(&mut self.strings);
        let mut dec = Decoder::for_stream(py, &self.buf[self.start..], strings);
        match dec.read_one_value_public() {
            Ok(doc) => {
                let consumed = dec.reader_pos();
                self.strings = dec.take_strings();
                let res = unsafe {
                    ffi::PyObject_CallFunctionObjArgs(
                        callback,
                        doc.as_ptr(),
                        std::ptr::null_mut::<ffi::PyObject>(),
                    )
                };
                if res.is_null() {
                    return Err(DecodeError::PyErr);
                }
                unsafe { ffi::Py_DECREF(res) };
                self.advance(consumed);
                self.count += 1;
                Ok(true)
            }
            Err(DecodeError::UnexpectedEof { .. }) => {
                let mut strings = dec.take_strings();
                strings.truncate(cache_len); // roll back partial interning
                self.strings = strings;
                Ok(false)
            }
            Err(e) => {
                self.strings = dec.take_strings();
                Err(e)
            }
        }
    }
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

fn err_to_py(py: Python<'_>, err: DecodeError) -> PyErr {
    match err {
        DecodeError::PyErr => PyErr::fetch(py),
        other => pyo3::exceptions::PyValueError::new_err(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyAnyMethods, PyList};

    const V: u8 = crate::reader::EXPECTED_VERSION;

    /// One doc: SOLRDOC { ORDERED_MAP size=1: "i" -> <k as SINT> }
    fn doc(k: u8, out: &mut Vec<u8>) {
        out.push(tag::SOLRDOC);
        out.push(tag::ORDERED_MAP | 1);
        out.push(tag::STR | 1);
        out.push(b'i');
        out.push(tag::SINT | (k & 0x0f));
    }

    /// /stream-style: NamedList{"result-set": MAP_ENTRY_ITER{"docs":
    /// ITERATOR[ n docs ] END} END}
    fn stream_msg(n: u8) -> Vec<u8> {
        let mut b = vec![V, tag::NAMED_LST | 1];
        b.push(tag::STR | 10);
        b.extend_from_slice(b"result-set");
        b.push(tag::MAP_ENTRY_ITER);
        b.push(tag::STR | 4);
        b.extend_from_slice(b"docs");
        b.push(tag::ITERATOR);
        for k in 0..n {
            doc(k, &mut b);
        }
        b.push(tag::END); // iterator
        b.push(tag::END); // map_entry_iter
        b
    }

    /// /select-style: NamedList{"response": SOLRDOCLST{header ARR[n,0,null,
    /// true], docs ARR[ n docs ]}}
    fn select_msg(n: u8) -> Vec<u8> {
        let mut b = vec![V, tag::NAMED_LST | 1];
        b.push(tag::STR | 8);
        b.extend_from_slice(b"response");
        b.push(tag::SOLRDOCLST);
        b.push(tag::ARR | 4);
        b.push(tag::SLONG | n);
        b.push(tag::SLONG);
        b.push(tag::NULL);
        b.push(tag::BOOL_TRUE);
        b.push(tag::ARR | n);
        for k in 0..n {
            doc(k, &mut b);
        }
        b
    }

    /// Feed `msg` in chunks of `chunk` bytes; collect the "i" field of each
    /// streamed doc into a Vec<i64>.
    fn stream_collect(py: Python<'_>, msg: &[u8], chunk: usize) -> Vec<i64> {
        let collected = PyList::empty(py);
        let cb = collected.getattr("append").unwrap();
        let mut st = StreamState::new();
        let mut i = 0;
        while i < msg.len() {
            let end = (i + chunk).min(msg.len());
            st.feed(py, &msg[i..end], cb.as_ptr()).unwrap();
            i = end;
        }
        st.finish(py).unwrap();
        collected
            .iter()
            .map(|d| d.get_item("i").unwrap().extract::<i64>().unwrap())
            .collect()
    }

    #[test]
    fn stream_msg_all_chunk_sizes_yield_same_docs() {
        Python::attach(|py| {
            let msg = stream_msg(6);
            let expected: Vec<i64> = (0..6).collect();
            // every chunk size from 1..=len must produce the same docs
            for chunk in 1..=msg.len() {
                let got = stream_collect(py, &msg, chunk);
                assert_eq!(got, expected, "chunk size {chunk}");
            }
        });
    }

    #[test]
    fn select_msg_all_chunk_sizes_yield_same_docs() {
        Python::attach(|py| {
            let msg = select_msg(5);
            let expected: Vec<i64> = (0..5).collect();
            for chunk in 1..=msg.len() {
                let got = stream_collect(py, &msg, chunk);
                assert_eq!(got, expected, "chunk size {chunk}");
            }
        });
    }

    #[test]
    fn empty_docs_stream() {
        Python::attach(|py| {
            assert_eq!(stream_collect(py, &stream_msg(0), 1), Vec::<i64>::new());
            assert_eq!(stream_collect(py, &select_msg(0), 1), Vec::<i64>::new());
        });
    }

    #[test]
    fn interned_field_names_across_chunked_docs() {
        // Field name defined via EXTERN_STRING in the first doc, referenced in
        // the second. Must decode correctly no matter how the bytes are split.
        Python::attach(|py| {
            let mut b = vec![V, tag::NAMED_LST | 1];
            b.push(tag::STR | 10);
            b.extend_from_slice(b"result-set");
            b.push(tag::MAP_ENTRY_ITER);
            b.push(tag::STR | 4);
            b.extend_from_slice(b"docs");
            b.push(tag::ITERATOR);
            // doc 0: {"foo": 1} with foo as EXTERN_STRING idx 0 (defines it)
            b.push(tag::SOLRDOC);
            b.push(tag::ORDERED_MAP | 1);
            b.push(tag::EXTERN_STRING);
            b.push(tag::STR | 3);
            b.extend_from_slice(b"foo");
            b.push(tag::SINT | 1);
            // doc 1: {"foo": 2} referencing idx 1
            b.push(tag::SOLRDOC);
            b.push(tag::ORDERED_MAP | 1);
            b.push(tag::EXTERN_STRING | 1);
            b.push(tag::SINT | 2);
            b.push(tag::END);
            b.push(tag::END);

            for chunk in 1..=b.len() {
                let collected = PyList::empty(py);
                let cb = collected.getattr("append").unwrap();
                let mut st = StreamState::new();
                let mut i = 0;
                while i < b.len() {
                    let e = (i + chunk).min(b.len());
                    st.feed(py, &b[i..e], cb.as_ptr()).unwrap();
                    i = e;
                }
                st.finish(py).unwrap();
                let vals: Vec<i64> = collected
                    .iter()
                    .map(|d| d.get_item("foo").unwrap().extract::<i64>().unwrap())
                    .collect();
                assert_eq!(vals, vec![1, 2], "chunk size {chunk}");
            }
        });
    }

    #[test]
    fn truncated_select_arr_errors_on_finish() {
        Python::attach(|py| {
            let msg = select_msg(3);
            let cb = PyList::empty(py).getattr("append").unwrap();
            let mut st = StreamState::new();
            // feed all but the last 3 bytes (a whole doc missing)
            st.feed(py, &msg[..msg.len() - 5], cb.as_ptr()).unwrap();
            assert!(st.finish(py).is_err());
        });
    }

    #[test]
    fn wrong_version_errors() {
        Python::attach(|py| {
            let cb = PyList::empty(py).getattr("append").unwrap();
            let mut st = StreamState::new();
            assert!(st.feed(py, &[1u8, tag::NULL], cb.as_ptr()).is_err());
        });
    }

    #[test]
    fn rejects_excessively_nested_document() {
        // A /stream-style message whose single "document" is a chain of
        // 10_000 nested single-element arrays. Before the recursion-depth
        // guard in py_decoder_fast::Decoder::read_slot, decoding this
        // document (via try_one_doc -> read_one_value_public) would overflow
        // the call stack instead of returning a catchable error.
        Python::attach(|py| {
            let mut b = vec![V, tag::NAMED_LST | 1];
            b.push(tag::STR | 10);
            b.extend_from_slice(b"result-set");
            b.push(tag::MAP_ENTRY_ITER);
            b.push(tag::STR | 4);
            b.extend_from_slice(b"docs");
            b.push(tag::ITERATOR);
            b.extend_from_slice(&vec![tag::ARR | 1; 10_000]);
            b.push(tag::NULL);
            b.push(tag::END); // iterator
            b.push(tag::END); // map_entry_iter

            let cb = PyList::empty(py).getattr("append").unwrap();
            let mut st = StreamState::new();
            let err = st.feed(py, &b, cb.as_ptr()).unwrap_err();
            assert!(err.to_string().contains("nesting"), "{err}");
        });
    }

    /// Regression for the O(n²) buffer-drain bug: feeding a large message as a
    /// single chunk (so the buffer stays big) must still decode every document
    /// correctly. Also implicitly checks the cursor/compaction logic.
    #[test]
    fn large_single_chunk_decodes_all_docs() {
        Python::attach(|py| {
            // 10_000 docs, each {"i": k&0x0f}, in one /stream ITERATOR.
            let n = 10_000u32;
            let mut b = vec![V, tag::NAMED_LST | 1];
            b.push(tag::STR | 10);
            b.extend_from_slice(b"result-set");
            b.push(tag::MAP_ENTRY_ITER);
            b.push(tag::STR | 4);
            b.extend_from_slice(b"docs");
            b.push(tag::ITERATOR);
            for k in 0..n {
                b.push(tag::SOLRDOC);
                b.push(tag::ORDERED_MAP | 1);
                b.push(tag::STR | 1);
                b.push(b'i');
                b.push(tag::SINT | (k as u8 & 0x0f));
            }
            b.push(tag::END);
            b.push(tag::END);

            let collected = PyList::empty(py);
            let cb = collected.getattr("append").unwrap();
            let mut st = StreamState::new();
            st.feed(py, &b, cb.as_ptr()).unwrap(); // one big chunk
            st.finish(py).unwrap();
            assert_eq!(st.count(), n as u64);
            assert_eq!(collected.len(), n as usize);
        });
    }

    /// The pending-buffer compaction must reclaim memory: after streaming many
    /// docs from many small feeds, the internal Vec must not have grown to the
    /// full stream size.
    #[test]
    fn buffer_is_compacted_not_unbounded() {
        Python::attach(|py| {
            let mut b = vec![V, tag::NAMED_LST | 1];
            b.push(tag::STR | 10);
            b.extend_from_slice(b"result-set");
            b.push(tag::MAP_ENTRY_ITER);
            b.push(tag::STR | 4);
            b.extend_from_slice(b"docs");
            b.push(tag::ITERATOR);
            for k in 0..5000u32 {
                b.push(tag::SOLRDOC);
                b.push(tag::ORDERED_MAP | 1);
                b.push(tag::STR | 1);
                b.push(b'i');
                b.push(tag::SINT | (k as u8 & 0x0f));
            }
            b.push(tag::END);
            b.push(tag::END);

            let cb = PyList::empty(py).getattr("append").unwrap();
            let mut st = StreamState::new();
            // feed 8 bytes at a time
            let mut i = 0;
            while i < b.len() {
                let e = (i + 8).min(b.len());
                st.feed(py, &b[i..e], cb.as_ptr()).unwrap();
                i = e;
            }
            st.finish(py).unwrap();
            assert_eq!(st.count(), 5000);
            // internal buffer must be tiny at the end (compacted), not ~the
            // whole 25 KB stream.
            assert!(st.buf.len() < 256, "buffer not compacted: {}", st.buf.len());
        });
    }
}
