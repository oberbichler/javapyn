//! Low-level byte-reading primitives shared by both the `Value`-tree decoder
//! ([`crate::decoder`], used for `deserialize_json`) and the direct-to-Python
//! decoder ([`crate::py_decoder`], used for `deserialize`).
//!
//! See [`crate::decoder`] for a description of the javabin tag layout.

/// Tag byte constants, named identically to the Java constants in
/// `JavaBinCodec` for easy cross-referencing.
#[allow(dead_code)]
pub mod tag {
    pub const NULL: u8 = 0;
    pub const BOOL_TRUE: u8 = 1;
    pub const BOOL_FALSE: u8 = 2;
    pub const BYTE: u8 = 3;
    pub const SHORT: u8 = 4;
    pub const DOUBLE: u8 = 5;
    pub const INT: u8 = 6;
    pub const LONG: u8 = 7;
    pub const FLOAT: u8 = 8;
    pub const DATE: u8 = 9;
    pub const MAP: u8 = 10;
    pub const SOLRDOC: u8 = 11;
    pub const SOLRDOCLST: u8 = 12;
    pub const BYTEARR: u8 = 13;
    pub const ITERATOR: u8 = 14;
    pub const END: u8 = 15;
    pub const SOLRINPUTDOC: u8 = 16;
    pub const MAP_ENTRY_ITER: u8 = 17;
    pub const ENUM_FIELD_VALUE: u8 = 18;
    pub const MAP_ENTRY: u8 = 19;
    pub const UUID: u8 = 20;
    pub const PRIMITIVE_ARR: u8 = 21;

    pub const STR: u8 = 1 << 5;
    pub const SINT: u8 = 2 << 5;
    pub const SLONG: u8 = 3 << 5;
    pub const ARR: u8 = 4 << 5;
    pub const ORDERED_MAP: u8 = 5 << 5;
    pub const NAMED_LST: u8 = 6 << 5;
    pub const EXTERN_STRING: u8 = 7 << 5;
}

pub const EXPECTED_VERSION: u8 = 2;

/// Maximum nesting depth (containers within containers: `ARR`, `MAP`,
/// `NAMED_LST`, `SOLRDOC`, `ITERATOR`, ...) permitted while decoding a single
/// javabin value.
///
/// Every decoder in this crate recurses once per nesting level. Without a
/// limit, a small adversarial or corrupted payload with enough nested
/// containers (tens of thousands of levels, a few dozen KB on the wire) can
/// overflow the call stack and crash the whole host process with an
/// uncatchable SIGSEGV -- no `try`/`except` in Python can recover from that,
/// which defeats the point of returning a `Result`/raising a Python
/// exception for every other malformed-input case. 128 levels is far beyond
/// any real Solr response shape (which nests at most a handful of levels
/// deep) while still keeping stack usage bounded to a small, safe multiple
/// of that.
pub const MAX_NESTING_DEPTH: u32 = 128;

/// Errors that can occur while decoding a javabin byte stream.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("unexpected end of input at byte offset {offset}")]
    UnexpectedEof { offset: usize },

    #[error(
        "invalid javabin version (expected {EXPECTED_VERSION}, found {found}); \
         the data may not be in javabin format"
    )]
    InvalidVersion { found: u8 },

    #[error("invalid UTF-8 string at byte offset {offset}: {source}")]
    InvalidUtf8 {
        offset: usize,
        #[source]
        source: std::str::Utf8Error,
    },

    #[error("unknown javabin tag byte 0x{tag:02x} at byte offset {offset}")]
    UnknownTag { tag: u8, offset: usize },

    #[error("expected {expected} at byte offset {offset}, found {found}")]
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
        offset: usize,
    },

    #[error("trailing data after decoded value: {remaining} unread byte(s)")]
    TrailingData { remaining: usize },

    #[error(
        "javabin value nesting exceeds the maximum supported depth ({max_depth}) \
         at byte offset {offset}; the input is likely malformed or malicious"
    )]
    NestingTooDeep { offset: usize, max_depth: u32 },

    /// A CPython C-API call failed and has already set a Python exception.
    /// Only produced by the `pyo3::ffi` fast path ([`crate::py_decoder_fast`]);
    /// carries no message of its own (the pending Python exception is used).
    #[error("python C-API call failed")]
    PyErr,

    // -- Arrow decoder errors -------------------------------------------------
    #[error("unsupported Arrow column type: {type_name}")]
    UnsupportedArrowType { type_name: String },

    #[error("javabin value does not fit the Arrow column type (expected {expected})")]
    ArrowValueMismatch { expected: &'static str },

    #[error(
        "document has child documents (_childDocuments_), which cannot be \
         represented in a flat Arrow table"
    )]
    ChildDocumentInArrow,

    #[error("failed to build Arrow RecordBatch: {msg}")]
    ArrowBuild { msg: String },
}

pub type Result<T> = std::result::Result<T, DecodeError>;

/// Stateful javabin byte reader over an in-memory byte slice.
///
/// Holds only the cursor position and raw byte primitives; the two decoders
/// ([`crate::decoder::Decoder`] and [`crate::py_decoder::Decoder`]) each add
/// their own string-cache and container-building logic on top.
pub struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or(DecodeError::UnexpectedEof { offset: self.pos })?;
        self.pos += 1;
        Ok(b)
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    pub fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.data.len())
            .ok_or(DecodeError::UnexpectedEof { offset: self.pos })?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        let b = self.read_exact(2)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_exact(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_exact(8)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_i32()? as u32))
    }

    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_i64()? as u64))
    }

    /// Variable-length unsigned int (7 bits per byte, MSB = continuation).
    pub fn read_vint(&mut self) -> Result<u32> {
        let mut b = self.read_u8()?;
        let mut i: u32 = (b & 0x7F) as u32;
        let mut shift = 7u32;
        while b & 0x80 != 0 {
            b = self.read_u8()?;
            i |= ((b & 0x7F) as u32) << shift;
            shift += 7;
        }
        Ok(i)
    }

    /// Variable-length unsigned long (7 bits per byte, MSB = continuation).
    pub fn read_vlong(&mut self) -> Result<u64> {
        let mut b = self.read_u8()?;
        let mut i: u64 = (b & 0x7F) as u64;
        let mut shift = 7u32;
        while b & 0x80 != 0 {
            b = self.read_u8()?;
            i |= ((b & 0x7F) as u64) << shift;
            shift += 7;
        }
        Ok(i)
    }

    /// Decode the "tag + embedded size" scheme shared by `STR`, `ARR`,
    /// `ORDERED_MAP`, `NAMED_LST` and `EXTERN_STRING` (as a size/index),
    /// given the tag byte that was already read.
    pub fn read_size(&mut self, tag_byte: u8) -> Result<usize> {
        let mut sz = (tag_byte & 0x1f) as usize;
        if sz == 0x1f {
            sz += self.read_vint()? as usize;
        }
        Ok(sz)
    }

    /// Read `len` bytes and borrow-validate them as UTF-8, without copying.
    pub fn read_utf8_slice(&mut self, len: usize) -> Result<&'a str> {
        let start = self.pos;
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes).map_err(|source| DecodeError::InvalidUtf8 {
            offset: start,
            source,
        })
    }

    /// Read a `STR`-tagged string's raw UTF-8 bytes, given the already
    /// consumed tag byte.
    pub fn read_str_tagged(&mut self, tag_byte: u8) -> Result<&'a str> {
        let len = self.read_size(tag_byte)?;
        self.read_utf8_slice(len)
    }

    /// Number of bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Bound a wire-supplied element count (e.g. a `MAP`/`ARR`/`NAMED_LST`
    /// size prefix, which a malicious or corrupted payload can set to almost
    /// `u32::MAX`) down to a safe `Vec::with_capacity` hint.
    ///
    /// Every element of every container this crate decodes consumes at least
    /// one byte off the wire, so `claimed` can never legitimately exceed the
    /// number of bytes left in the input. Without this cap, a payload of only
    /// a few bytes could claim e.g. 4 billion entries and force an immediate
    /// multi-gigabyte allocation attempt before a single byte of the
    /// (nonexistent) elements is even read -- on some allocators/platforms
    /// (notably under strict overcommit settings, e.g. some containers) that
    /// aborts the process instead of raising a catchable error. Capping the
    /// hint here does not change decoding behavior for truthful input: the
    /// element-reading loop still runs `claimed` times and still produces the
    /// usual `UnexpectedEof` if the container turns out to be shorter than
    /// declared; this only bounds the *initial* allocation.
    pub fn capacity_hint(&self, claimed: usize) -> usize {
        claimed.min(self.remaining())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_hint_caps_an_oversized_claim_to_remaining_bytes() {
        let data = [0u8; 10];
        let r = Reader::new(&data);
        assert_eq!(r.capacity_hint(1_000_000_000), 10);
    }

    #[test]
    fn capacity_hint_does_not_reduce_a_truthful_claim() {
        let data = [0u8; 10];
        let r = Reader::new(&data);
        assert_eq!(r.capacity_hint(3), 3);
    }

    #[test]
    fn capacity_hint_accounts_for_already_consumed_bytes() {
        let data = [0u8; 10];
        let mut r = Reader::new(&data);
        r.read_exact(4).unwrap();
        assert_eq!(r.remaining(), 6);
        assert_eq!(r.capacity_hint(1_000_000_000), 6);
    }
}
