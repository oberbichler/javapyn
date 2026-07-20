//! Direct-to-Arrow decoder: decodes a javabin document sequence straight into
//! columnar Arrow arrays, without building one Python object per value.
//!
//! # Why
//!
//! [`crate::py_decoder_fast`] is fast, but for a large tabular result the cost
//! is dominated by constructing ~one `PyObject` per field per row. When the
//! caller ultimately wants a DataFrame, that is wasted work: Arrow stores each
//! column as a single typed buffer. This module appends values straight into
//! typed [`arrow`] builders, so a whole `RecordBatch` is handed to Python as
//! *one* zero-copy C-Data-Interface capsule instead of millions of objects.
//!
//! The heavy decode loop touches no Python API, so callers run it under
//! `Python::allow_threads`.
//!
//! # Schema
//!
//! The caller supplies the target Arrow schema (typically derived from the
//! Solr collection schema). Every field in `fl` maps to one column. Supported
//! column types mirror the Solr field types actually used by `SOLR_*`
//! collections:
//!
//! | Arrow `DataType`                    | javabin value tag                 |
//! |-------------------------------------|-----------------------------------|
//! | `Int32`                             | `SINT` / `INT`                    |
//! | `Int64`                             | `SLONG` / `LONG`                  |
//! | `Float32`                           | `FLOAT`                           |
//! | `Float64`                           | `DOUBLE`                          |
//! | `Boolean`                           | `BOOL_TRUE` / `BOOL_FALSE`        |
//! | `Utf8`                              | `STR` / `EXTERN_STRING`           |
//! | `Binary`                            | `BYTEARR`                         |
//! | `Timestamp(Millisecond, None)`      | `DATE`                            |
//! | `List<one of the above>`            | multi-valued field (`ARR`)        |
//!
//! A field that is absent from a given document becomes a `null` in that
//! column (Solr documents are sparse). A document's `_childDocuments_` cannot
//! be represented in a flat table and is rejected.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int32Builder,
    Int64Builder, ListBuilder, StringBuilder, TimestampMillisecondBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;

use crate::reader::{DecodeError, Reader, tag};

type Result<T> = std::result::Result<T, DecodeError>;

/// A javabin scalar value already read from the stream, ready to be appended
/// to a column (or to be an element of a list column). Borrows string/byte
/// data from the input buffer to avoid copying until the Arrow builder copies.
enum Scalar<'a> {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// DATE: milliseconds since the Unix epoch.
    Date(i64),
    Str(&'a str),
    Bytes(&'a [u8]),
}

/// One output column's typed builder.
enum ColumnBuilder {
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Boolean(BooleanBuilder),
    Utf8(StringBuilder),
    Binary(BinaryBuilder),
    TimestampMillis(TimestampMillisecondBuilder),
    ListInt32(ListBuilder<Int32Builder>),
    ListInt64(ListBuilder<Int64Builder>),
    ListFloat32(ListBuilder<Float32Builder>),
    ListFloat64(ListBuilder<Float64Builder>),
    ListBoolean(ListBuilder<BooleanBuilder>),
    ListUtf8(ListBuilder<StringBuilder>),
    ListTimestampMillis(ListBuilder<TimestampMillisecondBuilder>),
}

impl ColumnBuilder {
    /// Create an empty builder for `field`'s Arrow type, or an error if the
    /// type is unsupported.
    fn for_field(field: &Field) -> Result<Self> {
        let scalar = |dt: &DataType| -> Option<ColumnBuilder> {
            Some(match dt {
                DataType::Int32 => ColumnBuilder::Int32(Int32Builder::new()),
                DataType::Int64 => ColumnBuilder::Int64(Int64Builder::new()),
                DataType::Float32 => ColumnBuilder::Float32(Float32Builder::new()),
                DataType::Float64 => ColumnBuilder::Float64(Float64Builder::new()),
                DataType::Boolean => ColumnBuilder::Boolean(BooleanBuilder::new()),
                DataType::Utf8 => ColumnBuilder::Utf8(StringBuilder::new()),
                DataType::Binary => ColumnBuilder::Binary(BinaryBuilder::new()),
                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    ColumnBuilder::TimestampMillis(TimestampMillisecondBuilder::new())
                }
                _ => return None,
            })
        };

        match field.data_type() {
            DataType::List(inner) => {
                let b = match inner.data_type() {
                    DataType::Int32 => {
                        ColumnBuilder::ListInt32(ListBuilder::new(Int32Builder::new()))
                    }
                    DataType::Int64 => {
                        ColumnBuilder::ListInt64(ListBuilder::new(Int64Builder::new()))
                    }
                    DataType::Float32 => {
                        ColumnBuilder::ListFloat32(ListBuilder::new(Float32Builder::new()))
                    }
                    DataType::Float64 => {
                        ColumnBuilder::ListFloat64(ListBuilder::new(Float64Builder::new()))
                    }
                    DataType::Boolean => {
                        ColumnBuilder::ListBoolean(ListBuilder::new(BooleanBuilder::new()))
                    }
                    DataType::Utf8 => {
                        ColumnBuilder::ListUtf8(ListBuilder::new(StringBuilder::new()))
                    }
                    DataType::Timestamp(TimeUnit::Millisecond, _) => {
                        ColumnBuilder::ListTimestampMillis(ListBuilder::new(
                            TimestampMillisecondBuilder::new(),
                        ))
                    }
                    other => {
                        return Err(DecodeError::UnsupportedArrowType {
                            type_name: format!("List<{other}>"),
                        });
                    }
                };
                Ok(b)
            }
            dt => scalar(dt).ok_or_else(|| DecodeError::UnsupportedArrowType {
                type_name: dt.to_string(),
            }),
        }
    }

    /// Append a null for a missing/absent field.
    fn append_null(&mut self) {
        match self {
            ColumnBuilder::Int32(b) => b.append_null(),
            ColumnBuilder::Int64(b) => b.append_null(),
            ColumnBuilder::Float32(b) => b.append_null(),
            ColumnBuilder::Float64(b) => b.append_null(),
            ColumnBuilder::Boolean(b) => b.append_null(),
            ColumnBuilder::Utf8(b) => b.append_null(),
            ColumnBuilder::Binary(b) => b.append_null(),
            ColumnBuilder::TimestampMillis(b) => b.append_null(),
            ColumnBuilder::ListInt32(b) => b.append_null(),
            ColumnBuilder::ListInt64(b) => b.append_null(),
            ColumnBuilder::ListFloat32(b) => b.append_null(),
            ColumnBuilder::ListFloat64(b) => b.append_null(),
            ColumnBuilder::ListBoolean(b) => b.append_null(),
            ColumnBuilder::ListUtf8(b) => b.append_null(),
            ColumnBuilder::ListTimestampMillis(b) => b.append_null(),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            ColumnBuilder::Int32(b) => Arc::new(b.finish()),
            ColumnBuilder::Int64(b) => Arc::new(b.finish()),
            ColumnBuilder::Float32(b) => Arc::new(b.finish()),
            ColumnBuilder::Float64(b) => Arc::new(b.finish()),
            ColumnBuilder::Boolean(b) => Arc::new(b.finish()),
            ColumnBuilder::Utf8(b) => Arc::new(b.finish()),
            ColumnBuilder::Binary(b) => Arc::new(b.finish()),
            ColumnBuilder::TimestampMillis(b) => Arc::new(b.finish()),
            ColumnBuilder::ListInt32(b) => Arc::new(b.finish()),
            ColumnBuilder::ListInt64(b) => Arc::new(b.finish()),
            ColumnBuilder::ListFloat32(b) => Arc::new(b.finish()),
            ColumnBuilder::ListFloat64(b) => Arc::new(b.finish()),
            ColumnBuilder::ListBoolean(b) => Arc::new(b.finish()),
            ColumnBuilder::ListUtf8(b) => Arc::new(b.finish()),
            ColumnBuilder::ListTimestampMillis(b) => Arc::new(b.finish()),
        }
    }

    fn is_list(&self) -> bool {
        matches!(
            self,
            ColumnBuilder::ListInt32(_)
                | ColumnBuilder::ListInt64(_)
                | ColumnBuilder::ListFloat32(_)
                | ColumnBuilder::ListFloat64(_)
                | ColumnBuilder::ListBoolean(_)
                | ColumnBuilder::ListUtf8(_)
                | ColumnBuilder::ListTimestampMillis(_)
        )
    }
}

/// Coerce a decoded [`Scalar`] into a scalar (non-list) column, erroring on a
/// type it cannot represent. `Null` appends a null.
fn append_scalar(col: &mut ColumnBuilder, v: Scalar<'_>) -> Result<()> {
    macro_rules! type_err {
        ($want:expr) => {
            Err(DecodeError::ArrowValueMismatch { expected: $want })
        };
    }
    match col {
        ColumnBuilder::Int32(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Int(x) => b.append_value(x),
            // /stream encodes small integers as (compact) longs; accept when
            // they fit an i32.
            Scalar::Long(x) if i32::try_from(x).is_ok() => b.append_value(x as i32),
            _ => return type_err!("int32"),
        },
        ColumnBuilder::Int64(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Long(x) => b.append_value(x),
            Scalar::Int(x) => b.append_value(x as i64),
            _ => return type_err!("int64"),
        },
        ColumnBuilder::Float32(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Float(x) => b.append_value(x),
            Scalar::Double(x) => b.append_value(x as f32),
            _ => return type_err!("float32"),
        },
        ColumnBuilder::Float64(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Double(x) => b.append_value(x),
            Scalar::Float(x) => b.append_value(x as f64),
            _ => return type_err!("float64"),
        },
        ColumnBuilder::Boolean(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Bool(x) => b.append_value(x),
            _ => return type_err!("boolean"),
        },
        ColumnBuilder::Utf8(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Str(s) => b.append_value(s),
            _ => return type_err!("utf8"),
        },
        ColumnBuilder::Binary(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Bytes(bytes) => b.append_value(bytes),
            _ => return type_err!("binary"),
        },
        ColumnBuilder::TimestampMillis(b) => match v {
            Scalar::Null => b.append_null(),
            Scalar::Date(x) => b.append_value(x),
            Scalar::Long(x) => b.append_value(x),
            _ => return type_err!("timestamp(ms)"),
        },
        _ => return Err(DecodeError::ArrowValueMismatch { expected: "list" }),
    }
    Ok(())
}

/// Append one element of a multi-valued field into a list builder's inner
/// builder (the caller has already opened the list slot).
fn append_list_element(col: &mut ColumnBuilder, v: Scalar<'_>) -> Result<()> {
    macro_rules! type_err {
        ($want:expr) => {
            Err(DecodeError::ArrowValueMismatch { expected: $want })
        };
    }
    match col {
        ColumnBuilder::ListInt32(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Int(x) => b.values().append_value(x),
            Scalar::Long(x) if i32::try_from(x).is_ok() => b.values().append_value(x as i32),
            _ => return type_err!("list<int32>"),
        },
        ColumnBuilder::ListInt64(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Long(x) => b.values().append_value(x),
            Scalar::Int(x) => b.values().append_value(x as i64),
            _ => return type_err!("list<int64>"),
        },
        ColumnBuilder::ListFloat32(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Float(x) => b.values().append_value(x),
            Scalar::Double(x) => b.values().append_value(x as f32),
            _ => return type_err!("list<float32>"),
        },
        ColumnBuilder::ListFloat64(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Double(x) => b.values().append_value(x),
            Scalar::Float(x) => b.values().append_value(x as f64),
            _ => return type_err!("list<float64>"),
        },
        ColumnBuilder::ListBoolean(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Bool(x) => b.values().append_value(x),
            _ => return type_err!("list<boolean>"),
        },
        ColumnBuilder::ListUtf8(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Str(s) => b.values().append_value(s),
            _ => return type_err!("list<utf8>"),
        },
        ColumnBuilder::ListTimestampMillis(b) => match v {
            Scalar::Null => b.values().append_null(),
            Scalar::Date(x) => b.values().append_value(x),
            Scalar::Long(x) => b.values().append_value(x),
            _ => return type_err!("list<timestamp(ms)>"),
        },
        _ => return Err(DecodeError::ArrowValueMismatch { expected: "scalar" }),
    }
    Ok(())
}

/// Finish the current list slot of a list column (marks the row present).
fn finish_list_row(col: &mut ColumnBuilder) {
    match col {
        ColumnBuilder::ListInt32(b) => b.append(true),
        ColumnBuilder::ListInt64(b) => b.append(true),
        ColumnBuilder::ListFloat32(b) => b.append(true),
        ColumnBuilder::ListFloat64(b) => b.append(true),
        ColumnBuilder::ListBoolean(b) => b.append(true),
        ColumnBuilder::ListUtf8(b) => b.append(true),
        ColumnBuilder::ListTimestampMillis(b) => b.append(true),
        _ => {}
    }
}

/// The columnar decoder: one builder per schema field, plus a name→column
/// index map and the javabin `EXTERN_STRING` cache resolved to column indices.
pub struct ArrowDecoder {
    schema: Arc<Schema>,
    builders: Vec<ColumnBuilder>,
    /// Field name -> column index.
    name_to_col: std::collections::HashMap<String, usize>,
    /// `EXTERN_STRING` cache: for each interned string (in encounter order),
    /// its resolved column index (or `None` if it's not a schema field / not a
    /// field-name position). Also stores the string itself so non-cached
    /// lookups and value strings still work.
    extern_strings: Vec<String>,
    /// Per-batch: which columns have received a value for the current row.
    row_seen: Vec<bool>,
    /// Number of complete rows appended.
    rows: usize,
}

impl ArrowDecoder {
    pub fn new(schema: Arc<Schema>) -> Result<Self> {
        let mut builders = Vec::with_capacity(schema.fields().len());
        let mut name_to_col = std::collections::HashMap::with_capacity(schema.fields().len());
        for (i, field) in schema.fields().iter().enumerate() {
            builders.push(ColumnBuilder::for_field(field)?);
            name_to_col.insert(field.name().clone(), i);
        }
        let n = builders.len();
        Ok(Self {
            schema,
            builders,
            name_to_col,
            extern_strings: Vec::new(),
            row_seen: vec![false; n],
            rows: 0,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Decode every document in `data` (a complete javabin response) into the
    /// columns. Returns the number of documents appended.
    ///
    /// This walks the envelope to the `docs` sequence, then appends each
    /// document as a row. Used for the single-shot path.
    pub fn decode_response(&mut self, data: &[u8]) -> Result<usize> {
        let mut reader = Reader::new(data);
        let version = reader.read_u8()?;
        if version != crate::reader::EXPECTED_VERSION {
            return Err(DecodeError::InvalidVersion { found: version });
        }
        let phase = self.envelope_to_docs(&mut reader)?;
        let before = self.rows;
        match phase {
            DocsSeq::Arr(n) => {
                for _ in 0..n {
                    self.append_document(&mut reader)?;
                }
            }
            DocsSeq::Iter => loop {
                if reader.pos >= reader.data.len() {
                    return Err(DecodeError::UnexpectedEof { offset: reader.pos });
                }
                if reader.data[reader.pos] == tag::END {
                    // END terminates the doc iterator; the loop exits and the
                    // reader is not used again, so no need to advance past it.
                    break;
                }
                self.append_document(&mut reader)?;
            },
            DocsSeq::None => {}
        }
        Ok(self.rows - before)
    }

    /// Consume the response envelope up to the `docs` sequence, returning its
    /// kind. Mirrors the py_decoder_fast streaming envelope walk but discards
    /// envelope values without building Python objects.
    fn envelope_to_docs(&mut self, r: &mut Reader<'_>) -> Result<DocsSeq> {
        let t = r.read_u8()?;
        let hi = t >> 5;
        match hi {
            5 | 6 => {
                let sz = r.read_size(t)?;
                for _ in 0..sz {
                    let name = self.read_field_name(r)?;
                    if let Some(p) = self.envelope_value(r, &name)? {
                        return Ok(p);
                    }
                }
            }
            _ if t == tag::MAP_ENTRY_ITER => loop {
                if self.peek_end(r)? {
                    break;
                }
                let name = self.read_field_name(r)?;
                if let Some(p) = self.envelope_value(r, &name)? {
                    return Ok(p);
                }
            },
            _ => return Ok(DocsSeq::None),
        }
        Ok(DocsSeq::None)
    }

    fn envelope_value(&mut self, r: &mut Reader<'_>, key: &str) -> Result<Option<DocsSeq>> {
        if key == "response" || key == "result-set" {
            let t = r.read_u8()?;
            let hi = t >> 5;
            match hi {
                5 | 6 => {
                    let sz = r.read_size(t)?;
                    for _ in 0..sz {
                        let name = self.read_field_name(r)?;
                        if let Some(p) = self.docs_entry(r, &name)? {
                            return Ok(Some(p));
                        }
                    }
                    return Ok(Some(DocsSeq::None));
                }
                _ if t == tag::SOLRDOCLST => {
                    self.skip_value(r)?; // header array
                    let dt = r.read_u8()?;
                    if dt >> 5 == 4 {
                        return Ok(Some(DocsSeq::Arr(r.read_size(dt)?)));
                    }
                    return Ok(Some(DocsSeq::None));
                }
                _ if t == tag::MAP_ENTRY_ITER => {
                    loop {
                        if self.peek_end(r)? {
                            break;
                        }
                        let name = self.read_field_name(r)?;
                        if let Some(p) = self.docs_entry(r, &name)? {
                            return Ok(Some(p));
                        }
                    }
                    return Ok(Some(DocsSeq::None));
                }
                _ => {
                    r.pos -= 1;
                    self.skip_value(r)?;
                    return Ok(None);
                }
            }
        }
        self.skip_value(r)?;
        Ok(None)
    }

    fn docs_entry(&mut self, r: &mut Reader<'_>, name: &str) -> Result<Option<DocsSeq>> {
        if name == "docs" {
            let t = r.read_u8()?;
            if t >> 5 == 4 {
                return Ok(Some(DocsSeq::Arr(r.read_size(t)?)));
            }
            if t == tag::ITERATOR {
                return Ok(Some(DocsSeq::Iter));
            }
            r.pos -= 1;
            self.skip_value(r)?;
            return Ok(Some(DocsSeq::None));
        }
        self.skip_value(r)?;
        Ok(None)
    }

    fn peek_end(&mut self, r: &mut Reader<'_>) -> Result<bool> {
        let b = *r
            .data
            .get(r.pos)
            .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
        if b == tag::END {
            r.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Append one document (row): read its fields, route each to its column,
    /// then null-fill columns not present in this document.
    ///
    /// Documents appear in several encodings depending on the handler:
    /// `SOLRDOC` (`/select`), or a plain map — `ORDERED_MAP`/`NAMED_LST`
    /// (fixed length) or `MAP_ENTRY_ITER` (`END`-terminated) — for `/stream`
    /// result-set rows (including the trailing `{"EOF":true,...}` marker). All
    /// are handled as field-name/value sequences.
    pub fn append_document(&mut self, r: &mut Reader<'_>) -> Result<()> {
        for s in self.row_seen.iter_mut() {
            *s = false;
        }

        let doc_tag = r.read_u8()?;
        let hi = doc_tag >> 5;
        if doc_tag == tag::SOLRDOC {
            let inner = r.read_u8()?;
            let sz = r.read_size(inner)?;
            self.append_fields_fixed(r, sz)?;
        } else if hi == 5 || hi == 6 {
            // ORDERED_MAP / NAMED_LST as a document
            let sz = r.read_size(doc_tag)?;
            self.append_fields_fixed(r, sz)?;
        } else if doc_tag == tag::MAP_ENTRY_ITER {
            self.append_fields_iter(r)?;
        } else {
            return Err(DecodeError::TypeMismatch {
                expected: "SolrDocument or map",
                found: "other tag",
                offset: r.pos - 1,
            });
        }

        // null-fill columns not seen this row
        for (i, seen) in self.row_seen.iter().enumerate() {
            if !*seen {
                self.builders[i].append_null();
            }
        }
        self.rows += 1;
        Ok(())
    }

    /// Read `sz` field entries (name + value, or a child SOLRDOC) into columns.
    fn append_fields_fixed(&mut self, r: &mut Reader<'_>, sz: usize) -> Result<()> {
        for _ in 0..sz {
            let (col_idx, is_field) = self.read_entry_key(r)?;
            if !is_field {
                return Err(DecodeError::ChildDocumentInArrow);
            }
            self.route_field(r, col_idx)?;
        }
        Ok(())
    }

    /// Read `END`-terminated field entries (from a `MAP_ENTRY_ITER` document).
    fn append_fields_iter(&mut self, r: &mut Reader<'_>) -> Result<()> {
        loop {
            let peek = *r
                .data
                .get(r.pos)
                .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
            if peek == tag::END {
                r.pos += 1;
                break;
            }
            let (col_idx, is_field) = self.read_entry_key(r)?;
            if !is_field {
                return Err(DecodeError::ChildDocumentInArrow);
            }
            self.route_field(r, col_idx)?;
        }
        Ok(())
    }

    /// Route one field's value into its column (or skip if not in schema).
    fn route_field(&mut self, r: &mut Reader<'_>, col_idx: Option<usize>) -> Result<()> {
        match col_idx {
            Some(i) => {
                self.read_value_into_column(r, i)?;
                self.row_seen[i] = true;
            }
            None => {
                self.skip_value(r)?;
            }
        }
        Ok(())
    }

    /// Read a document-entry key. Returns `(column index or None, is_field)`.
    /// `is_field == false` means the entry was actually a child `SOLRDOC`.
    fn read_entry_key(&mut self, r: &mut Reader<'_>) -> Result<(Option<usize>, bool)> {
        let t = r.read_u8()?;
        let hi = t >> 5;
        match hi {
            1 => {
                let s = r.read_str_tagged(t)?;
                Ok((self.name_to_col.get(s).copied(), true))
            }
            7 => {
                let idx = r.read_size(t)?;
                if idx != 0 {
                    let s = self
                        .extern_strings
                        .get(idx - 1)
                        .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
                    Ok((self.name_to_col.get(s).copied(), true))
                } else {
                    let inner = r.read_u8()?;
                    let s = r.read_str_tagged(inner)?.to_string();
                    let col = self.name_to_col.get(&s).copied();
                    self.extern_strings.push(s);
                    Ok((col, true))
                }
            }
            0 if t == tag::SOLRDOC => Ok((None, false)),
            _ => Err(DecodeError::TypeMismatch {
                expected: "field name or child document",
                found: "other tag",
                offset: r.pos - 1,
            }),
        }
    }

    /// Read a field name (STR/EXTERN_STRING) as an owned string, used for
    /// envelope keys.
    fn read_field_name(&mut self, r: &mut Reader<'_>) -> Result<String> {
        let t = r.read_u8()?;
        let hi = t >> 5;
        match hi {
            1 => Ok(r.read_str_tagged(t)?.to_string()),
            7 => {
                let idx = r.read_size(t)?;
                if idx != 0 {
                    self.extern_strings
                        .get(idx - 1)
                        .cloned()
                        .ok_or(DecodeError::UnexpectedEof { offset: r.pos })
                } else {
                    let inner = r.read_u8()?;
                    let s = r.read_str_tagged(inner)?.to_string();
                    self.extern_strings.push(s.clone());
                    Ok(s)
                }
            }
            _ => Err(DecodeError::TypeMismatch {
                expected: "field name (string)",
                found: "other tag",
                offset: r.pos - 1,
            }),
        }
    }

    /// Read one javabin value and append it to column `i`. For a list column
    /// the value must be an `ARR`; for a scalar column it must be a matching
    /// scalar.
    fn read_value_into_column(&mut self, r: &mut Reader<'_>, i: usize) -> Result<()> {
        if self.builders[i].is_list() {
            let t = r.read_u8()?;
            if t >> 5 == 4 {
                // ARR: fixed-length multi-valued field.
                let n = r.read_size(t)?;
                for _ in 0..n {
                    let v = read_scalar(r)?;
                    append_list_element(&mut self.builders[i], v)?;
                }
                finish_list_row(&mut self.builders[i]);
                Ok(())
            } else if t == tag::ITERATOR {
                // ITERATOR: /export emits some multi-valued docValues fields as
                // an END-terminated sequence rather than a fixed-length ARR.
                loop {
                    let peek = *r
                        .data
                        .get(r.pos)
                        .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
                    if peek == tag::END {
                        r.pos += 1;
                        break;
                    }
                    let v = read_scalar(r)?;
                    append_list_element(&mut self.builders[i], v)?;
                }
                finish_list_row(&mut self.builders[i]);
                Ok(())
            } else if t == tag::NULL {
                // absent multi-valued field
                self.builders[i].append_null();
                Ok(())
            } else {
                // A multi-valued field encoded as a single scalar: accept it as
                // a one-element list for robustness.
                r.pos -= 1;
                let v = read_scalar(r)?;
                append_list_element(&mut self.builders[i], v)?;
                finish_list_row(&mut self.builders[i]);
                Ok(())
            }
        } else {
            // Scalar column. Solr's /export can emit a single-valued docValues
            // field as a one-element ARR/ITERATOR; take the first element.
            let t = r.read_u8()?;
            if t >> 5 == 4 {
                let n = r.read_size(t)?;
                if n == 0 {
                    append_scalar(&mut self.builders[i], Scalar::Null)?;
                } else {
                    let first = read_scalar(r)?;
                    append_scalar(&mut self.builders[i], first)?;
                    for _ in 1..n {
                        let _ = read_scalar(r)?;
                    }
                }
                Ok(())
            } else if t == tag::ITERATOR {
                let mut first: Option<Scalar> = None;
                loop {
                    let peek = *r
                        .data
                        .get(r.pos)
                        .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
                    if peek == tag::END {
                        r.pos += 1;
                        break;
                    }
                    let v = read_scalar(r)?;
                    if first.is_none() {
                        first = Some(v);
                    }
                }
                append_scalar(&mut self.builders[i], first.unwrap_or(Scalar::Null))
            } else {
                r.pos -= 1;
                let v = read_scalar(r)?;
                append_scalar(&mut self.builders[i], v)
            }
        }
    }

    /// Skip one javabin value without materialising it (for non-schema fields
    /// and envelope values).
    fn skip_value(&mut self, r: &mut Reader<'_>) -> Result<()> {
        skip_value(r, &mut self.extern_strings, 0)
    }

    /// Finish the current builders into a `RecordBatch` and reset row count.
    pub fn finish_batch(&mut self) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = self.builders.iter_mut().map(|b| b.finish()).collect();
        self.rows = 0;
        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| DecodeError::ArrowBuild { msg: e.to_string() })
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Clear the `EXTERN_STRING` cache (used by the streaming path when it
    /// re-parses an envelope that wasn't yet fully buffered).
    pub fn reset_extern_cache(&mut self) {
        self.extern_strings.clear();
    }
}

/// Which kind of document sequence follows the envelope.
enum DocsSeq {
    Arr(usize),
    Iter,
    None,
}

/// Read exactly one javabin scalar value (borrowing str/bytes from the input).
/// Compact ints/longs, full ints/longs, floats, doubles, dates, bools, null,
/// strings, extern strings (resolved to their bytes) and byte arrays are all
/// supported; anything else (nested containers as a scalar) is an error.
fn read_scalar<'a>(r: &mut Reader<'a>) -> Result<Scalar<'a>> {
    let start = r.pos;
    let t = r.read_u8()?;
    let hi = t >> 5;
    if hi != 0 {
        return Ok(match hi {
            1 => Scalar::Str(r.read_str_tagged(t)?),
            2 => Scalar::Int(read_small_int(r, t)?),
            3 => Scalar::Long(read_small_long(r, t)?),
            7 => {
                // extern string as a *value* is unusual, but resolve it:
                let idx = r.read_size(t)?;
                if idx != 0 {
                    return Err(DecodeError::ArrowValueMismatch {
                        expected: "inline value (unexpected extern-string reference)",
                    });
                }
                let inner = r.read_u8()?;
                Scalar::Str(r.read_str_tagged(inner)?)
            }
            _ => {
                return Err(DecodeError::TypeMismatch {
                    expected: "scalar value",
                    found: "container",
                    offset: start,
                });
            }
        });
    }
    Ok(match t {
        tag::NULL => Scalar::Null,
        tag::BOOL_TRUE => Scalar::Bool(true),
        tag::BOOL_FALSE => Scalar::Bool(false),
        tag::BYTE => Scalar::Int(r.read_i8()? as i32),
        tag::SHORT => Scalar::Int(r.read_i16()? as i32),
        tag::INT => Scalar::Int(r.read_i32()?),
        tag::LONG => Scalar::Long(r.read_i64()?),
        tag::FLOAT => Scalar::Float(r.read_f32()?),
        tag::DOUBLE => Scalar::Double(r.read_f64()?),
        tag::DATE => Scalar::Date(r.read_i64()?),
        tag::BYTEARR => {
            let len = r.read_vint()? as usize;
            Scalar::Bytes(r.read_exact(len)?)
        }
        _ => {
            return Err(DecodeError::TypeMismatch {
                expected: "scalar value",
                found: "container/other",
                offset: start,
            });
        }
    })
}

fn read_small_int(r: &mut Reader<'_>, tag_byte: u8) -> Result<i32> {
    let mut v = (tag_byte & 0x0F) as i32;
    if tag_byte & 0x10 != 0 {
        v |= (r.read_vint()? as i32) << 4;
    }
    Ok(v)
}

fn read_small_long(r: &mut Reader<'_>, tag_byte: u8) -> Result<i64> {
    let mut v = (tag_byte & 0x0F) as i64;
    if tag_byte & 0x10 != 0 {
        v |= (r.read_vlong()? as i64) << 4;
    }
    Ok(v)
}

/// Byte-navigation-only skip of one value: advances the cursor past a complete
/// value without touching any string cache. Used purely to test whether a full
/// document is present in the buffer before committing it.
fn skip_value_bytes(r: &mut Reader<'_>) -> Result<()> {
    let mut throwaway = Vec::new();
    skip_value(r, &mut throwaway, 0)
}

/// Byte-navigation-only check that a whole document is buffered; advances the
/// cursor past it. No string cache is touched (that happens only in the real
/// `append_document`). Handles all document encodings: `SOLRDOC`,
/// `ORDERED_MAP`/`NAMED_LST` (fixed size), and `MAP_ENTRY_ITER`
/// (`END`-terminated).
fn skip_document_bytes(r: &mut Reader<'_>) -> Result<()> {
    let t = r.read_u8()?;
    let hi = t >> 5;
    if t == tag::SOLRDOC {
        let inner = r.read_u8()?;
        let n = r.read_size(inner)?;
        skip_entries_fixed(r, n)
    } else if hi == 5 || hi == 6 {
        let n = r.read_size(t)?;
        skip_entries_fixed(r, n)
    } else if t == tag::MAP_ENTRY_ITER {
        loop {
            let peek = *r
                .data
                .get(r.pos)
                .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
            if peek == tag::END {
                r.pos += 1;
                break;
            }
            skip_value_bytes(r)?; // name
            skip_value_bytes(r)?; // value
        }
        Ok(())
    } else {
        Err(DecodeError::TypeMismatch {
            expected: "SolrDocument or map",
            found: "other tag",
            offset: r.pos - 1,
        })
    }
}

/// Skip `n` document entries (name + value, or a child SOLRDOC).
fn skip_entries_fixed(r: &mut Reader<'_>, n: usize) -> Result<()> {
    for _ in 0..n {
        let peek = *r
            .data
            .get(r.pos)
            .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
        if peek == tag::SOLRDOC {
            skip_value_bytes(r)?; // child document: one value
        } else {
            skip_value_bytes(r)?; // field name
            skip_value_bytes(r)?; // field value
        }
    }
    Ok(())
}

/// Skip one javabin value of any type, keeping the extern-string cache in sync
/// (definitions still register). Recursive for containers.
///
/// `depth` is the nesting depth of the value about to be read (the top-level
/// call passes `0`); bounded by [`crate::reader::MAX_NESTING_DEPTH`] so that a
/// deeply nested (adversarial or corrupted) value being skipped can't
/// overflow the call stack.
fn skip_value(r: &mut Reader<'_>, externs: &mut Vec<String>, depth: u32) -> Result<()> {
    let depth = depth + 1;
    if depth > crate::reader::MAX_NESTING_DEPTH {
        return Err(DecodeError::NestingTooDeep {
            offset: r.pos,
            max_depth: crate::reader::MAX_NESTING_DEPTH,
        });
    }
    let t = r.read_u8()?;
    let hi = t >> 5;
    if hi != 0 {
        match hi {
            1 => {
                let _ = r.read_str_tagged(t)?;
            }
            2 => {
                read_small_int(r, t)?;
            }
            3 => {
                read_small_long(r, t)?;
            }
            4 => {
                let n = r.read_size(t)?;
                for _ in 0..n {
                    skip_value(r, externs, depth)?;
                }
            }
            5 | 6 => {
                let n = r.read_size(t)?;
                for _ in 0..n {
                    skip_value(r, externs, depth)?; // key
                    skip_value(r, externs, depth)?; // value
                }
            }
            7 => {
                let idx = r.read_size(t)?;
                if idx == 0 {
                    let inner = r.read_u8()?;
                    let s = r.read_str_tagged(inner)?.to_string();
                    externs.push(s);
                }
            }
            _ => unreachable!(),
        }
        return Ok(());
    }
    match t {
        tag::NULL | tag::BOOL_TRUE | tag::BOOL_FALSE | tag::END => {}
        tag::BYTE => {
            r.read_i8()?;
        }
        tag::SHORT => {
            r.read_i16()?;
        }
        tag::INT | tag::FLOAT => {
            r.read_exact(4)?;
        }
        tag::LONG | tag::DOUBLE | tag::DATE => {
            r.read_exact(8)?;
        }
        tag::BYTEARR => {
            let len = r.read_vint()? as usize;
            r.read_exact(len)?;
        }
        tag::MAP => {
            let n = r.read_vint()? as usize;
            for _ in 0..n {
                skip_value(r, externs, depth)?;
                skip_value(r, externs, depth)?;
            }
        }
        tag::SOLRDOC => {
            let inner = r.read_u8()?;
            let n = r.read_size(inner)?;
            for _ in 0..n {
                skip_value(r, externs, depth)?;
            }
        }
        tag::SOLRDOCLST => {
            skip_value(r, externs, depth)?; // header
            skip_value(r, externs, depth)?; // docs
        }
        tag::ITERATOR | tag::MAP_ENTRY_ITER => loop {
            let b = *r
                .data
                .get(r.pos)
                .ok_or(DecodeError::UnexpectedEof { offset: r.pos })?;
            if b == tag::END {
                r.pos += 1;
                break;
            }
            skip_value(r, externs, depth)?;
            if t == tag::MAP_ENTRY_ITER {
                skip_value(r, externs, depth)?;
            }
        },
        tag::ENUM_FIELD_VALUE => {
            skip_value(r, externs, depth)?;
            skip_value(r, externs, depth)?;
        }
        tag::MAP_ENTRY => {
            skip_value(r, externs, depth)?;
            skip_value(r, externs, depth)?;
        }
        tag::PRIMITIVE_ARR => {
            let sub = r.read_u8()?;
            let len = r.read_vint()? as usize;
            let elem = match sub {
                tag::BYTE | tag::BOOL_TRUE | tag::BOOL_FALSE => 1,
                tag::SHORT => 2,
                tag::INT | tag::FLOAT => 4,
                tag::LONG | tag::DOUBLE => 8,
                _ => {
                    return Err(DecodeError::UnknownTag {
                        tag: sub,
                        offset: r.pos - 1,
                    });
                }
            };
            r.read_exact(len * elem)?;
        }
        other => {
            return Err(DecodeError::UnknownTag {
                tag: other,
                offset: r.pos - 1,
            });
        }
    }
    Ok(())
}

// -- incremental streaming into batches ---------------------------------------

/// Incremental, chunk-fed Arrow decoder. Accumulates network bytes, appends
/// each complete document into the columnar builders, and yields a
/// `RecordBatch` whenever `batch_size` rows have accumulated. Keeps the byte
/// buffer at roughly one document plus the current chunk.
pub struct ArrowStreamState {
    dec: ArrowDecoder,
    /// Pending bytes; `start` is the first unconsumed offset (cursor, so that
    /// per-document consumption is O(1); compacted amortised).
    buf: Vec<u8>,
    start: usize,
    phase: StreamPhase,
    version_checked: bool,
    batch_size: usize,
    /// Completed batches waiting to be handed to Python.
    ready: Vec<RecordBatch>,
}

enum StreamPhase {
    Envelope,
    DocsArr { remaining: usize },
    DocsIter,
    Done,
}

impl ArrowStreamState {
    pub fn new(schema: Arc<Schema>, batch_size: usize) -> Result<Self> {
        Ok(Self {
            dec: ArrowDecoder::new(schema)?,
            buf: Vec::new(),
            start: 0,
            phase: StreamPhase::Envelope,
            version_checked: false,
            batch_size: batch_size.max(1),
            ready: Vec::new(),
        })
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.dec.schema()
    }

    #[inline]
    fn pending(&self) -> &[u8] {
        &self.buf[self.start..]
    }

    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }
        if self.start == self.buf.len() {
            self.buf.clear();
            self.start = 0;
        } else if self.start >= self.buf.len() - self.start {
            self.buf.drain(..self.start);
            self.start = 0;
        }
    }

    /// Feed one chunk; returns any batches completed as a result.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<RecordBatch>> {
        self.buf.extend_from_slice(chunk);
        self.drive()?;
        self.compact();
        Ok(std::mem::take(&mut self.ready))
    }

    /// Signal end of input; returns any final batch. Errors on truncation
    /// (a fixed-length ARR that didn't receive all its documents).
    pub fn finish(&mut self) -> Result<Vec<RecordBatch>> {
        if let StreamPhase::DocsArr { remaining } = self.phase
            && remaining > 0
        {
            return Err(DecodeError::UnexpectedEof {
                offset: self.buf.len(),
            });
        }
        if self.dec.rows() > 0 {
            let batch = self.dec.finish_batch()?;
            self.ready.push(batch);
        }
        Ok(std::mem::take(&mut self.ready))
    }

    fn maybe_flush(&mut self) -> Result<()> {
        if self.dec.rows() >= self.batch_size {
            let batch = self.dec.finish_batch()?;
            self.ready.push(batch);
        }
        Ok(())
    }

    fn drive(&mut self) -> Result<()> {
        loop {
            match self.phase {
                StreamPhase::Done => return Ok(()),
                StreamPhase::Envelope => match self.try_envelope()? {
                    Some(p) => self.phase = p,
                    None => return Ok(()),
                },
                StreamPhase::DocsArr { remaining } => {
                    if remaining == 0 {
                        self.phase = StreamPhase::Done;
                        continue;
                    }
                    if self.try_one_doc()? {
                        self.phase = StreamPhase::DocsArr {
                            remaining: remaining - 1,
                        };
                        self.maybe_flush()?;
                    } else {
                        return Ok(());
                    }
                }
                StreamPhase::DocsIter => {
                    if self.pending().is_empty() {
                        return Ok(());
                    }
                    if self.pending()[0] == tag::END {
                        self.start += 1;
                        self.phase = StreamPhase::Done;
                        continue;
                    }
                    if self.try_one_doc()? {
                        self.maybe_flush()?;
                    } else {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Parse the envelope from the buffer start. Returns the docs phase, or
    /// `None` if more bytes are needed. Re-parses from the start each retry
    /// (nothing consumed until the whole envelope is present).
    fn try_envelope(&mut self) -> Result<Option<StreamPhase>> {
        // Borrow the pending bytes and the decoder disjointly.
        let pending = &self.buf[self.start..];
        let mut r = Reader::new(pending);
        let v = match r.read_u8() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        if !self.version_checked && v != crate::reader::EXPECTED_VERSION {
            return Err(DecodeError::InvalidVersion { found: v });
        }
        // Envelope navigation must register extern strings into the *real*
        // decoder cache, because document field-name references share it.
        // Reset it first so a partial-then-retry can't double-register.
        self.dec.reset_extern_cache();
        let phase = match self.dec.envelope_to_docs(&mut r) {
            Ok(p) => p,
            Err(DecodeError::UnexpectedEof { .. }) => {
                self.dec.reset_extern_cache();
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        let consumed = r.pos;
        self.version_checked = true;
        self.start += consumed;
        Ok(Some(match phase {
            DocsSeq::Arr(n) => StreamPhase::DocsArr { remaining: n },
            DocsSeq::Iter => StreamPhase::DocsIter,
            DocsSeq::None => StreamPhase::Done,
        }))
    }

    /// If a whole document is buffered, append it and advance; else return
    /// false (need more bytes). Completeness is checked by a byte-only skip so
    /// the builders are only ever touched with a fully-present document.
    fn try_one_doc(&mut self) -> Result<bool> {
        let pending = &self.buf[self.start..];
        // 1. is a full document present?
        let mut probe = Reader::new(pending);
        match skip_document_bytes(&mut probe) {
            Ok(()) => {}
            Err(DecodeError::UnexpectedEof { .. }) => return Ok(false),
            Err(e) => return Err(e),
        }
        let doc_len = probe.pos;
        // 2. commit it into the builders.
        let mut r = Reader::new(&pending[..doc_len]);
        self.dec.append_document(&mut r)?;
        self.start += doc_len;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, BooleanArray, Float32Array, Int32Array, Int64Array, ListArray, StringArray,
        TimestampMillisecondArray,
    };
    use arrow::datatypes::{DataType, Field};

    const V: u8 = crate::reader::EXPECTED_VERSION;

    fn write_vint(out: &mut Vec<u8>, mut i: u32) {
        while i & !0x7F != 0 {
            out.push(((i & 0x7f) | 0x80) as u8);
            i >>= 7;
        }
        out.push(i as u8);
    }

    fn str_field(out: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        if b.len() < 0x1f {
            out.push(tag::STR | b.len() as u8);
        } else {
            out.push(tag::STR | 0x1f);
            write_vint(out, (b.len() - 0x1f) as u32);
        }
        out.extend_from_slice(b);
    }

    /// Build a `/select` message: NAMED_LST{"response": SOLRDOCLST{header ARR
    /// [n,0,null,true], docs ARR[docs]}}. `docs` is a slice of pre-encoded
    /// SOLRDOC byte vectors.
    fn select_msg(docs: &[Vec<u8>]) -> Vec<u8> {
        let n = docs.len() as u8;
        let mut b = vec![V, tag::NAMED_LST | 1];
        str_field(&mut b, "response");
        b.push(tag::SOLRDOCLST);
        b.push(tag::ARR | 4);
        b.push(tag::SLONG | n);
        b.push(tag::SLONG);
        b.push(tag::NULL);
        b.push(tag::BOOL_TRUE);
        b.push(tag::ARR | n);
        for d in docs {
            b.extend_from_slice(d);
        }
        b
    }

    /// Encode one SOLRDOC from (field-name, encoded-value) pairs.
    fn doc(fields: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut b = vec![tag::SOLRDOC, tag::ORDERED_MAP | fields.len() as u8];
        for (name, val) in fields {
            str_field(&mut b, name);
            b.extend_from_slice(val);
        }
        b
    }

    fn v_sint(x: i32) -> Vec<u8> {
        // small positive int via SINT compact if it fits, else full INT
        if (0..0x0f).contains(&x) {
            vec![tag::SINT | x as u8]
        } else {
            let mut b = vec![tag::INT];
            b.extend_from_slice(&x.to_be_bytes());
            b
        }
    }
    fn v_long(x: i64) -> Vec<u8> {
        let mut b = vec![tag::LONG];
        b.extend_from_slice(&x.to_be_bytes());
        b
    }
    fn v_float(x: f32) -> Vec<u8> {
        let mut b = vec![tag::FLOAT];
        b.extend_from_slice(&x.to_be_bytes());
        b
    }
    fn v_bool(x: bool) -> Vec<u8> {
        vec![if x { tag::BOOL_TRUE } else { tag::BOOL_FALSE }]
    }
    fn v_date(ms: i64) -> Vec<u8> {
        let mut b = vec![tag::DATE];
        b.extend_from_slice(&ms.to_be_bytes());
        b
    }
    fn v_str(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        str_field(&mut b, s);
        b
    }
    fn v_list_str(items: &[&str]) -> Vec<u8> {
        let mut b = vec![tag::ARR | items.len() as u8];
        for s in items {
            str_field(&mut b, s);
        }
        b
    }

    fn decode_test(schema: Schema, msg: &[u8]) -> RecordBatch {
        let mut d = ArrowDecoder::new(Arc::new(schema)).unwrap();
        d.decode_response(msg).unwrap();
        d.finish_batch().unwrap()
    }

    #[test]
    fn all_scalar_types() {
        let schema = Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("l", DataType::Int64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("b", DataType::Boolean, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("d", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        ]);
        let docs = vec![doc(&[
            ("i", v_sint(7)),
            ("l", v_long(1_870_516_012_295_651_331)),
            ("f", v_float(1.5)),
            ("b", v_bool(true)),
            ("s", v_str("hello")),
            ("d", v_date(1_517_270_400_000)),
        ])];
        let batch = decode_test(schema, &select_msg(&docs));
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            7
        );
        assert_eq!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            1_870_516_012_295_651_331
        );
        assert_eq!(
            batch
                .column(2)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0),
            1.5
        );
        assert!(
            batch
                .column(3)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );
        assert_eq!(
            batch
                .column(4)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "hello"
        );
        assert_eq!(
            batch
                .column(5)
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap()
                .value(0),
            1_517_270_400_000
        );
    }

    #[test]
    fn list_of_strings_column() {
        // Multi-valued (List<Utf8>) column: a present field with two values,
        // a doc where the field is entirely absent (-> null list, not an
        // empty list), and a doc with a single value encoded as a one-element
        // ARR.
        let schema = Schema::new(vec![Field::new(
            "genres",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        )]);
        let docs = vec![
            doc(&[("genres", v_list_str(&["Action", "Sci-Fi"]))]),
            doc(&[]),
            doc(&[("genres", v_list_str(&["Drama"]))]),
        ];
        let batch = decode_test(schema, &select_msg(&docs));
        assert_eq!(batch.num_rows(), 3);

        let list = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        assert!(!list.is_null(0));
        let row0 = list.value(0);
        let row0 = row0.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row0.len(), 2);
        assert_eq!(row0.value(0), "Action");
        assert_eq!(row0.value(1), "Sci-Fi");

        assert!(list.is_null(1));

        assert!(!list.is_null(2));
        let row2 = list.value(2);
        let row2 = row2.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row2.len(), 1);
        assert_eq!(row2.value(0), "Drama");
    }

    #[test]
    fn rejects_excessively_nested_skipped_field() {
        // A field that isn't part of the schema is skipped via `skip_value`
        // rather than materialised into a column. Before the recursion-depth
        // guard, decoding a document where that skipped field's value is
        // deeply nested (attacker-controlled) would overflow the call stack
        // instead of returning a catchable error.
        let mut deep = vec![tag::ARR | 1; 10_000];
        deep.push(tag::NULL);

        let docs = vec![doc(&[("i", v_sint(1)), ("deep", deep)])];
        let schema = Schema::new(vec![Field::new("i", DataType::Int32, true)]);
        let mut dec = ArrowDecoder::new(Arc::new(schema)).unwrap();
        let err = dec.decode_response(&select_msg(&docs)).unwrap_err();
        assert!(err.to_string().contains("nesting"), "{err}");
    }
}
