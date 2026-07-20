//! Decoder for Apache Solr's `javabin` (protocol version 2) format, decoding
//! into an intermediate [`Value`] tree.
//!
//! This is a faithful re-implementation of the read-side of
//! `org.apache.solr.common.util.JavaBinCodec` (see the [Apache Solr JavaBinCodec.java source](https://github.com/apache/solr/blob/main/solr/solrj/src/java/org/apache/solr/common/util/JavaBinCodec.java)).
//!
//! Tag layout
//! ----------
//! A tag byte's upper 3 bits select the decoding branch:
//!
//! - `0b000_?????` (0-31): a fixed single-purpose tag (`NULL`, `BOOL_TRUE`,
//!   `INT`, `MAP`, `SOLRDOC`, ... see [`crate::reader::tag`]).
//! - `0b001_?????` (`STR`): UTF-8 string, size = UTF-8 byte length.
//! - `0b010_?????` (`SINT`): compact positive int.
//! - `0b011_?????` (`SLONG`): compact non-negative long.
//! - `0b100_?????` (`ARR`): ordered array, size = element count.
//! - `0b101_?????` (`ORDERED_MAP`): `SimpleOrderedMap` (decodes like
//!   `NAMED_LST`).
//! - `0b110_?????` (`NAMED_LST`): `NamedList`.
//! - `0b111_?????` (`EXTERN_STRING`): string-table-cached string.
//!
//! For the tags above (except the fixed ones), the lower 5 bits of the tag
//! byte hold a size/index; when all 5 bits are set (`0x1f`) the actual value
//! is `0x1f + read_vint()`.
//!
//! This module is used for [`crate::deserialize_json`]; for the
//! `deserialize` entry point (returning native Python objects), see
//! [`crate::py_decoder`], which decodes directly into `PyObject`s in a
//! single pass instead of building this intermediate tree.

use crate::reader::{DecodeError, Reader, Result, tag};
use crate::value::Value;

/// Result of reading one "slot" in a context that may be terminated by the
/// javabin `END` sentinel (used by `ITERATOR` and `MAP_ENTRY_ITER`).
enum Slot {
    End,
    Value(Value),
}

/// Stateful javabin reader over an in-memory byte slice.
struct Decoder<'a> {
    reader: Reader<'a>,
    /// String table used by `EXTERN_STRING`, populated in encounter order for
    /// the lifetime of a single top-level decode call (mirrors
    /// `JavaBinCodec.stringsList`, which is per-codec-instance and a codec
    /// instance is only ever used for a single `unmarshal` call).
    strings: Vec<String>,
    /// Current container-nesting depth; see [`crate::reader::MAX_NESTING_DEPTH`].
    depth: u32,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            reader: Reader::new(data),
            strings: Vec::new(),
            depth: 0,
        }
    }

    // -- string handling ----------------------------------------------------

    /// Read an `EXTERN_STRING`-tagged value (either a string-table reference
    /// or a fresh string that gets registered in the table).
    fn read_extern_string(&mut self, tag_byte: u8) -> Result<String> {
        let idx = self.reader.read_size(tag_byte)?;
        if idx != 0 {
            self.strings
                .get(idx - 1)
                .cloned()
                .ok_or(DecodeError::UnexpectedEof {
                    offset: self.reader.pos,
                })
        } else {
            let inner_tag = self.reader.read_u8()?;
            let s = self.reader.read_str_tagged(inner_tag)?.to_string();
            self.strings.push(s.clone());
            Ok(s)
        }
    }

    // -- top-level dispatch ---------------------------------------------------

    /// Read one value, expecting it *not* to be the `END` sentinel.
    fn read_value(&mut self) -> Result<Value> {
        match self.read_slot()? {
            Slot::Value(v) => Ok(v),
            Slot::End => Err(DecodeError::TypeMismatch {
                expected: "value",
                found: "END marker",
                offset: self.reader.pos - 1,
            }),
        }
    }

    /// Read one value or the `END` sentinel (used by `ITERATOR` and
    /// `MAP_ENTRY_ITER`, which don't have a known element count).
    ///
    /// Wraps [`Self::read_slot_inner`] with a nesting-depth guard: every
    /// recursive descent into a nested value goes through this function, so
    /// counting here bounds the recursion depth of the whole decoder.
    fn read_slot(&mut self) -> Result<Slot> {
        self.depth += 1;
        if self.depth > crate::reader::MAX_NESTING_DEPTH {
            self.depth -= 1;
            return Err(DecodeError::NestingTooDeep {
                offset: self.reader.pos,
                max_depth: crate::reader::MAX_NESTING_DEPTH,
            });
        }
        let result = self.read_slot_inner();
        self.depth -= 1;
        result
    }

    fn read_slot_inner(&mut self) -> Result<Slot> {
        let start = self.reader.pos;
        let t = self.reader.read_u8()?;
        let hi = t >> 5;

        if hi != 0 {
            let v = match hi {
                1 => Value::Str(self.reader.read_str_tagged(t)?.to_string()),
                2 => Value::Int(self.read_small_int(t)?),
                3 => Value::Long(self.read_small_long(t)?),
                4 => Value::List(self.read_array(t)?),
                5 | 6 => Value::NamedList(self.read_named_list(t)?),
                7 => Value::Str(self.read_extern_string(t)?),
                _ => unreachable!("3-bit value out of range"),
            };
            return Ok(Slot::Value(v));
        }

        let v = match t {
            tag::NULL => Value::Null,
            tag::BOOL_TRUE => Value::Bool(true),
            tag::BOOL_FALSE => Value::Bool(false),
            tag::BYTE => Value::Byte(self.reader.read_i8()?),
            tag::SHORT => Value::Short(self.reader.read_i16()?),
            tag::DOUBLE => Value::Double(self.reader.read_f64()?),
            tag::INT => Value::Int(self.reader.read_i32()?),
            tag::LONG => Value::Long(self.reader.read_i64()?),
            tag::FLOAT => Value::Float(self.reader.read_f32()?),
            tag::DATE => Value::Date(self.reader.read_i64()?),
            tag::MAP => Value::Map(self.read_map()?),
            tag::SOLRDOC => self.read_solr_document()?,
            tag::SOLRDOCLST => self.read_solr_document_list()?,
            tag::BYTEARR => {
                let len = self.reader.read_vint()? as usize;
                Value::Bytes(self.reader.read_exact(len)?.to_vec())
            }
            tag::ITERATOR => Value::List(self.read_iterator()?),
            tag::END => return Ok(Slot::End),
            tag::SOLRINPUTDOC => self.read_solr_input_document()?,
            tag::MAP_ENTRY_ITER => Value::Map(self.read_map_entry_iter()?),
            tag::ENUM_FIELD_VALUE => self.read_enum_field_value()?,
            tag::MAP_ENTRY => {
                let key = self.read_value()?;
                let val = self.read_value()?;
                Value::Map(vec![(key, val)])
            }
            tag::PRIMITIVE_ARR => self.read_primitive_array()?,
            tag::UUID => {
                return Err(DecodeError::UnknownTag {
                    tag: t,
                    offset: start,
                });
            }
            other => {
                return Err(DecodeError::UnknownTag {
                    tag: other,
                    offset: start,
                });
            }
        };

        Ok(Slot::Value(v))
    }

    // -- compact numeric encodings -------------------------------------------

    fn read_small_int(&mut self, tag_byte: u8) -> Result<i32> {
        let mut v = (tag_byte & 0x0F) as i32;
        if tag_byte & 0x10 != 0 {
            v |= (self.reader.read_vint()? as i32) << 4;
        }
        Ok(v)
    }

    fn read_small_long(&mut self, tag_byte: u8) -> Result<i64> {
        let mut v = (tag_byte & 0x0F) as i64;
        if tag_byte & 0x10 != 0 {
            v |= (self.reader.read_vlong()? as i64) << 4;
        }
        Ok(v)
    }

    // -- containers -----------------------------------------------------------

    fn read_array(&mut self, tag_byte: u8) -> Result<Vec<Value>> {
        let sz = self.reader.read_size(tag_byte)?;
        let mut items = Vec::with_capacity(self.reader.capacity_hint(sz));
        for _ in 0..sz {
            items.push(self.read_value()?);
        }
        Ok(items)
    }

    fn read_named_list(&mut self, tag_byte: u8) -> Result<Vec<(String, Value)>> {
        let sz = self.reader.read_size(tag_byte)?;
        let mut entries = Vec::with_capacity(self.reader.capacity_hint(sz));
        for _ in 0..sz {
            let name = self.expect_str()?;
            let val = self.read_value()?;
            entries.push((name, val));
        }
        Ok(entries)
    }

    fn read_map(&mut self) -> Result<Vec<(Value, Value)>> {
        let sz = self.reader.read_vint()? as usize;
        let mut entries = Vec::with_capacity(self.reader.capacity_hint(sz));
        for _ in 0..sz {
            let key = self.read_value()?;
            let val = self.read_value()?;
            entries.push((key, val));
        }
        Ok(entries)
    }

    fn read_map_entry_iter(&mut self) -> Result<Vec<(Value, Value)>> {
        let mut entries = Vec::new();
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(key) => {
                    let val = self.read_value()?;
                    entries.push((key, val));
                }
            }
        }
        Ok(entries)
    }

    fn read_iterator(&mut self) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(v) => items.push(v),
            }
        }
        Ok(items)
    }

    fn read_enum_field_value(&mut self) -> Result<Value> {
        let int_val = self.read_value()?;
        let str_val = self.read_value()?;
        Ok(Value::NamedList(vec![
            ("int".to_string(), int_val),
            ("string".to_string(), str_val),
        ]))
    }

    fn read_primitive_array(&mut self) -> Result<Value> {
        let sub_tag = self.reader.read_u8()?;
        let len = self.reader.read_vint()? as usize;

        let items = match sub_tag {
            tag::FLOAT => (0..len)
                .map(|_| self.reader.read_f32().map(Value::Float))
                .collect::<Result<Vec<_>>>()?,
            tag::INT => (0..len)
                .map(|_| self.reader.read_i32().map(Value::Int))
                .collect::<Result<Vec<_>>>()?,
            tag::LONG => (0..len)
                .map(|_| self.reader.read_i64().map(Value::Long))
                .collect::<Result<Vec<_>>>()?,
            tag::DOUBLE => (0..len)
                .map(|_| self.reader.read_f64().map(Value::Double))
                .collect::<Result<Vec<_>>>()?,
            tag::SHORT => (0..len)
                .map(|_| self.reader.read_i16().map(Value::Short))
                .collect::<Result<Vec<_>>>()?,
            tag::BOOL_TRUE | tag::BOOL_FALSE => (0..len)
                .map(|_| {
                    self.reader
                        .read_u8()
                        .map(|b| Value::Bool(b != tag::BOOL_FALSE))
                })
                .collect::<Result<Vec<_>>>()?,
            tag::BYTE => {
                let bytes = self.reader.read_exact(len)?.to_vec();
                return Ok(Value::Bytes(bytes));
            }
            other => {
                return Err(DecodeError::UnknownTag {
                    tag: other,
                    offset: self.reader.pos - 1,
                });
            }
        };

        Ok(Value::List(items))
    }

    // -- Solr-specific containers ----------------------------------------------

    fn read_solr_document(&mut self) -> Result<Value> {
        let inner_tag = self.reader.read_u8()?;
        let sz = self.reader.read_size(inner_tag)?;

        let mut fields = Vec::new();
        let mut children = Vec::new();

        for _ in 0..sz {
            let obj = self.read_value()?;
            match obj {
                Value::SolrDocument { .. } => children.push(obj),
                Value::Str(field_name) => {
                    let field_val = self.read_value()?;
                    fields.push((field_name, field_val));
                }
                other => {
                    return Err(DecodeError::TypeMismatch {
                        expected: "field name (string) or child SolrDocument",
                        found: value_kind(&other),
                        offset: self.reader.pos,
                    });
                }
            }
        }

        Ok(Value::SolrDocument { fields, children })
    }

    fn read_solr_input_document(&mut self) -> Result<Value> {
        // SOLRINPUTDOC: VInt size, then a document boost (float, ignored),
        // then `size` entries of either [boost(float), name, value],
        // [name, value] or a nested child document.
        let sz = self.reader.read_vint()? as usize;
        let _doc_boost = self.read_value()?; // always a Float, historically the doc boost

        let mut fields = Vec::new();
        let mut children = Vec::new();

        for _ in 0..sz {
            let mut obj = self.read_value()?;

            // A leading per-field boost (Float) may precede the field name;
            // skip it and read the actual name next.
            if matches!(obj, Value::Float(_)) {
                obj = self.read_value()?;
            }

            match obj {
                Value::SolrDocument { .. } => children.push(obj),
                Value::Str(field_name) => {
                    let field_val = self.read_value()?;
                    fields.push((field_name, field_val));
                }
                other => {
                    return Err(DecodeError::TypeMismatch {
                        expected: "field name (string) or child SolrDocument",
                        found: value_kind(&other),
                        offset: self.reader.pos,
                    });
                }
            }
        }

        Ok(Value::SolrDocument { fields, children })
    }

    fn read_solr_document_list(&mut self) -> Result<Value> {
        let header = self.read_value()?;
        let Value::List(header_items) = header else {
            return Err(DecodeError::TypeMismatch {
                expected: "SolrDocumentList header array",
                found: value_kind(&header),
                offset: self.reader.pos,
            });
        };

        let num_found = header_items.first().and_then(value_as_i64).unwrap_or(0);
        let start = header_items.get(1).and_then(value_as_i64).unwrap_or(0);
        let max_score = header_items.get(2).and_then(value_as_f64);
        let num_found_exact = header_items.get(3).and_then(|v| match v {
            Value::Bool(b) => Some(*b),
            _ => None,
        });

        let docs_value = self.read_value()?;
        let Value::List(docs) = docs_value else {
            return Err(DecodeError::TypeMismatch {
                expected: "SolrDocumentList docs array",
                found: value_kind(&docs_value),
                offset: self.reader.pos,
            });
        };

        Ok(Value::SolrDocumentList {
            num_found,
            start,
            max_score,
            num_found_exact,
            docs,
        })
    }

    fn expect_str(&mut self) -> Result<String> {
        let offset = self.reader.pos;
        match self.read_value()? {
            Value::Str(s) => Ok(s),
            other => Err(DecodeError::TypeMismatch {
                expected: "string",
                found: value_kind(&other),
                offset,
            }),
        }
    }
}

fn value_as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Long(n) => Some(*n),
        Value::Int(n) => Some(*n as i64),
        _ => None,
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(n) => Some(*n as f64),
        Value::Double(n) => Some(*n),
        _ => None,
    }
}

fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Byte(_) => "byte",
        Value::Short(_) => "short",
        Value::Int(_) => "int",
        Value::Long(_) => "long",
        Value::Float(_) => "float",
        Value::Double(_) => "double",
        Value::Date(_) => "date",
        Value::Str(_) => "string",
        Value::Bytes(_) => "bytes",
        Value::List(_) => "array",
        Value::Map(_) => "map",
        Value::NamedList(_) => "named list",
        Value::SolrDocument { .. } => "solr document",
        Value::SolrDocumentList { .. } => "solr document list",
    }
}

/// Decode a complete javabin (protocol version 2) message into a [`Value`]
/// tree.
///
/// # Errors
///
/// Returns [`DecodeError`] if the version byte doesn't match, the data is
/// truncated, contains invalid UTF-8 in a string, or an unknown/unsupported
/// tag byte is encountered.
pub fn decode(data: &[u8]) -> Result<Value> {
    let mut decoder = Decoder::new(data);

    let version = decoder.reader.read_u8()?;
    if version != tag_expected_version() {
        return Err(DecodeError::InvalidVersion { found: version });
    }

    let value = decoder.read_value()?;

    if decoder.reader.pos != decoder.reader.data.len() {
        return Err(DecodeError::TrailingData {
            remaining: decoder.reader.data.len() - decoder.reader.pos,
        });
    }

    Ok(value)
}

fn tag_expected_version() -> u8 {
    crate::reader::EXPECTED_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_VERSION: u8 = crate::reader::EXPECTED_VERSION;

    fn with_version(mut body: Vec<u8>) -> Vec<u8> {
        let mut out = vec![EXPECTED_VERSION];
        out.append(&mut body);
        out
    }

    #[test]
    fn decodes_null() {
        let data = with_version(vec![tag::NULL]);
        assert_eq!(decode(&data).unwrap(), Value::Null);
    }

    #[test]
    fn decodes_bools() {
        assert_eq!(
            decode(&with_version(vec![tag::BOOL_TRUE])).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            decode(&with_version(vec![tag::BOOL_FALSE])).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn decodes_full_int_and_long() {
        let mut body = vec![tag::INT];
        body.extend_from_slice(&(-42i32).to_be_bytes());
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Int(-42));

        let mut body = vec![tag::LONG];
        body.extend_from_slice(&(-1234567890123i64).to_be_bytes());
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Long(-1234567890123)
        );
    }

    #[test]
    fn decodes_compact_sint_small() {
        // val = 5 (< 0x0f), fits directly in the low nibble, no VInt follows.
        let body = vec![tag::SINT | 0x05];
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Int(5));
    }

    #[test]
    fn decodes_compact_sint_large() {
        // val = 1000 = 0b11_1110_1000
        // low nibble = val & 0x0f = 0b1000 = 8, remaining = val >> 4 = 62
        let val: i32 = 1000;
        let low = (val & 0x0f) as u8;
        let rest = (val >> 4) as u32;
        let mut body = vec![tag::SINT | 0x10 | low];
        write_vint(&mut body, rest);
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Int(val));
    }

    #[test]
    fn decodes_compact_slong() {
        let val: i64 = 70_000;
        let low = (val & 0x0f) as u8;
        let rest = (val >> 4) as u64;
        let mut body = vec![tag::SLONG | 0x10 | low];
        write_vlong(&mut body, rest);
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Long(val));
    }

    #[test]
    fn decodes_float_double_date() {
        let mut body = vec![tag::FLOAT];
        body.extend_from_slice(&1.5f32.to_be_bytes());
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Float(1.5));

        let mut body = vec![tag::DOUBLE];
        body.extend_from_slice(&2.25f64.to_be_bytes());
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Double(2.25));

        let mut body = vec![tag::DATE];
        body.extend_from_slice(&1_700_000_000_000i64.to_be_bytes());
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Date(1_700_000_000_000)
        );
    }

    #[test]
    fn decodes_short_string() {
        let s = "hello";
        let mut body = vec![tag::STR | s.len() as u8];
        body.extend_from_slice(s.as_bytes());
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Str("hello".to_string())
        );
    }

    #[test]
    fn decodes_long_string_with_extended_size() {
        let s = "x".repeat(100);
        let mut body = vec![tag::STR | 0x1f];
        write_vint(&mut body, (s.len() - 0x1f) as u32);
        body.extend_from_slice(s.as_bytes());
        assert_eq!(decode(&with_version(body)).unwrap(), Value::Str(s));
    }

    #[test]
    fn decodes_byte_array() {
        let mut body = vec![tag::BYTEARR];
        write_vint(&mut body, 3);
        body.extend_from_slice(&[1, 2, 3]);
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Bytes(vec![1, 2, 3])
        );
    }

    #[test]
    fn decodes_array() {
        // [1, 2] as compact SINT values.
        let body = vec![tag::ARR | 2, tag::SINT | 1, tag::SINT | 2];
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::List(vec![Value::Int(1), Value::Int(2)])
        );
    }

    #[test]
    fn decodes_named_list() {
        // NamedList{"a": 1}
        let mut body = vec![tag::NAMED_LST | 1];
        body.push(tag::STR | 1);
        body.push(b'a');
        body.push(tag::SINT | 1);
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::NamedList(vec![("a".to_string(), Value::Int(1))])
        );
    }

    #[test]
    fn decodes_generic_map_with_extern_string_key() {
        // MAP{"a": 1} where "a" is written as an EXTERN_STRING (idx=0, first use)
        let mut body = vec![tag::MAP];
        write_vint(&mut body, 1);
        body.push(tag::EXTERN_STRING); // idx 0
        body.push(tag::STR | 1);
        body.push(b'a');
        body.push(tag::SINT | 1);
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Map(vec![(Value::Str("a".to_string()), Value::Int(1))])
        );
    }

    #[test]
    fn decodes_repeated_extern_string_reference() {
        // ARR of 2 EXTERN_STRING "foo": first occurrence defines it (idx=0),
        // second occurrence references it by idx=1.
        let mut body = vec![tag::ARR | 2];
        body.push(tag::EXTERN_STRING);
        body.push(tag::STR | 3);
        body.extend_from_slice(b"foo");
        body.push(tag::EXTERN_STRING | 1);
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::List(vec![
                Value::Str("foo".to_string()),
                Value::Str("foo".to_string())
            ])
        );
    }

    #[test]
    fn decodes_iterator_terminated_by_end() {
        let body = vec![tag::ITERATOR, tag::SINT | 1, tag::SINT | 2, tag::END];
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::List(vec![Value::Int(1), Value::Int(2)])
        );
    }

    #[test]
    fn decodes_map_entry_iter() {
        let mut body = vec![tag::MAP_ENTRY_ITER];
        body.push(tag::EXTERN_STRING);
        body.push(tag::STR | 1);
        body.push(b'a');
        body.push(tag::SINT | 1);
        body.push(tag::END);
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::Map(vec![(Value::Str("a".to_string()), Value::Int(1))])
        );
    }

    #[test]
    fn decodes_solr_document() {
        // SOLRDOC { ORDERED_MAP size=1: "id" -> "42" }
        let mut body = vec![tag::SOLRDOC, tag::ORDERED_MAP | 1];
        body.push(tag::STR | 2);
        body.extend_from_slice(b"id");
        body.push(tag::STR | 2);
        body.extend_from_slice(b"42");

        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::SolrDocument {
                fields: vec![("id".to_string(), Value::Str("42".to_string()))],
                children: vec![],
            }
        );
    }

    #[test]
    fn decodes_solr_document_with_child() {
        // SOLRDOC { ORDERED_MAP size=2: "id" -> "1", <child SOLRDOC "id" -> "2"> }
        let mut body = vec![tag::SOLRDOC, tag::ORDERED_MAP | 2];
        body.push(tag::STR | 2);
        body.extend_from_slice(b"id");
        body.push(tag::STR | 1);
        body.push(b'1');
        // child doc
        body.push(tag::SOLRDOC);
        body.push(tag::ORDERED_MAP | 1);
        body.push(tag::STR | 2);
        body.extend_from_slice(b"id");
        body.push(tag::STR | 1);
        body.push(b'2');

        let decoded = decode(&with_version(body)).unwrap();
        match decoded {
            Value::SolrDocument { fields, children } => {
                assert_eq!(
                    fields,
                    vec![("id".to_string(), Value::Str("1".to_string()))]
                );
                assert_eq!(children.len(), 1);
                assert_eq!(
                    children[0],
                    Value::SolrDocument {
                        fields: vec![("id".to_string(), Value::Str("2".to_string()))],
                        children: vec![],
                    }
                );
            }
            other => panic!("expected SolrDocument, got {other:?}"),
        }
    }

    #[test]
    fn decodes_solr_document_list() {
        // SOLRDOCLST { header ARR[numFound=1(long), start=0(long), maxScore=NULL,
        //   numFoundExact=BOOL_TRUE], docs ARR[ 1 x SOLRDOC ] }
        let mut body = vec![tag::SOLRDOCLST];

        body.push(tag::ARR | 4);
        body.push(tag::SLONG | 1); // numFound = 1
        body.push(tag::SLONG | 0); // start = 0
        body.push(tag::NULL); // maxScore = null
        body.push(tag::BOOL_TRUE); // numFoundExact = true

        body.push(tag::ARR | 1);
        body.push(tag::SOLRDOC);
        body.push(tag::ORDERED_MAP | 1);
        body.push(tag::STR | 2);
        body.extend_from_slice(b"id");
        body.push(tag::STR | 1);
        body.push(b'1');

        let decoded = decode(&with_version(body)).unwrap();
        match decoded {
            Value::SolrDocumentList {
                num_found,
                start,
                max_score,
                num_found_exact,
                docs,
            } => {
                assert_eq!(num_found, 1);
                assert_eq!(start, 0);
                assert_eq!(max_score, None);
                assert_eq!(num_found_exact, Some(true));
                assert_eq!(docs.len(), 1);
            }
            other => panic!("expected SolrDocumentList, got {other:?}"),
        }
    }

    #[test]
    fn decodes_primitive_int_array() {
        let mut body = vec![tag::PRIMITIVE_ARR, tag::INT];
        write_vint(&mut body, 2);
        body.extend_from_slice(&1i32.to_be_bytes());
        body.extend_from_slice(&2i32.to_be_bytes());
        assert_eq!(
            decode(&with_version(body)).unwrap(),
            Value::List(vec![Value::Int(1), Value::Int(2)])
        );
    }

    #[test]
    fn rejects_wrong_version() {
        let data = vec![1u8, tag::NULL];
        assert!(matches!(
            decode(&data),
            Err(DecodeError::InvalidVersion { found: 1 })
        ));
    }

    #[test]
    fn rejects_truncated_input() {
        let data = vec![EXPECTED_VERSION, tag::INT, 0, 0];
        assert!(matches!(
            decode(&data),
            Err(DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn rejects_oversized_map_size_claim_without_huge_allocation() {
        // A MAP claiming ~4 billion entries but with no further bytes at
        // all. Before capping the `Vec::with_capacity` hint to the number of
        // remaining input bytes, this would ask the allocator to reserve
        // space for ~4 billion `(Value, Value)` tuples up front -- on a Rust
        // allocation failure that calls `handle_alloc_error`, which aborts
        // the process unconditionally (uncatchable, unlike a `Result`). This
        // must instead fail fast with an ordinary decode error.
        let mut body = vec![tag::MAP];
        write_vint(&mut body, u32::MAX - 10);
        assert!(matches!(
            decode(&with_version(body)),
            Err(DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn rejects_oversized_array_size_claim_without_huge_allocation() {
        let mut body = vec![tag::ARR | 0x1f];
        write_vint(&mut body, u32::MAX - 10 - 0x1f);
        assert!(matches!(
            decode(&with_version(body)),
            Err(DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn rejects_oversized_named_list_size_claim_without_huge_allocation() {
        let mut body = vec![tag::NAMED_LST | 0x1f];
        write_vint(&mut body, u32::MAX - 10 - 0x1f);
        assert!(matches!(
            decode(&with_version(body)),
            Err(DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn rejects_trailing_data() {
        let data = vec![EXPECTED_VERSION, tag::NULL, 0xFF];
        assert!(matches!(
            decode(&data),
            Err(DecodeError::TrailingData { remaining: 1 })
        ));
    }

    #[test]
    fn rejects_excessively_nested_input() {
        // A chain of single-element ARR tags nested far deeper than any real
        // document, terminated by a NULL leaf. Before the recursion-depth
        // guard this would overflow the call stack (SIGSEGV) instead of
        // returning a catchable error.
        let mut body = vec![tag::ARR | 1; 10_000];
        body.push(tag::NULL);
        assert!(matches!(
            decode(&with_version(body)),
            Err(DecodeError::NestingTooDeep { .. })
        ));
    }

    #[test]
    fn decodes_nesting_within_the_depth_limit() {
        // Well within the limit: must still decode correctly, proving the
        // depth guard doesn't reject legitimate (if unusually deep) input.
        let depth = 100;
        let mut body = vec![tag::ARR | 1; depth];
        body.push(tag::NULL);
        let value = decode(&with_version(body)).unwrap();

        let mut v = &value;
        for _ in 0..depth {
            match v {
                Value::List(items) => v = &items[0],
                other => panic!("expected nested list, got {other:?}"),
            }
        }
        assert_eq!(*v, Value::Null);
    }

    // -- test helpers ---------------------------------------------------------

    fn write_vint(out: &mut Vec<u8>, mut i: u32) {
        while i & !0x7F != 0 {
            out.push(((i & 0x7f) | 0x80) as u8);
            i >>= 7;
        }
        out.push(i as u8);
    }

    fn write_vlong(out: &mut Vec<u8>, mut i: u64) {
        while i & !0x7F != 0 {
            out.push(((i & 0x7f) | 0x80) as u8);
            i >>= 7;
        }
        out.push(i as u8);
    }
}
