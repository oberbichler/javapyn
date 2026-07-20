//! Direct-to-`PyObject` decoder for javabin, used by [`crate::deserialize`].
//!
//! Unlike [`crate::decoder`] (which builds an intermediate [`crate::value::Value`]
//! tree, used for `deserialize_json`), this module decodes straight into
//! native Python objects in a single pass. This avoids materializing a
//! parallel Rust data structure just to immediately convert it, roughly
//! halving allocations for container-heavy payloads (which is the common
//! case for Solr responses: many nested `NamedList`/`SolrDocument` objects).
//!
//! It also **interns** every `EXTERN_STRING`-tagged string (via
//! `PyString::intern`) instead of allocating a fresh `PyString` per
//! occurrence. This is a deliberate, format-aware optimization: in practice
//! Solr's `JavaBinCodec` only ever uses `EXTERN_STRING` for `NamedList`/`Map`/
//! `SolrDocument` *keys* (i.e. field names), never for arbitrary field
//! *values* (see `writeExternString` call sites in `JavaBinCodec.java`).
//! Field names repeat identically across every document in a
//! `SolrDocumentList`, so interning means the 2nd+ occurrence of a given
//! field name is just a cheap `Py<PyString>` clone (refcount bump) instead of
//! a fresh UTF-8 decode + allocation + Python object construction. Ordinary
//! `STR`-tagged values (actual field values, which are highly variable) are
//! deliberately *not* interned, to avoid bloating CPython's global intern
//! table with one-off strings.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use crate::reader::{DecodeError, Reader, tag};

/// All fallible operations in this module produce a [`PyErr`] directly
/// (rather than [`DecodeError`]) so that `?` can freely mix javabin decode
/// errors with PyO3 API errors (e.g. `PyDict::set_item`).
type Result<T> = PyResult<T>;

impl From<DecodeError> for PyErr {
    fn from(err: DecodeError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Result of reading one "slot": either a decoded value, the `END` sentinel,
/// or (only directly beneath a `SOLRDOC`'s field list) a child `SolrDocument`,
/// which must be told apart from an ordinary dict-shaped value (`Map`/
/// `NamedList`) so [`Decoder::read_solr_document`] can route it to
/// `_childDocuments_` instead of treating it as a `(name, value)` pair.
enum Slot<'py> {
    End,
    SolrDoc(Bound<'py, PyDict>),
    Value(Bound<'py, PyAny>),
}

/// Result of [`Decoder::read_field_or_child`]: one entry of a
/// `SolrDocument`'s field list is either a field name or a nested child
/// document.
enum FieldOrChild<'py> {
    Name(Bound<'py, PyString>),
    Child(Bound<'py, PyDict>),
}

struct Decoder<'a, 'py> {
    reader: Reader<'a>,
    py: Python<'py>,
    /// Interned field-name strings, indexed like `JavaBinCodec.stringsList`
    /// (see module docs for why interning is safe/beneficial here).
    strings: Vec<Bound<'py, PyString>>,
    /// Current container-nesting depth; see [`crate::reader::MAX_NESTING_DEPTH`].
    depth: u32,
}

impl<'a, 'py> Decoder<'a, 'py> {
    fn new(py: Python<'py>, data: &'a [u8]) -> Self {
        Self {
            reader: Reader::new(data),
            py,
            strings: Vec::new(),
            depth: 0,
        }
    }

    // -- string handling ------------------------------------------------------

    fn read_extern_string(&mut self, tag_byte: u8) -> Result<Bound<'py, PyString>> {
        let idx = self.reader.read_size(tag_byte)?;
        if idx != 0 {
            self.strings.get(idx - 1).cloned().ok_or(
                DecodeError::UnexpectedEof {
                    offset: self.reader.pos,
                }
                .into(),
            )
        } else {
            let inner_tag = self.reader.read_u8()?;
            let s = self.reader.read_str_tagged(inner_tag)?;
            let interned = PyString::intern(self.py, s);
            self.strings.push(interned.clone());
            Ok(interned)
        }
    }

    // -- top-level dispatch ---------------------------------------------------

    /// Read one value, expecting it *not* to be the `END` sentinel. A child
    /// `SolrDocument` (if encountered outside of `read_solr_document`'s
    /// field loop) is simply treated as its dict.
    fn read_value(&mut self) -> Result<Bound<'py, PyAny>> {
        match self.read_slot()? {
            Slot::Value(v) => Ok(v),
            Slot::SolrDoc(d) => Ok(d.into_any()),
            Slot::End => Err(DecodeError::TypeMismatch {
                expected: "value",
                found: "END marker",
                offset: self.reader.pos - 1,
            }
            .into()),
        }
    }

    /// Read one value or the `END` sentinel (used by `ITERATOR` and
    /// `MAP_ENTRY_ITER`, which don't have a known element count).
    fn read_value_or_end(&mut self) -> Result<Option<Bound<'py, PyAny>>> {
        match self.read_slot()? {
            Slot::End => Ok(None),
            Slot::Value(v) => Ok(Some(v)),
            Slot::SolrDoc(d) => Ok(Some(d.into_any())),
        }
    }

    /// Wraps [`Self::read_slot_inner`] with a nesting-depth guard; see
    /// `decoder::Decoder::read_slot` for the rationale (identical here).
    fn read_slot(&mut self) -> Result<Slot<'py>> {
        self.depth += 1;
        if self.depth > crate::reader::MAX_NESTING_DEPTH {
            self.depth -= 1;
            return Err(DecodeError::NestingTooDeep {
                offset: self.reader.pos,
                max_depth: crate::reader::MAX_NESTING_DEPTH,
            }
            .into());
        }
        let result = self.read_slot_inner();
        self.depth -= 1;
        result
    }

    fn read_slot_inner(&mut self) -> Result<Slot<'py>> {
        let start = self.reader.pos;
        let t = self.reader.read_u8()?;
        let hi = t >> 5;

        if hi != 0 {
            let v = match hi {
                1 => PyString::new(self.py, self.reader.read_str_tagged(t)?).into_any(),
                2 => self.read_small_int(t)?.into_bound_py_any(self.py)?,
                3 => self.read_small_long(t)?.into_bound_py_any(self.py)?,
                4 => self.read_array(t)?.into_any(),
                5 | 6 => self.read_named_list(t)?.into_any(),
                7 => self.read_extern_string(t)?.into_any(),
                _ => unreachable!("3-bit value out of range"),
            };
            return Ok(Slot::Value(v));
        }

        let v = match t {
            tag::NULL => self.py.None().into_bound(self.py),
            tag::BOOL_TRUE => true.into_bound_py_any(self.py)?,
            tag::BOOL_FALSE => false.into_bound_py_any(self.py)?,
            tag::BYTE => self.reader.read_i8()?.into_bound_py_any(self.py)?,
            tag::SHORT => self.reader.read_i16()?.into_bound_py_any(self.py)?,
            tag::DOUBLE => self.reader.read_f64()?.into_bound_py_any(self.py)?,
            tag::INT => self.reader.read_i32()?.into_bound_py_any(self.py)?,
            tag::LONG => self.reader.read_i64()?.into_bound_py_any(self.py)?,
            tag::FLOAT => (self.reader.read_f32()? as f64).into_bound_py_any(self.py)?,
            tag::DATE => self.reader.read_i64()?.into_bound_py_any(self.py)?,
            tag::MAP => self.read_map()?.into_any(),
            tag::SOLRDOC => return Ok(Slot::SolrDoc(self.read_solr_document()?)),
            tag::SOLRDOCLST => self.read_solr_document_list()?.into_any(),
            tag::BYTEARR => {
                let len = self.reader.read_vint()? as usize;
                PyBytes::new(self.py, self.reader.read_exact(len)?).into_any()
            }
            tag::ITERATOR => self.read_iterator()?.into_any(),
            tag::END => return Ok(Slot::End),
            tag::SOLRINPUTDOC => self.read_solr_input_document()?.into_any(),
            tag::MAP_ENTRY_ITER => self.read_map_entry_iter()?.into_any(),
            tag::ENUM_FIELD_VALUE => self.read_enum_field_value()?.into_any(),
            tag::MAP_ENTRY => {
                let key = self.read_value()?;
                let val = self.read_value()?;
                let dict = PyDict::new(self.py);
                dict.set_item(key, val)?;
                dict.into_any()
            }
            tag::PRIMITIVE_ARR => self.read_primitive_array()?,
            other => {
                return Err(DecodeError::UnknownTag {
                    tag: other,
                    offset: start,
                }
                .into());
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

    fn read_array(&mut self, tag_byte: u8) -> Result<Bound<'py, PyList>> {
        let sz = self.reader.read_size(tag_byte)?;
        let list = PyList::empty(self.py);
        for _ in 0..sz {
            list.append(self.read_value()?)?;
        }
        Ok(list)
    }

    fn read_named_list(&mut self, tag_byte: u8) -> Result<Bound<'py, PyDict>> {
        let sz = self.reader.read_size(tag_byte)?;
        let dict = PyDict::new(self.py);
        for _ in 0..sz {
            let name = self.expect_str()?;
            let val = self.read_value()?;
            dict.set_item(name, val)?;
        }
        Ok(dict)
    }

    fn read_map(&mut self) -> Result<Bound<'py, PyAny>> {
        let sz = self.reader.read_vint()? as usize;

        // Peek whether every key will be a string: if so, a dict is the
        // natural (and JSON-compatible) representation, matching the JSON
        // reference shape. Since we can't peek without consuming, decode
        // eagerly into pairs and fall back to a list-of-pairs if a
        // non-string key shows up (rare: only reachable via a non-Map<String,?>
        // Java Map, which practically doesn't occur in Solr responses).
        let mut pairs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> =
            Vec::with_capacity(self.reader.capacity_hint(sz));
        let mut all_string_keys = true;
        for _ in 0..sz {
            let key = self.read_value()?;
            let val = self.read_value()?;
            if !key.is_instance_of::<PyString>() {
                all_string_keys = false;
            }
            pairs.push((key, val));
        }

        if all_string_keys {
            let dict = PyDict::new(self.py);
            for (k, v) in pairs {
                dict.set_item(k, v)?;
            }
            Ok(dict.into_any())
        } else {
            let list = PyList::empty(self.py);
            for (k, v) in pairs {
                let pair = PyList::new(self.py, [k, v])?;
                list.append(pair)?;
            }
            Ok(list.into_any())
        }
    }

    fn read_map_entry_iter(&mut self) -> Result<Bound<'py, PyDict>> {
        let dict = PyDict::new(self.py);
        loop {
            match self.read_value_or_end()? {
                None => break,
                Some(key) => {
                    let val = self.read_value()?;
                    dict.set_item(key, val)?;
                }
            }
        }
        Ok(dict)
    }

    fn read_iterator(&mut self) -> Result<Bound<'py, PyList>> {
        let list = PyList::empty(self.py);
        loop {
            match self.read_value_or_end()? {
                None => break,
                Some(v) => list.append(v)?,
            }
        }
        Ok(list)
    }

    fn read_enum_field_value(&mut self) -> Result<Bound<'py, PyDict>> {
        let int_val = self.read_value()?;
        let str_val = self.read_value()?;
        let dict = PyDict::new(self.py);
        dict.set_item("int", int_val)?;
        dict.set_item("string", str_val)?;
        Ok(dict)
    }

    fn read_primitive_array(&mut self) -> Result<Bound<'py, PyAny>> {
        let sub_tag = self.reader.read_u8()?;
        let len = self.reader.read_vint()? as usize;

        match sub_tag {
            tag::FLOAT => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    list.append(self.reader.read_f32()? as f64)?;
                }
                Ok(list.into_any())
            }
            tag::INT => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    list.append(self.reader.read_i32()?)?;
                }
                Ok(list.into_any())
            }
            tag::LONG => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    list.append(self.reader.read_i64()?)?;
                }
                Ok(list.into_any())
            }
            tag::DOUBLE => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    list.append(self.reader.read_f64()?)?;
                }
                Ok(list.into_any())
            }
            tag::SHORT => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    list.append(self.reader.read_i16()?)?;
                }
                Ok(list.into_any())
            }
            tag::BOOL_TRUE | tag::BOOL_FALSE => {
                let list = PyList::empty(self.py);
                for _ in 0..len {
                    let b = self.reader.read_u8()?;
                    list.append(b != tag::BOOL_FALSE)?;
                }
                Ok(list.into_any())
            }
            tag::BYTE => {
                let bytes = self.reader.read_exact(len)?;
                Ok(PyBytes::new(self.py, bytes).into_any())
            }
            other => Err(DecodeError::UnknownTag {
                tag: other,
                offset: self.reader.pos - 1,
            }
            .into()),
        }
    }

    // -- Solr-specific containers ----------------------------------------------

    /// Read one entry of a `SolrDocument`'s (or `SolrInputDocument`'s) field
    /// list: either a field name (always a `STR`/`EXTERN_STRING`-tagged
    /// string) or a nested child document (`SOLRDOC`).
    ///
    /// This bypasses the generic [`Decoder::read_slot`]/[`Slot`] machinery
    /// deliberately: dispatching on the raw tag byte here means telling a
    /// field name apart from a child document never needs a Python-level
    /// `isinstance` check (unlike going through a type-erased `Bound<PyAny>`
    /// first), which matters because this runs once per field of every
    /// document in a response.
    ///
    /// If `skip_float_boost` is set, a leading `FLOAT`-tagged per-field boost
    /// (as written by `SolrInputDocument` encoding) is silently consumed and
    /// skipped before looking for the actual name/child.
    fn read_field_or_child(&mut self, skip_float_boost: bool) -> Result<FieldOrChild<'py>> {
        loop {
            let start = self.reader.pos;
            let t = self.reader.read_u8()?;
            let hi = t >> 5;

            match hi {
                1 => {
                    let name = PyString::new(self.py, self.reader.read_str_tagged(t)?);
                    return Ok(FieldOrChild::Name(name));
                }
                7 => return Ok(FieldOrChild::Name(self.read_extern_string(t)?)),
                0 if t == tag::SOLRDOC => {
                    return Ok(FieldOrChild::Child(self.read_solr_document()?));
                }
                0 if skip_float_boost && t == tag::FLOAT => {
                    self.reader.read_f32()?; // discard per-field boost
                    continue;
                }
                _ => {
                    return Err(DecodeError::TypeMismatch {
                        expected: "field name (string) or child SolrDocument",
                        found: "other",
                        offset: start,
                    }
                    .into());
                }
            }
        }
    }

    fn read_solr_document(&mut self) -> Result<Bound<'py, PyDict>> {
        let inner_tag = self.reader.read_u8()?;
        let sz = self.reader.read_size(inner_tag)?;

        let dict = PyDict::new(self.py);
        let mut children: Option<Bound<'py, PyList>> = None;

        for _ in 0..sz {
            match self.read_field_or_child(false)? {
                FieldOrChild::Child(child) => {
                    let list = match &children {
                        Some(l) => l.clone(),
                        None => {
                            let l = PyList::empty(self.py);
                            children = Some(l.clone());
                            l
                        }
                    };
                    list.append(child)?;
                }
                FieldOrChild::Name(field_name) => {
                    let field_val = self.read_value()?;
                    dict.set_item(field_name, field_val)?;
                }
            }
        }

        if let Some(children) = children {
            dict.set_item("_childDocuments_", children)?;
        }

        Ok(dict)
    }

    fn read_solr_input_document(&mut self) -> Result<Bound<'py, PyDict>> {
        // SOLRINPUTDOC: VInt size, then a document boost (float, ignored),
        // then `size` entries of either [boost(float), name, value],
        // [name, value] or a nested child document.
        let sz = self.reader.read_vint()? as usize;
        let _doc_boost = self.read_value()?; // always a Float, historically the doc boost

        let dict = PyDict::new(self.py);
        let mut children: Option<Bound<'py, PyList>> = None;

        for _ in 0..sz {
            match self.read_field_or_child(true)? {
                FieldOrChild::Child(child) => {
                    let list = match &children {
                        Some(l) => l.clone(),
                        None => {
                            let l = PyList::empty(self.py);
                            children = Some(l.clone());
                            l
                        }
                    };
                    list.append(child)?;
                }
                FieldOrChild::Name(field_name) => {
                    let field_val = self.read_value()?;
                    dict.set_item(field_name, field_val)?;
                }
            }
        }

        if let Some(children) = children {
            dict.set_item("_childDocuments_", children)?;
        }

        Ok(dict)
    }

    fn read_solr_document_list(&mut self) -> Result<Bound<'py, PyDict>> {
        let header = self.read_value()?;
        let header = header
            .cast_into::<PyList>()
            .map_err(|_| DecodeError::TypeMismatch {
                expected: "SolrDocumentList header array",
                found: "non-array",
                offset: self.reader.pos,
            })?;

        let dict = PyDict::new(self.py);
        dict.set_item(
            "numFound",
            header
                .get_item(0)
                .ok()
                .unwrap_or_else(|| self.py.None().into_bound(self.py)),
        )?;
        dict.set_item(
            "start",
            header
                .get_item(1)
                .ok()
                .unwrap_or_else(|| self.py.None().into_bound(self.py)),
        )?;
        dict.set_item(
            "maxScore",
            header
                .get_item(2)
                .ok()
                .unwrap_or_else(|| self.py.None().into_bound(self.py)),
        )?;
        dict.set_item(
            "numFoundExact",
            header
                .get_item(3)
                .ok()
                .unwrap_or_else(|| self.py.None().into_bound(self.py)),
        )?;

        let docs = self.read_value()?;
        dict.set_item("docs", docs)?;

        Ok(dict)
    }

    fn expect_str(&mut self) -> Result<Bound<'py, PyString>> {
        let offset = self.reader.pos;
        match self.read_value()? {
            v if v.is_instance_of::<PyString>() => {
                Ok(v.cast_into::<PyString>().expect("checked above"))
            }
            _ => Err(DecodeError::TypeMismatch {
                expected: "string",
                found: "non-string",
                offset,
            }
            .into()),
        }
    }
}

/// Decode a complete javabin (protocol version 2) message directly into
/// native Python objects.
///
/// See the module docs for why this is faster than decoding into
/// [`crate::value::Value`] first and converting afterwards.
pub fn decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let mut decoder = Decoder::new(py, data);

    let version = decoder.reader.read_u8()?;
    if version != crate::reader::EXPECTED_VERSION {
        return Err(DecodeError::InvalidVersion { found: version }.into());
    }

    let value = decoder.read_value()?;

    if decoder.reader.pos != decoder.reader.data.len() {
        return Err(DecodeError::TrailingData {
            remaining: decoder.reader.data.len() - decoder.reader.pos,
        }
        .into());
    }

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyBool, PyFloat};

    const V: u8 = crate::reader::EXPECTED_VERSION;

    fn with_version(mut body: Vec<u8>) -> Vec<u8> {
        let mut out = vec![V];
        out.append(&mut body);
        out
    }

    fn write_vint(out: &mut Vec<u8>, mut i: u32) {
        while i & !0x7F != 0 {
            out.push(((i & 0x7f) | 0x80) as u8);
            i >>= 7;
        }
        out.push(i as u8);
    }

    #[test]
    fn decodes_null_and_bools() {
        Python::attach(|py| {
            assert!(
                decode(py, &with_version(vec![tag::NULL]))
                    .unwrap()
                    .is_none()
            );
            assert!(
                decode(py, &with_version(vec![tag::BOOL_TRUE]))
                    .unwrap()
                    .cast::<PyBool>()
                    .unwrap()
                    .is_true()
            );
            assert!(
                !decode(py, &with_version(vec![tag::BOOL_FALSE]))
                    .unwrap()
                    .cast::<PyBool>()
                    .unwrap()
                    .is_true()
            );
        });
    }

    #[test]
    fn decodes_ints_and_longs() {
        Python::attach(|py| {
            let v = decode(py, &with_version(vec![tag::SINT | 5])).unwrap();
            assert_eq!(v.extract::<i64>().unwrap(), 5);

            let mut body = vec![tag::INT];
            body.extend_from_slice(&(-42i32).to_be_bytes());
            let v = decode(py, &with_version(body)).unwrap();
            assert_eq!(v.extract::<i64>().unwrap(), -42);

            let mut body = vec![tag::LONG];
            body.extend_from_slice(&1_870_516_012_295_651_331i64.to_be_bytes());
            let v = decode(py, &with_version(body)).unwrap();
            assert_eq!(v.extract::<i64>().unwrap(), 1_870_516_012_295_651_331);
        });
    }

    #[test]
    fn decodes_float_and_double() {
        Python::attach(|py| {
            let mut body = vec![tag::FLOAT];
            body.extend_from_slice(&1.5f32.to_be_bytes());
            let v = decode(py, &with_version(body)).unwrap();
            assert!(v.cast::<PyFloat>().is_ok());
            assert_eq!(v.extract::<f64>().unwrap(), 1.5);

            let mut body = vec![tag::DOUBLE];
            body.extend_from_slice(&2.25f64.to_be_bytes());
            let v = decode(py, &with_version(body)).unwrap();
            assert_eq!(v.extract::<f64>().unwrap(), 2.25);
        });
    }

    #[test]
    fn decodes_string_and_bytes() {
        Python::attach(|py| {
            let s = "hello";
            let mut body = vec![tag::STR | s.len() as u8];
            body.extend_from_slice(s.as_bytes());
            let v = decode(py, &with_version(body)).unwrap();
            assert_eq!(v.extract::<String>().unwrap(), "hello");

            let mut body = vec![tag::BYTEARR];
            write_vint(&mut body, 3);
            body.extend_from_slice(&[1, 2, 3]);
            let v = decode(py, &with_version(body)).unwrap();
            assert_eq!(v.extract::<Vec<u8>>().unwrap(), vec![1, 2, 3]);
        });
    }

    #[test]
    fn decodes_array_and_named_list() {
        Python::attach(|py| {
            let body = vec![tag::ARR | 2, tag::SINT | 1, tag::SINT | 2];
            let v = decode(py, &with_version(body)).unwrap();
            let list = v.cast::<PyList>().unwrap();
            assert_eq!(list.len(), 2);
            assert_eq!(list.get_item(0).unwrap().extract::<i64>().unwrap(), 1);

            let mut body = vec![tag::NAMED_LST | 1];
            body.push(tag::STR | 1);
            body.push(b'a');
            body.push(tag::SINT | 1);
            let v = decode(py, &with_version(body)).unwrap();
            let dict = v.cast::<PyDict>().unwrap();
            assert_eq!(
                dict.get_item("a")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1
            );
        });
    }

    #[test]
    fn interns_repeated_extern_string_field_names() {
        Python::attach(|py| {
            // Two SolrDocuments, each with a single field named "foo" written
            // via EXTERN_STRING: first occurrence defines it (idx=0), second
            // references it (idx=1). Both decoded keys must be the *same*
            // interned Python string object.
            let mut body = vec![tag::ARR | 2];

            body.push(tag::SOLRDOC);
            body.push(tag::ORDERED_MAP | 1);
            body.push(tag::EXTERN_STRING); // idx 0: define "foo"
            body.push(tag::STR | 3);
            body.extend_from_slice(b"foo");
            body.push(tag::SINT | 1);

            body.push(tag::SOLRDOC);
            body.push(tag::ORDERED_MAP | 1);
            body.push(tag::EXTERN_STRING | 1); // idx 1: reference "foo"
            body.push(tag::SINT | 2);

            let v = decode(py, &with_version(body)).unwrap();
            let list = v.cast::<PyList>().unwrap();
            let doc0 = list.get_item(0).unwrap();
            let doc1 = list.get_item(1).unwrap();
            let dict0 = doc0.cast::<PyDict>().unwrap();
            let dict1 = doc1.cast::<PyDict>().unwrap();

            let key0 = dict0.keys().get_item(0).unwrap();
            let key1 = dict1.keys().get_item(0).unwrap();
            assert!(key0.is(&key1), "expected the same interned string object");
        });
    }

    #[test]
    fn decodes_solr_document_with_child() {
        Python::attach(|py| {
            let mut body = vec![tag::SOLRDOC, tag::ORDERED_MAP | 2];
            body.push(tag::STR | 2);
            body.extend_from_slice(b"id");
            body.push(tag::STR | 1);
            body.push(b'1');
            body.push(tag::SOLRDOC);
            body.push(tag::ORDERED_MAP | 1);
            body.push(tag::STR | 2);
            body.extend_from_slice(b"id");
            body.push(tag::STR | 1);
            body.push(b'2');

            let v = decode(py, &with_version(body)).unwrap();
            let dict = v.cast::<PyDict>().unwrap();
            assert_eq!(
                dict.get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "1"
            );
            let children = dict.get_item("_childDocuments_").unwrap().unwrap();
            let children = children.cast::<PyList>().unwrap();
            assert_eq!(children.len(), 1);
            let child = children.get_item(0).unwrap();
            let child = child.cast::<PyDict>().unwrap();
            assert_eq!(
                child
                    .get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "2"
            );
        });
    }

    #[test]
    fn decodes_solr_document_list() {
        Python::attach(|py| {
            let mut body = vec![tag::SOLRDOCLST];
            body.push(tag::ARR | 4);
            body.push(tag::SLONG | 1);
            body.push(tag::SLONG | 0);
            body.push(tag::NULL);
            body.push(tag::BOOL_TRUE);
            body.push(tag::ARR | 1);
            body.push(tag::SOLRDOC);
            body.push(tag::ORDERED_MAP | 1);
            body.push(tag::STR | 2);
            body.extend_from_slice(b"id");
            body.push(tag::STR | 1);
            body.push(b'1');

            let v = decode(py, &with_version(body)).unwrap();
            let dict = v.cast::<PyDict>().unwrap();
            assert_eq!(
                dict.get_item("numFound")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1
            );
            assert!(dict.get_item("maxScore").unwrap().unwrap().is_none());
            assert!(
                dict.get_item("numFoundExact")
                    .unwrap()
                    .unwrap()
                    .cast::<PyBool>()
                    .unwrap()
                    .is_true()
            );
            let docs = dict.get_item("docs").unwrap().unwrap();
            assert_eq!(docs.cast::<PyList>().unwrap().len(), 1);
        });
    }

    #[test]
    fn rejects_wrong_version() {
        Python::attach(|py| {
            let err = decode(py, &[1u8, tag::NULL]).unwrap_err();
            assert!(err.to_string().contains("version"));
        });
    }

    #[test]
    fn rejects_oversized_map_size_claim_without_huge_allocation() {
        // See decoder::tests::rejects_oversized_map_size_claim_without_huge_allocation:
        // a MAP claiming ~4 billion entries with no further bytes must fail
        // fast with a catchable error rather than attempting a huge
        // `Vec::with_capacity` up front.
        Python::attach(|py| {
            let mut body = vec![tag::MAP];
            write_vint(&mut body, u32::MAX - 10);
            assert!(decode(py, &with_version(body)).is_err());
        });
    }

    #[test]
    fn rejects_excessively_nested_input() {
        // See decoder::tests::rejects_excessively_nested_input: before the
        // recursion-depth guard this would overflow the call stack (SIGSEGV)
        // instead of returning a catchable Python exception.
        Python::attach(|py| {
            let mut body = vec![tag::ARR | 1; 10_000];
            body.push(tag::NULL);
            let err = decode(py, &with_version(body)).unwrap_err();
            assert!(err.to_string().contains("nesting"), "{err}");
        });
    }

    #[test]
    fn decodes_nesting_within_the_depth_limit() {
        Python::attach(|py| {
            let depth = 100;
            let mut body = vec![tag::ARR | 1; depth];
            body.push(tag::NULL);
            let value = decode(py, &with_version(body)).unwrap();

            let mut v = value;
            for _ in 0..depth {
                let list = v.cast::<PyList>().unwrap();
                assert_eq!(list.len(), 1);
                v = list.get_item(0).unwrap();
            }
            assert!(v.is_none());
        });
    }

    #[test]
    fn generic_map_falls_back_to_pair_list_for_non_string_keys() {
        Python::attach(|py| {
            // MAP{1: "a"} - an integer key, so the result must be a list of
            // [key, value] pairs rather than a dict.
            let mut body = vec![tag::MAP];
            write_vint(&mut body, 1);
            body.push(tag::SINT | 1);
            body.push(tag::STR | 1);
            body.push(b'a');

            let v = decode(py, &with_version(body)).unwrap();
            let list = v.cast::<PyList>().unwrap();
            assert_eq!(list.len(), 1);
            let pair = list.get_item(0).unwrap();
            let pair = pair.cast::<PyList>().unwrap();
            assert_eq!(pair.get_item(0).unwrap().extract::<i64>().unwrap(), 1);
            assert_eq!(pair.get_item(1).unwrap().extract::<String>().unwrap(), "a");
        });
    }
}
