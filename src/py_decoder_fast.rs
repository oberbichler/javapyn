//! FFI-level fast path for [`crate::deserialize`].
//!
//! [`crate::py_decoder`] is a clean, safe PyO3 implementation, but every
//! `Bound`/`set_item`/`append`/`into_bound_py_any` call carries per-object
//! overhead (bound-wrapper construction, `PyResult` plumbing, refcount
//! juggling). For a Solr response of a few thousand documents that means
//! constructing ~1M Python objects, and that overhead dominates.
//!
//! This module talks to `pyo3::ffi` directly. It keeps exactly the same
//! output shape and semantics as [`crate::py_decoder`], but:
//!
//! - builds lists with `PyList_New(len)` + `PyList_SetItem` (steals the
//!   reference, no per-element append overhead),
//! - inserts dict items with `PyDict_SetItem` on raw pointers,
//! - creates ints/floats/bools/None with the dedicated C constructors and
//!   cached singletons,
//! - interns repeated field-name strings (see [`crate::py_decoder`] docs for
//!   why that's correct and beneficial).
//!
//! Deliberately uses `PyList_SetItem` (a real, bounds/type-checked function
//! call) rather than the `PyList_SET_ITEM` macro, which reaches directly into
//! `PyListObject`'s field layout and is therefore unavailable under
//! `Py_LIMITED_API`/`abi3` (this crate builds one `abi3` wheel per platform
//! rather than one per Python minor version; see `Cargo.toml`). Both steal
//! the reference identically; `PyList_SetItem` additionally returns an error
//! code instead of assuming success.
//!
//! # Safety
//!
//! All `unsafe` here upholds CPython's reference-counting contract: [`Obj`]
//! owns exactly one strong reference and releases it on drop; functions that
//! *steal* a reference (`PyList_SetItem`) are handed an [`Obj`] by value via
//! [`Obj::into_ptr`] so the `Drop` doesn't also release it; borrowing APIs
//! (`PyDict_SetItem`, which increments) are given borrowed pointers and the
//! [`Obj`] retains ownership. Every constructor return value is null-checked
//! and turned into a Python exception.

use pyo3::ffi;
use pyo3::prelude::*;

use crate::reader::{DecodeError, Reader, tag};

type Result<T> = std::result::Result<T, DecodeError>;

/// An owned strong reference to a Python object.
///
/// Analogous to a minimal `Py<PyAny>` but without any generic machinery.
/// Decrefs on drop unless the reference has been moved out via
/// [`Obj::into_ptr`].
pub(crate) struct Obj(*mut ffi::PyObject);

impl Obj {
    /// Wrap a freshly-created (owned) pointer, returning an error if it's
    /// null (i.e. a Python exception is already set).
    #[inline]
    unsafe fn from_owned(ptr: *mut ffi::PyObject) -> Result<Self> {
        if ptr.is_null() {
            Err(DecodeError::PyErr)
        } else {
            Ok(Obj(ptr))
        }
    }

    /// Wrap a borrowed pointer (e.g. a cached singleton), taking a new strong
    /// reference to it.
    #[inline]
    unsafe fn from_borrowed(ptr: *mut ffi::PyObject) -> Self {
        unsafe { ffi::Py_INCREF(ptr) };
        Obj(ptr)
    }

    /// Consume `self`, returning the raw pointer *without* decref'ing. The
    /// caller now owns the reference (used when passing to a reference-
    /// stealing API like `PyList_SetItem`).
    #[inline]
    pub(crate) fn into_ptr(self) -> *mut ffi::PyObject {
        let ptr = self.0;
        std::mem::forget(self);
        ptr
    }

    /// Borrow the raw pointer without transferring ownership.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut ffi::PyObject {
        self.0
    }
}

impl Drop for Obj {
    #[inline]
    fn drop(&mut self) {
        unsafe { ffi::Py_DECREF(self.0) };
    }
}

/// One entry beneath a `SOLRDOC` field list: a field name or a child doc.
enum FieldOrChild {
    Name(Obj),
    Child(Obj),
}

pub(crate) struct Decoder<'a, 'py> {
    reader: Reader<'a>,
    /// GIL token. Not read directly, but its existence is what makes every
    /// `unsafe` `ffi::*` call in this module sound: holding a `Python<'py>`
    /// proves the GIL is held for `'py`, which the CPython C-API requires.
    #[allow(dead_code)]
    py: Python<'py>,
    /// Interned field-name strings (see module docs). Stored as [`Obj`] so
    /// each cached entry keeps one strong ref alive for the whole decode.
    strings: Vec<Obj>,
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

    // -- helpers used by the incremental streaming decoder --------------------

    /// Construct a decoder over `data` seeded with a pre-existing interned
    /// string cache (so `EXTERN_STRING` references resolve across chunks).
    pub(crate) fn for_stream(py: Python<'py>, data: &'a [u8], strings: Vec<Obj>) -> Self {
        Self {
            reader: Reader::new(data),
            py,
            strings,
            depth: 0,
        }
    }

    /// Current read offset into the slice.
    pub(crate) fn reader_pos(&self) -> usize {
        self.reader.pos
    }

    /// Read the leading version byte.
    pub(crate) fn reader_read_u8(&mut self) -> Result<u8> {
        self.reader.read_u8()
    }

    /// Move the interned-string cache out (to carry it to the next chunk).
    pub(crate) fn take_strings(self) -> Vec<Obj> {
        self.strings
    }

    /// Decode exactly one value (a document) from the current position.
    pub(crate) fn read_one_value_public(&mut self) -> Result<Obj> {
        self.read_value()
    }

    /// Consume the response envelope up to (and including) the `docs`
    /// container's tag byte, discarding the envelope values. Returns which
    /// kind of document sequence follows. Used by the incremental streaming
    /// decoder, which needs to reach the doc stream without materialising the
    /// wrapper.
    ///
    /// Any `UnexpectedEof` means the (small) envelope isn't fully buffered yet
    /// and the caller should wait for more bytes.
    pub(crate) fn stream_envelope_to_docs(&mut self) -> Result<DocsPhase> {
        let t = self.reader.read_u8()?;
        let hi = t >> 5;
        match hi {
            // top-level NamedList / SimpleOrderedMap
            5 | 6 => {
                let sz = self.reader.read_size(t)?;
                for _ in 0..sz {
                    let name = self.expect_str()?;
                    if let Some(p) = self.envelope_value_to_docs(&name)? {
                        return Ok(p);
                    }
                }
            }
            _ => {
                if t == tag::MAP_ENTRY_ITER {
                    loop {
                        // END terminates the map
                        if self.peek_end()? {
                            break;
                        }
                        let key = self.read_value()?;
                        if let Some(p) = self.envelope_value_to_docs(&key)? {
                            return Ok(p);
                        }
                    }
                } else {
                    // Not a recognised envelope: treat the whole thing as a
                    // single value that is itself the docs? Unsupported for
                    // streaming — signal empty.
                    return Ok(DocsPhase::None);
                }
            }
        }
        Ok(DocsPhase::None)
    }

    /// Handle one envelope entry. Returns `Some(phase)` if this entry is the
    /// `response`/`result-set` container and we've reached its `docs`
    /// sequence; `None` (after skipping the value) otherwise.
    fn envelope_value_to_docs(&mut self, key: &Obj) -> Result<Option<DocsPhase>> {
        let ks = obj_as_str(key);
        if ks == Some("response") || ks == Some("result-set") {
            let t = self.reader.read_u8()?;
            let hi = t >> 5;
            match hi {
                5 | 6 => {
                    // response as NamedList: find its docs entry
                    let sz = self.reader.read_size(t)?;
                    for _ in 0..sz {
                        let name = self.expect_str()?;
                        if let Some(p) = self.docs_entry(&name)? {
                            return Ok(Some(p));
                        }
                    }
                    return Ok(Some(DocsPhase::None));
                }
                _ => match t {
                    tag::SOLRDOCLST => {
                        // header array (discard), then docs ARR tag
                        let _header = self.read_value()?;
                        let dt = self.reader.read_u8()?;
                        if dt >> 5 == 4 {
                            let sz = self.reader.read_size(dt)?;
                            return Ok(Some(DocsPhase::Arr(sz)));
                        }
                        return Ok(Some(DocsPhase::None));
                    }
                    tag::MAP_ENTRY_ITER => {
                        loop {
                            if self.peek_end()? {
                                break;
                            }
                            let name = self.read_value()?;
                            if let Some(p) = self.docs_entry(&name)? {
                                return Ok(Some(p));
                            }
                        }
                        return Ok(Some(DocsPhase::None));
                    }
                    _ => {
                        self.reader.pos -= 1;
                        let _ = self.read_value()?;
                        return Ok(None);
                    }
                },
            }
        }
        // not the container: skip its value
        let _ = self.read_value()?;
        Ok(None)
    }

    /// For an entry named `docs`, read its container tag and return the phase.
    /// For any other entry, skip its value and return `None`.
    fn docs_entry(&mut self, name: &Obj) -> Result<Option<DocsPhase>> {
        if obj_as_str(name) == Some("docs") {
            let t = self.reader.read_u8()?;
            let hi = t >> 5;
            if hi == 4 {
                let sz = self.reader.read_size(t)?;
                return Ok(Some(DocsPhase::Arr(sz)));
            }
            if t == tag::ITERATOR {
                return Ok(Some(DocsPhase::Iter));
            }
            // unexpected docs encoding: skip
            self.reader.pos -= 1;
            let _ = self.read_value()?;
            return Ok(Some(DocsPhase::None));
        }
        let _ = self.read_value()?;
        Ok(None)
    }

    /// Peek whether the next byte is the `END` marker; consume it if so.
    fn peek_end(&mut self) -> Result<bool> {
        let b = *self
            .reader
            .data
            .get(self.reader.pos)
            .ok_or(DecodeError::UnexpectedEof {
                offset: self.reader.pos,
            })?;
        if b == tag::END {
            self.reader.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // -- scalar object constructors ------------------------------------------
    //
    // These are free functions (not `&self` methods) so that call sites like
    // `py_long(self.reader.read_i64()?)` don't create an immutable borrow of
    // `self` that conflicts with the mutable borrow of `self.reader`.

    // -- string handling ------------------------------------------------------

    /// Read an `EXTERN_STRING`-tagged field name, interning + caching it.
    fn read_extern_string(&mut self, tag_byte: u8) -> Result<Obj> {
        let idx = self.reader.read_size(tag_byte)?;
        if idx != 0 {
            let cached = self
                .strings
                .get(idx - 1)
                .ok_or(DecodeError::UnexpectedEof {
                    offset: self.reader.pos,
                })?;
            Ok(unsafe { Obj::from_borrowed(cached.as_ptr()) })
        } else {
            let inner_tag = self.reader.read_u8()?;
            let s = self.reader.read_str_tagged(inner_tag)?;
            let obj = py_str(s)?;
            // Intern in place: PyUnicode_InternInPlace may replace the pointer
            // with an existing interned object, adjusting refcounts itself.
            let mut ptr = obj.into_ptr();
            unsafe { ffi::PyUnicode_InternInPlace(&mut ptr) };
            let interned = Obj(ptr);
            // Cache a second strong ref for reuse by later index references.
            self.strings
                .push(unsafe { Obj::from_borrowed(interned.as_ptr()) });
            Ok(interned)
        }
    }

    // -- top-level dispatch ---------------------------------------------------

    #[inline]
    fn read_value(&mut self) -> Result<Obj> {
        match self.read_slot()? {
            Slot::Value(o) | Slot::Child(o) => Ok(o),
            Slot::End => Err(DecodeError::TypeMismatch {
                expected: "value",
                found: "END marker",
                offset: self.reader.pos - 1,
            }),
        }
    }

    /// Wraps [`Self::read_slot_inner`] with a nesting-depth guard; see
    /// `decoder::Decoder::read_slot` for the rationale (identical here).
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
            let o = match hi {
                1 => py_str(self.reader.read_str_tagged(t)?)?,
                2 => {
                    let v = self.read_small_int(t)?;
                    py_long(v as i64)?
                }
                3 => {
                    let v = self.read_small_long(t)?;
                    py_long(v)?
                }
                4 => self.read_array(t)?,
                5 | 6 => self.read_named_list(t)?,
                7 => self.read_extern_string(t)?,
                _ => unreachable!("3-bit value out of range"),
            };
            return Ok(Slot::Value(o));
        }

        let o = match t {
            tag::NULL => py_none(),
            tag::BOOL_TRUE => py_bool(true),
            tag::BOOL_FALSE => py_bool(false),
            tag::BYTE => py_long(self.reader.read_i8()? as i64)?,
            tag::SHORT => py_long(self.reader.read_i16()? as i64)?,
            tag::DOUBLE => py_float(self.reader.read_f64()?)?,
            tag::INT => py_long(self.reader.read_i32()? as i64)?,
            tag::LONG => py_long(self.reader.read_i64()?)?,
            tag::FLOAT => py_float(self.reader.read_f32()? as f64)?,
            tag::DATE => py_long(self.reader.read_i64()?)?,
            tag::MAP => self.read_map()?,
            tag::SOLRDOC => return Ok(Slot::Child(self.read_solr_document()?)),
            tag::SOLRDOCLST => self.read_solr_document_list()?,
            tag::BYTEARR => {
                let len = self.reader.read_vint()? as usize;
                let bytes = self.reader.read_exact(len)?;
                py_bytes(bytes)?
            }
            tag::ITERATOR => self.read_iterator()?,
            tag::END => return Ok(Slot::End),
            tag::SOLRINPUTDOC => self.read_solr_input_document()?,
            tag::MAP_ENTRY_ITER => self.read_map_entry_iter()?,
            tag::ENUM_FIELD_VALUE => self.read_enum_field_value()?,
            tag::MAP_ENTRY => {
                let key = self.read_value()?;
                let val = self.read_value()?;
                let dict = self.new_dict()?;
                self.dict_set(&dict, &key, &val)?;
                dict
            }
            tag::PRIMITIVE_ARR => self.read_primitive_array()?,
            other => {
                return Err(DecodeError::UnknownTag {
                    tag: other,
                    offset: start,
                });
            }
        };

        Ok(Slot::Value(o))
    }

    // -- compact numeric encodings -------------------------------------------

    #[inline]
    fn read_small_int(&mut self, tag_byte: u8) -> Result<i32> {
        let mut v = (tag_byte & 0x0F) as i32;
        if tag_byte & 0x10 != 0 {
            v |= (self.reader.read_vint()? as i32) << 4;
        }
        Ok(v)
    }

    #[inline]
    fn read_small_long(&mut self, tag_byte: u8) -> Result<i64> {
        let mut v = (tag_byte & 0x0F) as i64;
        if tag_byte & 0x10 != 0 {
            v |= (self.reader.read_vlong()? as i64) << 4;
        }
        Ok(v)
    }

    // -- container helpers ----------------------------------------------------

    #[inline]
    fn new_dict(&self) -> Result<Obj> {
        unsafe { Obj::from_owned(ffi::PyDict_New()) }
    }

    /// `dict[key] = val` via `PyDict_SetItem` (which increments both key and
    /// value; `self` retains ownership of both `Obj`s).
    #[inline]
    fn dict_set(&self, dict: &Obj, key: &Obj, val: &Obj) -> Result<()> {
        let rc = unsafe { ffi::PyDict_SetItem(dict.as_ptr(), key.as_ptr(), val.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(DecodeError::PyErr)
        }
    }

    /// `dict[key_cstr] = val` for a static ASCII key, via `PyDict_SetItemString`.
    #[inline]
    fn dict_set_str(&self, dict: &Obj, key: &std::ffi::CStr, val: &Obj) -> Result<()> {
        let rc = unsafe { ffi::PyDict_SetItemString(dict.as_ptr(), key.as_ptr(), val.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(DecodeError::PyErr)
        }
    }

    // -- containers -----------------------------------------------------------

    fn read_array(&mut self, tag_byte: u8) -> Result<Obj> {
        let sz = self.reader.read_size(tag_byte)?;
        self.read_list_of(sz)
    }

    /// Build a `list` of exactly `sz` decoded values using `PyList_New` +
    /// `PyList_SetItem` (steals references, no append overhead).
    fn read_list_of(&mut self, sz: usize) -> Result<Obj> {
        let list = unsafe { Obj::from_owned(ffi::PyList_New(sz as ffi::Py_ssize_t))? };
        for i in 0..sz {
            let item = self.read_value()?;
            // `PyList_SetItem` always steals the reference to `item`, even on
            // error (it already decrefs it internally before returning -1),
            // so no extra `Py_DECREF` is needed here.
            let rc = unsafe {
                ffi::PyList_SetItem(list.as_ptr(), i as ffi::Py_ssize_t, item.into_ptr())
            };
            if rc != 0 {
                return Err(DecodeError::PyErr);
            }
        }
        Ok(list)
    }

    fn read_named_list(&mut self, tag_byte: u8) -> Result<Obj> {
        let sz = self.reader.read_size(tag_byte)?;
        let dict = self.new_dict()?;
        for _ in 0..sz {
            let name = self.expect_str()?;
            let val = self.read_value()?;
            self.dict_set(&dict, &name, &val)?;
        }
        Ok(dict)
    }

    fn read_map(&mut self) -> Result<Obj> {
        let sz = self.reader.read_vint()? as usize;

        // Decode all pairs first; if every key is a str, produce a dict,
        // else fall back to a list of [key, value] pairs (see py_decoder).
        let mut pairs: Vec<(Obj, Obj)> = Vec::with_capacity(self.reader.capacity_hint(sz));
        let mut all_string_keys = true;
        for _ in 0..sz {
            let key = self.read_value()?;
            let val = self.read_value()?;
            if unsafe { ffi::PyUnicode_Check(key.as_ptr()) } == 0 {
                all_string_keys = false;
            }
            pairs.push((key, val));
        }

        if all_string_keys {
            let dict = self.new_dict()?;
            for (k, v) in &pairs {
                self.dict_set(&dict, k, v)?;
            }
            Ok(dict)
        } else {
            let list = unsafe { Obj::from_owned(ffi::PyList_New(sz as ffi::Py_ssize_t))? };
            for (i, (k, v)) in pairs.into_iter().enumerate() {
                let pair = unsafe { Obj::from_owned(ffi::PyList_New(2))? };
                // `PyList_SetItem` always steals the reference, even on
                // error, so no extra `Py_DECREF` is needed on failure.
                let rc = unsafe { ffi::PyList_SetItem(pair.as_ptr(), 0, k.into_ptr()) };
                if rc != 0 {
                    return Err(DecodeError::PyErr);
                }
                let rc = unsafe { ffi::PyList_SetItem(pair.as_ptr(), 1, v.into_ptr()) };
                if rc != 0 {
                    return Err(DecodeError::PyErr);
                }
                let rc = unsafe {
                    ffi::PyList_SetItem(list.as_ptr(), i as ffi::Py_ssize_t, pair.into_ptr())
                };
                if rc != 0 {
                    return Err(DecodeError::PyErr);
                }
            }
            Ok(list)
        }
    }

    fn read_map_entry_iter(&mut self) -> Result<Obj> {
        let dict = self.new_dict()?;
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(key) | Slot::Child(key) => {
                    let val = self.read_value()?;
                    self.dict_set(&dict, &key, &val)?;
                }
            }
        }
        Ok(dict)
    }

    fn read_iterator(&mut self) -> Result<Obj> {
        // Unknown length: append into a list built with PyList_Append.
        let list = unsafe { Obj::from_owned(ffi::PyList_New(0))? };
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(v) | Slot::Child(v) => {
                    let rc = unsafe { ffi::PyList_Append(list.as_ptr(), v.as_ptr()) };
                    if rc != 0 {
                        return Err(DecodeError::PyErr);
                    }
                }
            }
        }
        Ok(list)
    }

    fn read_enum_field_value(&mut self) -> Result<Obj> {
        let int_val = self.read_value()?;
        let str_val = self.read_value()?;
        let dict = self.new_dict()?;
        self.dict_set_str(&dict, c"int", &int_val)?;
        self.dict_set_str(&dict, c"string", &str_val)?;
        Ok(dict)
    }

    fn read_primitive_array(&mut self) -> Result<Obj> {
        let sub_tag = self.reader.read_u8()?;
        let len = self.reader.read_vint()? as usize;

        if sub_tag == tag::BYTE {
            let bytes = self.reader.read_exact(len)?;
            return py_bytes(bytes);
        }

        let list = unsafe { Obj::from_owned(ffi::PyList_New(len as ffi::Py_ssize_t))? };
        for i in 0..len {
            let item = match sub_tag {
                tag::FLOAT => py_float(self.reader.read_f32()? as f64)?,
                tag::INT => py_long(self.reader.read_i32()? as i64)?,
                tag::LONG => py_long(self.reader.read_i64()?)?,
                tag::DOUBLE => py_float(self.reader.read_f64()?)?,
                tag::SHORT => py_long(self.reader.read_i16()? as i64)?,
                tag::BOOL_TRUE | tag::BOOL_FALSE => {
                    let b = self.reader.read_u8()?;
                    py_bool(b != tag::BOOL_FALSE)
                }
                other => {
                    return Err(DecodeError::UnknownTag {
                        tag: other,
                        offset: self.reader.pos - 1,
                    });
                }
            };
            // `PyList_SetItem` always steals the reference to `item`, even on
            // error, so no extra `Py_DECREF` is needed here.
            let rc = unsafe {
                ffi::PyList_SetItem(list.as_ptr(), i as ffi::Py_ssize_t, item.into_ptr())
            };
            if rc != 0 {
                return Err(DecodeError::PyErr);
            }
        }
        Ok(list)
    }

    // -- Solr-specific containers ----------------------------------------------

    /// Read one field-list entry: a field name (STR/EXTERN_STRING) or a child
    /// document (SOLRDOC), dispatching on the raw tag byte to avoid any
    /// Python-level type check.
    fn read_field_or_child(&mut self, skip_float_boost: bool) -> Result<FieldOrChild> {
        loop {
            let start = self.reader.pos;
            let t = self.reader.read_u8()?;
            let hi = t >> 5;
            match hi {
                1 => return Ok(FieldOrChild::Name(py_str(self.reader.read_str_tagged(t)?)?)),
                7 => return Ok(FieldOrChild::Name(self.read_extern_string(t)?)),
                0 if t == tag::SOLRDOC => {
                    return Ok(FieldOrChild::Child(self.read_solr_document()?));
                }
                0 if skip_float_boost && t == tag::FLOAT => {
                    self.reader.read_f32()?;
                    continue;
                }
                _ => {
                    return Err(DecodeError::TypeMismatch {
                        expected: "field name (string) or child SolrDocument",
                        found: "other",
                        offset: start,
                    });
                }
            }
        }
    }

    fn read_solr_document(&mut self) -> Result<Obj> {
        let inner_tag = self.reader.read_u8()?;
        let sz = self.reader.read_size(inner_tag)?;

        let dict = self.new_dict()?;
        let mut children: Option<Obj> = None;

        for _ in 0..sz {
            match self.read_field_or_child(false)? {
                FieldOrChild::Child(child) => {
                    let list = match &children {
                        Some(l) => l,
                        None => children.insert(unsafe { Obj::from_owned(ffi::PyList_New(0))? }),
                    };
                    let rc = unsafe { ffi::PyList_Append(list.as_ptr(), child.as_ptr()) };
                    if rc != 0 {
                        return Err(DecodeError::PyErr);
                    }
                }
                FieldOrChild::Name(name) => {
                    let val = self.read_value()?;
                    self.dict_set(&dict, &name, &val)?;
                }
            }
        }

        if let Some(children) = children {
            self.dict_set_str(&dict, c"_childDocuments_", &children)?;
        }

        Ok(dict)
    }

    fn read_solr_input_document(&mut self) -> Result<Obj> {
        let sz = self.reader.read_vint()? as usize;
        let _doc_boost = self.read_value()?; // discard historical doc boost (Float)

        let dict = self.new_dict()?;
        let mut children: Option<Obj> = None;

        for _ in 0..sz {
            match self.read_field_or_child(true)? {
                FieldOrChild::Child(child) => {
                    let list = match &children {
                        Some(l) => l,
                        None => children.insert(unsafe { Obj::from_owned(ffi::PyList_New(0))? }),
                    };
                    let rc = unsafe { ffi::PyList_Append(list.as_ptr(), child.as_ptr()) };
                    if rc != 0 {
                        return Err(DecodeError::PyErr);
                    }
                }
                FieldOrChild::Name(name) => {
                    let val = self.read_value()?;
                    self.dict_set(&dict, &name, &val)?;
                }
            }
        }

        if let Some(children) = children {
            self.dict_set_str(&dict, c"_childDocuments_", &children)?;
        }

        Ok(dict)
    }

    fn read_solr_document_list(&mut self) -> Result<Obj> {
        let header = self.read_value()?;
        if unsafe { ffi::PyList_Check(header.as_ptr()) } == 0 {
            return Err(DecodeError::TypeMismatch {
                expected: "SolrDocumentList header array",
                found: "non-array",
                offset: self.reader.pos,
            });
        }

        let dict = self.new_dict()?;
        // Borrow header items (PyList_GetItem returns a borrowed ref).
        let get = |i: ffi::Py_ssize_t| -> Obj {
            let ptr = unsafe { ffi::PyList_GetItem(header.as_ptr(), i) };
            if ptr.is_null() {
                unsafe { ffi::PyErr_Clear() };
                py_none()
            } else {
                unsafe { Obj::from_borrowed(ptr) }
            }
        };
        let num_found = get(0);
        let start = get(1);
        let max_score = get(2);
        let num_found_exact = get(3);
        self.dict_set_str(&dict, c"numFound", &num_found)?;
        self.dict_set_str(&dict, c"start", &start)?;
        self.dict_set_str(&dict, c"maxScore", &max_score)?;
        self.dict_set_str(&dict, c"numFoundExact", &num_found_exact)?;

        let docs = self.read_value()?;
        self.dict_set_str(&dict, c"docs", &docs)?;

        Ok(dict)
    }

    fn expect_str(&mut self) -> Result<Obj> {
        let offset = self.reader.pos;
        let v = self.read_value()?;
        if unsafe { ffi::PyUnicode_Check(v.as_ptr()) } != 0 {
            Ok(v)
        } else {
            Err(DecodeError::TypeMismatch {
                expected: "string",
                found: "non-string",
                offset,
            })
        }
    }

    // -- streaming --------------------------------------------------------------
    //
    // The streaming entry points decode the small response *envelope*
    // (`responseHeader`, `numFound`, ...) normally, but as soon as they reach
    // the document sequence they hand each document to `callback` and drop it,
    // so the whole set is never materialised at once. This is possible because
    // every javabin document is self-contained and the doc sequences are
    // encoded as a fixed-length `ARR` (in a `SOLRDOCLST`, from `/select`) or a
    // variable-length `ITERATOR` (from `/export` and `/stream`).
    //
    // `callback(doc)` is a Python callable invoked once per document. Its
    // return value is ignored (and dropped).

    /// Decode the whole message, but stream documents to `callback` instead of
    /// collecting them. Returns the *envelope* — the same structure
    /// [`decode`] would return, except the documents are omitted (the `docs`
    /// list is left empty / not populated).
    fn stream(&mut self, callback: *mut ffi::PyObject) -> Result<Obj> {
        let t = self.reader.read_u8()?;
        let hi = t >> 5;
        match hi {
            // Top level is a NamedList / SimpleOrderedMap (the /select and
            // /stream case): {"responseHeader": ..., "response"/"result-set": ...}
            5 | 6 => self.stream_named_list(t, callback),
            _ => match t {
                // Top level is a MAP_ENTRY_ITER (rare top-level /export shape).
                tag::MAP_ENTRY_ITER => self.stream_map_entry_iter(callback),
                // Fallback: not a recognised envelope — decode normally. The
                // caller still gets a valid object; nothing is streamed.
                _ => {
                    self.reader.pos -= 1;
                    self.read_value()
                }
            },
        }
    }

    fn stream_named_list(&mut self, tag_byte: u8, callback: *mut ffi::PyObject) -> Result<Obj> {
        let sz = self.reader.read_size(tag_byte)?;
        let dict = self.new_dict()?;
        for _ in 0..sz {
            let name = self.expect_str()?;
            let val = self.stream_envelope_value(&name, callback)?;
            self.dict_set(&dict, &name, &val)?;
        }
        Ok(dict)
    }

    fn stream_map_entry_iter(&mut self, callback: *mut ffi::PyObject) -> Result<Obj> {
        let dict = self.new_dict()?;
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(key) | Slot::Child(key) => {
                    let val = self.stream_envelope_value(&key, callback)?;
                    self.dict_set(&dict, &key, &val)?;
                }
            }
        }
        Ok(dict)
    }

    /// Decode one envelope value. If it's the `response`/`result-set`
    /// container (which holds the documents), recurse into it in streaming
    /// mode; otherwise decode it normally.
    fn stream_envelope_value(&mut self, key: &Obj, callback: *mut ffi::PyObject) -> Result<Obj> {
        let key_str = obj_as_str(key);
        if key_str == Some("response") || key_str == Some("result-set") {
            let t = self.reader.read_u8()?;
            let hi = t >> 5;
            match hi {
                5 | 6 => return self.stream_response_named_list(t, callback),
                _ => match t {
                    tag::SOLRDOCLST => return self.stream_solr_doc_list(callback),
                    tag::MAP_ENTRY_ITER => return self.stream_response_map_entry_iter(callback),
                    _ => {
                        self.reader.pos -= 1;
                    }
                },
            }
        }
        self.read_value()
    }

    /// A `response`/`result-set` encoded as a NamedList: stream its `docs`
    /// entry, decode everything else normally.
    fn stream_response_named_list(
        &mut self,
        tag_byte: u8,
        callback: *mut ffi::PyObject,
    ) -> Result<Obj> {
        let sz = self.reader.read_size(tag_byte)?;
        let dict = self.new_dict()?;
        for _ in 0..sz {
            let name = self.expect_str()?;
            let val = self.stream_docs_or_value(&name, callback)?;
            self.dict_set(&dict, &name, &val)?;
        }
        Ok(dict)
    }

    /// A `response`/`result-set` encoded as a MAP_ENTRY_ITER (the /export and
    /// /stream case): stream its `docs` entry.
    fn stream_response_map_entry_iter(&mut self, callback: *mut ffi::PyObject) -> Result<Obj> {
        let dict = self.new_dict()?;
        loop {
            match self.read_slot()? {
                Slot::End => break,
                Slot::Value(key) | Slot::Child(key) => {
                    let val = self.stream_docs_or_value(&key, callback)?;
                    self.dict_set(&dict, &key, &val)?;
                }
            }
        }
        Ok(dict)
    }

    /// If `key == "docs"`, stream the following document sequence to
    /// `callback` and return an empty list placeholder; otherwise decode the
    /// value normally.
    fn stream_docs_or_value(&mut self, key: &Obj, callback: *mut ffi::PyObject) -> Result<Obj> {
        if obj_as_str(key) == Some("docs") {
            let t = self.reader.read_u8()?;
            let hi = t >> 5;
            match hi {
                // ARR of documents (fixed length)
                4 => {
                    let sz = self.reader.read_size(t)?;
                    for _ in 0..sz {
                        let doc = self.read_value()?;
                        self.invoke(callback, &doc)?;
                    }
                    return unsafe { Obj::from_owned(ffi::PyList_New(0)) };
                }
                _ => match t {
                    // ITERATOR of documents (END-terminated)
                    tag::ITERATOR => {
                        loop {
                            match self.read_slot()? {
                                Slot::End => break,
                                Slot::Value(doc) | Slot::Child(doc) => {
                                    self.invoke(callback, &doc)?;
                                }
                            }
                        }
                        return unsafe { Obj::from_owned(ffi::PyList_New(0)) };
                    }
                    _ => {
                        self.reader.pos -= 1;
                    }
                },
            }
        }
        self.read_value()
    }

    /// A `SOLRDOCLST`: stream the documents (second array) to `callback`,
    /// keep the header (numFound/start/maxScore/numFoundExact) in the returned
    /// dict, with an empty `docs` list.
    fn stream_solr_doc_list(&mut self, callback: *mut ffi::PyObject) -> Result<Obj> {
        let header = self.read_value()?;
        if unsafe { ffi::PyList_Check(header.as_ptr()) } == 0 {
            return Err(DecodeError::TypeMismatch {
                expected: "SolrDocumentList header array",
                found: "non-array",
                offset: self.reader.pos,
            });
        }
        let dict = self.new_dict()?;
        let get = |i: ffi::Py_ssize_t| -> Obj {
            let ptr = unsafe { ffi::PyList_GetItem(header.as_ptr(), i) };
            if ptr.is_null() {
                unsafe { ffi::PyErr_Clear() };
                py_none()
            } else {
                unsafe { Obj::from_borrowed(ptr) }
            }
        };
        let (nf, st, ms, nfe) = (get(0), get(1), get(2), get(3));
        self.dict_set_str(&dict, c"numFound", &nf)?;
        self.dict_set_str(&dict, c"start", &st)?;
        self.dict_set_str(&dict, c"maxScore", &ms)?;
        self.dict_set_str(&dict, c"numFoundExact", &nfe)?;

        // The documents array follows.
        let t = self.reader.read_u8()?;
        let hi = t >> 5;
        if hi == 4 {
            let sz = self.reader.read_size(t)?;
            for _ in 0..sz {
                let doc = self.read_value()?;
                self.invoke(callback, &doc)?;
            }
        } else {
            // Unexpected, decode normally to stay correct.
            self.reader.pos -= 1;
            let _ = self.read_value()?;
        }

        let empty = unsafe { Obj::from_owned(ffi::PyList_New(0))? };
        self.dict_set_str(&dict, c"docs", &empty)?;
        Ok(dict)
    }

    /// Call `callback(doc)`, dropping the result. `callback` is a borrowed
    /// reference to a Python callable.
    #[inline]
    fn invoke(&self, callback: *mut ffi::PyObject, doc: &Obj) -> Result<()> {
        let res = unsafe {
            ffi::PyObject_CallFunctionObjArgs(
                callback,
                doc.as_ptr(),
                std::ptr::null_mut::<ffi::PyObject>(),
            )
        };
        if res.is_null() {
            Err(DecodeError::PyErr)
        } else {
            unsafe { ffi::Py_DECREF(res) };
            Ok(())
        }
    }
}

/// Borrow a `&str` view of an `Obj` iff it is an exact `str`. Returns `None`
/// otherwise (or on any encoding error), which the callers treat as "not the
/// key we're looking for".
fn obj_as_str(o: &Obj) -> Option<&str> {
    unsafe {
        if ffi::PyUnicode_Check(o.as_ptr()) == 0 {
            return None;
        }
        let mut size: ffi::Py_ssize_t = 0;
        let ptr = ffi::PyUnicode_AsUTF8AndSize(o.as_ptr(), &mut size);
        if ptr.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        let bytes = std::slice::from_raw_parts(ptr as *const u8, size as usize);
        std::str::from_utf8(bytes).ok()
    }
}

enum Slot {
    End,
    Child(Obj),
    Value(Obj),
}

/// Which document sequence follows the envelope, as reported by
/// [`Decoder::stream_envelope_to_docs`].
pub(crate) enum DocsPhase {
    /// A fixed-length `ARR` of `len` documents.
    Arr(usize),
    /// A variable-length `ITERATOR` of documents, terminated by `END`.
    Iter,
    /// No recognisable document sequence (nothing to stream).
    None,
}

// -- free scalar constructors (see note at the `Decoder` impl) --------------

#[inline]
fn py_none() -> Obj {
    unsafe { Obj::from_borrowed(ffi::Py_None()) }
}

#[inline]
fn py_bool(b: bool) -> Obj {
    let ptr = if b {
        unsafe { ffi::Py_True() }
    } else {
        unsafe { ffi::Py_False() }
    };
    unsafe { Obj::from_borrowed(ptr) }
}

#[inline]
fn py_long(v: i64) -> Result<Obj> {
    unsafe { Obj::from_owned(ffi::PyLong_FromLongLong(v)) }
}

#[inline]
fn py_float(v: f64) -> Result<Obj> {
    unsafe { Obj::from_owned(ffi::PyFloat_FromDouble(v)) }
}

#[inline]
fn py_str(s: &str) -> Result<Obj> {
    unsafe {
        Obj::from_owned(ffi::PyUnicode_FromStringAndSize(
            s.as_ptr() as *const std::os::raw::c_char,
            s.len() as ffi::Py_ssize_t,
        ))
    }
}

#[inline]
fn py_bytes(b: &[u8]) -> Result<Obj> {
    unsafe {
        Obj::from_owned(ffi::PyBytes_FromStringAndSize(
            b.as_ptr() as *const std::os::raw::c_char,
            b.len() as ffi::Py_ssize_t,
        ))
    }
}

/// Decode a complete javabin (protocol version 2) message directly into
/// native Python objects via `pyo3::ffi`.
pub fn decode<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let mut decoder = Decoder::new(py, data);

    let version = decoder.reader.read_u8().map_err(|e| err_to_py(py, e))?;
    if version != crate::reader::EXPECTED_VERSION {
        return Err(err_to_py(
            py,
            DecodeError::InvalidVersion { found: version },
        ));
    }

    let value = decoder.read_value().map_err(|e| err_to_py(py, e))?;

    if decoder.reader.pos != decoder.reader.data.len() {
        return Err(err_to_py(
            py,
            DecodeError::TrailingData {
                remaining: decoder.reader.data.len() - decoder.reader.pos,
            },
        ));
    }

    // Transfer ownership of the root object into a Bound<PyAny>.
    let ptr = value.into_ptr();
    Ok(unsafe { Bound::from_owned_ptr(py, ptr) })
}

/// Streaming decode: like [`decode`], but each document in the result's
/// `docs` sequence is passed to `callback` (a Python callable) and then
/// dropped, instead of being collected. Returns the response *envelope* (the
/// same structure as [`decode`] but with an empty `docs` list), so metadata
/// like `numFound` is still available.
///
/// This keeps peak memory at ~one document rather than the whole result set,
/// which matters for large `/export` / `/stream` responses.
pub fn decode_stream<'py>(
    py: Python<'py>,
    data: &[u8],
    callback: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut decoder = Decoder::new(py, data);

    let version = decoder.reader.read_u8().map_err(|e| err_to_py(py, e))?;
    if version != crate::reader::EXPECTED_VERSION {
        return Err(err_to_py(
            py,
            DecodeError::InvalidVersion { found: version },
        ));
    }

    let envelope = decoder
        .stream(callback.as_ptr())
        .map_err(|e| err_to_py(py, e))?;

    if decoder.reader.pos != decoder.reader.data.len() {
        return Err(err_to_py(
            py,
            DecodeError::TrailingData {
                remaining: decoder.reader.data.len() - decoder.reader.pos,
            },
        ));
    }

    let ptr = envelope.into_ptr();
    Ok(unsafe { Bound::from_owned_ptr(py, ptr) })
}

/// Map a [`DecodeError`] to a Python exception. [`DecodeError::PyErr`] means
/// a Python exception is *already* set (a C API call failed), so we just
/// fetch it; everything else becomes a fresh `ValueError`.
fn err_to_py(py: Python<'_>, err: DecodeError) -> PyErr {
    match err {
        DecodeError::PyErr => PyErr::fetch(py),
        other => pyo3::exceptions::PyValueError::new_err(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyAnyMethods, PyBytes, PyList};

    const V: u8 = crate::reader::EXPECTED_VERSION;

    fn wv(mut body: Vec<u8>) -> Vec<u8> {
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

    /// Every fixture must decode identically via the safe PyO3 path and the
    /// ffi fast path — this is the core correctness guarantee for the unsafe
    /// code, checked by Python-level `==`.
    fn assert_fast_eq_safe(body_with_version: &[u8]) {
        Python::attach(|py| {
            let fast = decode(py, body_with_version).unwrap();
            let safe = crate::py_decoder::decode(py, body_with_version).unwrap();
            assert!(fast.eq(&safe).unwrap(), "fast {fast:?} != safe {safe:?}");
        });
    }

    #[test]
    fn scalars_match_safe() {
        assert_fast_eq_safe(&wv(vec![tag::NULL]));
        assert_fast_eq_safe(&wv(vec![tag::BOOL_TRUE]));
        assert_fast_eq_safe(&wv(vec![tag::BOOL_FALSE]));
        assert_fast_eq_safe(&wv(vec![tag::SINT | 5]));

        let mut b = vec![tag::INT];
        b.extend_from_slice(&(-42i32).to_be_bytes());
        assert_fast_eq_safe(&wv(b));

        let mut b = vec![tag::LONG];
        b.extend_from_slice(&1_870_516_012_295_651_331i64.to_be_bytes());
        assert_fast_eq_safe(&wv(b));

        let mut b = vec![tag::DOUBLE];
        b.extend_from_slice(&2.25f64.to_be_bytes());
        assert_fast_eq_safe(&wv(b));

        let mut b = vec![tag::FLOAT];
        b.extend_from_slice(&1.5f32.to_be_bytes());
        assert_fast_eq_safe(&wv(b));

        let mut b = vec![tag::DATE];
        b.extend_from_slice(&1_700_000_000_000i64.to_be_bytes());
        assert_fast_eq_safe(&wv(b));
    }

    #[test]
    fn string_and_bytes_match_safe() {
        // short string (fits in the inline 5-bit size field)
        let s = "hello wörld 😀";
        let bytes = s.as_bytes();
        assert!(bytes.len() < 0x1f);
        let mut b = vec![tag::STR | bytes.len() as u8];
        b.extend_from_slice(bytes);
        assert_fast_eq_safe(&wv(b));

        // long string (extended size: 0x1f + vint)
        let long = "x".repeat(100);
        let mut b = vec![tag::STR | 0x1f];
        write_vint(&mut b, (long.len() - 0x1f) as u32);
        b.extend_from_slice(long.as_bytes());
        assert_fast_eq_safe(&wv(b));

        let mut b = vec![tag::BYTEARR];
        write_vint(&mut b, 3);
        b.extend_from_slice(&[1, 2, 3]);
        assert_fast_eq_safe(&wv(b));

        // empty string
        assert_fast_eq_safe(&wv(vec![tag::STR]));
    }

    #[test]
    fn containers_match_safe() {
        // array [1, 2]
        assert_fast_eq_safe(&wv(vec![tag::ARR | 2, tag::SINT | 1, tag::SINT | 2]));
        // empty array
        assert_fast_eq_safe(&wv(vec![tag::ARR]));

        // NamedList {"a": 1}
        let mut b = vec![tag::NAMED_LST | 1];
        b.push(tag::STR | 1);
        b.push(b'a');
        b.push(tag::SINT | 1);
        assert_fast_eq_safe(&wv(b));

        // iterator [1, 2] END
        assert_fast_eq_safe(&wv(vec![
            tag::ITERATOR,
            tag::SINT | 1,
            tag::SINT | 2,
            tag::END,
        ]));
    }

    #[test]
    fn generic_map_non_string_key_matches_safe() {
        // MAP{1: "a"} -> list of [key, value] pairs
        let mut b = vec![tag::MAP];
        write_vint(&mut b, 1);
        b.push(tag::SINT | 1);
        b.push(tag::STR | 1);
        b.push(b'a');
        assert_fast_eq_safe(&wv(b));
    }

    #[test]
    fn solr_document_with_child_matches_safe() {
        let mut b = vec![tag::SOLRDOC, tag::ORDERED_MAP | 2];
        b.push(tag::STR | 2);
        b.extend_from_slice(b"id");
        b.push(tag::STR | 1);
        b.push(b'1');
        b.push(tag::SOLRDOC);
        b.push(tag::ORDERED_MAP | 1);
        b.push(tag::STR | 2);
        b.extend_from_slice(b"id");
        b.push(tag::STR | 1);
        b.push(b'2');
        assert_fast_eq_safe(&wv(b));
    }

    #[test]
    fn solr_document_list_matches_safe() {
        let mut b = vec![tag::SOLRDOCLST];
        b.push(tag::ARR | 4);
        b.push(tag::SLONG | 1);
        b.push(tag::SLONG);
        b.push(tag::NULL);
        b.push(tag::BOOL_TRUE);
        b.push(tag::ARR | 1);
        b.push(tag::SOLRDOC);
        b.push(tag::ORDERED_MAP | 1);
        b.push(tag::STR | 2);
        b.extend_from_slice(b"id");
        b.push(tag::STR | 1);
        b.push(b'1');
        assert_fast_eq_safe(&wv(b));
    }

    #[test]
    fn interns_repeated_field_names() {
        // Two docs each with EXTERN_STRING key "foo"; the two decoded keys
        // must be the *same* interned Python object.
        let mut b = vec![tag::ARR | 2];
        b.push(tag::SOLRDOC);
        b.push(tag::ORDERED_MAP | 1);
        b.push(tag::EXTERN_STRING);
        b.push(tag::STR | 3);
        b.extend_from_slice(b"foo");
        b.push(tag::SINT | 1);
        b.push(tag::SOLRDOC);
        b.push(tag::ORDERED_MAP | 1);
        b.push(tag::EXTERN_STRING | 1);
        b.push(tag::SINT | 2);

        Python::attach(|py| {
            let v = decode(py, &wv(b)).unwrap();
            let list = v.cast::<PyList>().unwrap();
            let k0 = list
                .get_item(0)
                .unwrap()
                .call_method0("keys")
                .unwrap()
                .call_method1("__iter__", ())
                .unwrap()
                .call_method0("__next__")
                .unwrap();
            let k1 = list
                .get_item(1)
                .unwrap()
                .call_method0("keys")
                .unwrap()
                .call_method1("__iter__", ())
                .unwrap()
                .call_method0("__next__")
                .unwrap();
            assert!(k0.is(&k1), "expected same interned string object");
        });
    }

    #[test]
    fn primitive_byte_array_is_bytes() {
        let mut b = vec![tag::PRIMITIVE_ARR, tag::BYTE];
        write_vint(&mut b, 3);
        b.extend_from_slice(&[9, 8, 7]);
        Python::attach(|py| {
            let v = decode(py, &wv(b)).unwrap();
            let bytes = v.cast::<PyBytes>().unwrap();
            assert_eq!(bytes.as_bytes(), &[9, 8, 7]);
        });
    }

    #[test]
    fn errors_match() {
        Python::attach(|py| {
            assert!(decode(py, &[1u8, tag::NULL]).is_err()); // wrong version
            assert!(decode(py, &[]).is_err()); // empty
            assert!(decode(py, &[V, tag::INT, 0, 0]).is_err()); // truncated int
            assert!(decode(py, &[V, tag::NULL, 0xFF]).is_err()); // trailing data
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
            assert!(decode(py, &wv(body)).is_err());
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
            let err = decode(py, &wv(body)).unwrap_err();
            assert!(err.to_string().contains("nesting"), "{err}");
        });
    }

    #[test]
    fn decodes_nesting_within_the_depth_limit() {
        Python::attach(|py| {
            let depth = 100;
            let mut body = vec![tag::ARR | 1; depth];
            body.push(tag::NULL);
            let value = decode(py, &wv(body)).unwrap();

            let mut v = value;
            for _ in 0..depth {
                let list = v.cast::<PyList>().unwrap();
                assert_eq!(list.len(), 1);
                v = list.get_item(0).unwrap();
            }
            assert!(v.is_none());
        });
    }

    /// Build a `/select`-style message: NamedList{"response": SOLRDOCLST{
    /// header[nf,0,null,true], docs ARR[ n x SOLRDOC{"i": <k>} ]}}.
    fn select_message(n: u8) -> Vec<u8> {
        let mut b = vec![tag::NAMED_LST | 1];
        b.push(tag::STR | 8);
        b.extend_from_slice(b"response");
        b.push(tag::SOLRDOCLST);
        // header array
        b.push(tag::ARR | 4);
        b.push(tag::SLONG | n); // numFound
        b.push(tag::SLONG); // start = 0
        b.push(tag::NULL); // maxScore
        b.push(tag::BOOL_TRUE); // numFoundExact
        // docs array
        b.push(tag::ARR | n);
        for k in 0..n {
            b.push(tag::SOLRDOC);
            b.push(tag::ORDERED_MAP | 1);
            b.push(tag::STR | 1);
            b.push(b'i');
            b.push(tag::SINT | (k & 0x0f));
        }
        wv(b)
    }

    /// Build a `/stream`-style message: NamedList{"result-set":
    /// MAP_ENTRY_ITER{"docs": ITERATOR[ n x SOLRDOC{"i": <k>} ] END} END}.
    fn stream_message(n: u8) -> Vec<u8> {
        let mut b = vec![tag::NAMED_LST | 1];
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
            b.push(tag::SINT | (k & 0x0f));
        }
        b.push(tag::END); // end iterator
        b.push(tag::END); // end map_entry_iter
        wv(b)
    }

    /// Collect streamed docs into a Python list via a closure callback, and
    /// return (envelope, collected_docs) for assertions.
    fn run_stream<'py>(py: Python<'py>, msg: &[u8]) -> (Bound<'py, PyAny>, Bound<'py, PyList>) {
        let collected = PyList::empty(py);
        let cb = collected.getattr("append").unwrap().into_any();
        let env = decode_stream(py, msg, &cb).unwrap();
        (env, collected)
    }

    #[test]
    fn stream_select_yields_same_docs_as_decode() {
        Python::attach(|py| {
            let msg = select_message(5);
            let full = decode(py, &msg).unwrap();
            let full_docs = full.get_item("response").unwrap().get_item("docs").unwrap();

            let (env, docs) = run_stream(py, &msg);
            // streamed docs equal the docs from a full decode
            assert!(docs.as_any().eq(&full_docs).unwrap());
            // envelope keeps numFound but has empty docs
            let resp = env.get_item("response").unwrap();
            assert_eq!(
                resp.get_item("numFound").unwrap().extract::<i64>().unwrap(),
                5
            );
            assert_eq!(resp.get_item("docs").unwrap().len().unwrap(), 0);
        });
    }

    #[test]
    fn stream_result_set_yields_same_docs_as_decode() {
        Python::attach(|py| {
            let msg = stream_message(4);
            let full = decode(py, &msg).unwrap();
            let full_docs = full
                .get_item("result-set")
                .unwrap()
                .get_item("docs")
                .unwrap();

            let (env, docs) = run_stream(py, &msg);
            assert!(docs.as_any().eq(&full_docs).unwrap());
            let resp = env.get_item("result-set").unwrap();
            assert_eq!(resp.get_item("docs").unwrap().len().unwrap(), 0);
        });
    }
}
