//! Intermediate representation of a decoded javabin value.
//!
//! This mirrors the object graph produced by Solr's
//! `org.apache.solr.common.util.JavaBinCodec`, without depending on
//! Python/PyO3 so it can be unit tested in plain Rust.

use std::fmt;

/// A single decoded javabin value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// Milliseconds since the Unix epoch (javabin `DATE` tag).
    Date(i64),
    Str(String),
    /// Raw bytes (javabin `BYTEARR` tag).
    Bytes(Vec<u8>),
    /// A javabin `ARR` (ordered list of values).
    List(Vec<Value>),
    /// A generic javabin `MAP` (arbitrary key/value types).
    Map(Vec<(Value, Value)>),
    /// A javabin `NAMED_LST` / `ORDERED_MAP` / `SIMPLE_MAP` (string-keyed,
    /// order-preserving, keys may repeat).
    ///
    /// An entry name is `None` when it was written as the `NULL` tag. That is
    /// legal on the wire -- `JavaBinCodec.readNamedList` reads the name with
    /// `(String) readVal(dis)`, which yields null -- and Solr does it in
    /// practice for the `facet.missing` bucket of a field facet.
    NamedList(Vec<(Option<String>, Value)>),
    /// A javabin `SOLRDOC`. Child documents (added via Solr's nested/child
    /// document feature) are kept separate from regular fields, mirroring
    /// `SolrDocument.getChildDocuments()`.
    SolrDocument {
        fields: Vec<(String, Value)>,
        children: Vec<Value>,
    },
    /// A javabin `SOLRDOCLST`: header info plus the contained documents.
    SolrDocumentList {
        num_found: i64,
        start: i64,
        max_score: Option<f64>,
        num_found_exact: Option<bool>,
        docs: Vec<Value>,
    },
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Value {
    /// Convert this value into a [`serde_json::Value`].
    ///
    /// `SolrDocument` becomes a JSON object of its fields. `SolrDocumentList`
    /// becomes a JSON object shaped like Solr's own `wt=json` response
    /// (`numFound`, `start`, `maxScore`, `docs`) so that results can be
    /// compared directly against the standard JSON API.
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::Value as J;

        match self {
            Value::Null => J::Null,
            Value::Bool(b) => J::Bool(*b),
            Value::Byte(v) => J::from(*v),
            Value::Short(v) => J::from(*v),
            Value::Int(v) => J::from(*v),
            Value::Long(v) => J::from(*v),
            Value::Float(v) => json_from_f64(*v as f64),
            Value::Double(v) => json_from_f64(*v),
            Value::Date(v) => J::from(*v),
            Value::Str(s) => J::String(s.clone()),
            Value::Bytes(b) => J::Array(b.iter().map(|byte| J::from(*byte)).collect()),
            Value::List(items) => J::Array(items.iter().map(Value::to_json).collect()),
            Value::Map(entries) => {
                // JSON objects require string keys; fall back to an array of
                // [key, value] pairs when a key is not a string.
                let all_string_keys = entries.iter().all(|(k, _)| matches!(k, Value::Str(_)));

                if all_string_keys {
                    let mut map = serde_json::Map::new();
                    for (k, v) in entries {
                        if let Value::Str(key) = k {
                            map.insert(key.clone(), v.to_json());
                        }
                    }
                    J::Object(map)
                } else {
                    J::Array(
                        entries
                            .iter()
                            .map(|(k, v)| J::Array(vec![k.to_json(), v.to_json()]))
                            .collect(),
                    )
                }
            }
            Value::NamedList(entries) => {
                let mut map = serde_json::Map::new();
                for (k, v) in entries {
                    // JSON has no null keys. Solr's own JSON writer renders a
                    // null map key as the empty string, so match that.
                    map.insert(k.clone().unwrap_or_default(), v.to_json());
                }
                J::Object(map)
            }
            Value::SolrDocument { fields, children } => {
                let mut map = serde_json::Map::new();
                for (k, v) in fields {
                    map.insert(k.clone(), v.to_json());
                }
                if !children.is_empty() {
                    map.insert(
                        "_childDocuments_".into(),
                        J::Array(children.iter().map(Value::to_json).collect()),
                    );
                }
                J::Object(map)
            }
            Value::SolrDocumentList {
                num_found,
                start,
                max_score,
                num_found_exact,
                docs,
            } => {
                let mut map = serde_json::Map::new();
                map.insert("numFound".into(), J::from(*num_found));
                map.insert("start".into(), J::from(*start));
                map.insert(
                    "maxScore".into(),
                    max_score.map(json_from_f64).unwrap_or(J::Null),
                );
                map.insert(
                    "numFoundExact".into(),
                    num_found_exact.map(J::Bool).unwrap_or(J::Null),
                );
                map.insert(
                    "docs".into(),
                    J::Array(docs.iter().map(Value::to_json).collect()),
                );
                J::Object(map)
            }
        }
    }
}

/// Convert a decoded float to JSON.
///
/// JSON has no literal for the non-finite values, so a `DOUBLE`/`FLOAT` tag
/// carrying one has to be rendered some other way. Solr's own JSON writer emits
/// the strings `"Infinity"`, `"-Infinity"` and `"NaN"`, so match that: it keeps
/// `deserialize_json` a drop-in for `wt=json`, and it is not lossy the way
/// serde_json's default (null) would be. These reach ordinary responses through
/// the stats component over large doubles and through overflowing function
/// queries.
fn json_from_f64(v: f64) -> serde_json::Value {
    match serde_json::Number::from_f64(v) {
        Some(n) => serde_json::Value::Number(n),
        None if v.is_nan() => serde_json::Value::String("NaN".into()),
        None if v > 0.0 => serde_json::Value::String("Infinity".into()),
        None => serde_json::Value::String("-Infinity".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_floats_become_solr_style_strings() {
        // JSON has no literal for these; Solr's own writer emits strings, and
        // serde_json's default of null would silently lose the value.
        let cases = [
            (f64::INFINITY, r#""Infinity""#),
            (f64::NEG_INFINITY, r#""-Infinity""#),
            (f64::NAN, r#""NaN""#),
        ];
        for (input, expected) in cases {
            assert_eq!(Value::Double(input).to_json().to_string(), expected);
        }
        assert_eq!(
            Value::Float(f32::INFINITY).to_json().to_string(),
            r#""Infinity""#
        );
    }

    #[test]
    fn finite_floats_stay_numbers() {
        assert_eq!(Value::Double(1.5).to_json().to_string(), "1.5");
        assert_eq!(Value::Double(-0.0).to_json().to_string(), "-0.0");
        assert_eq!(
            Value::Double(f64::MAX).to_json(),
            serde_json::json!(f64::MAX)
        );
    }
}

/// Count the total number of nodes in a `Value` tree (for benchmarking).
pub fn count_nodes(v: &Value) -> usize {
    match v {
        Value::List(items) => 1 + items.iter().map(count_nodes).sum::<usize>(),
        Value::Map(entries) => {
            1 + entries
                .iter()
                .map(|(k, v)| count_nodes(k) + count_nodes(v))
                .sum::<usize>()
        }
        Value::NamedList(entries) => {
            1 + entries
                .iter()
                .map(|(_, v)| 1 + count_nodes(v))
                .sum::<usize>()
        }
        Value::SolrDocument { fields, children } => {
            1 + fields
                .iter()
                .map(|(_, v)| 1 + count_nodes(v))
                .sum::<usize>()
                + children.iter().map(count_nodes).sum::<usize>()
        }
        Value::SolrDocumentList { docs, .. } => 1 + docs.iter().map(count_nodes).sum::<usize>(),
        _ => 1,
    }
}
