"""Fast Rust-based deserializer for Apache Solr's ``javabin`` (protocol v2) format."""

from ._core import (
    ArrowStreamDecoder,
    StreamDecoder,
    deserialize,
    deserialize_arrow,
    deserialize_json,
    deserialize_stream,
)

__all__ = [
    "ArrowStreamDecoder",
    "StreamDecoder",
    "deserialize",
    "deserialize_arrow",
    "deserialize_json",
    "deserialize_stream",
]
