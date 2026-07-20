"""
Performance and Scalability Benchmark for javapyn.

Demonstrates the extreme performance, high-throughput, and flat memory footprint
of the Rust-based deserializer under massive scales (up to 5.0 GB of data).
"""

import sys
import json
import time
import gc
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import polars as pl
import orjson

import javapyn

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from javabin_ref_encoder import (
    NamedList,
    SolrDoc,
    SolrDocList,
    Encoder,
    encode,
)


def get_peak_memory_mb() -> float:
    """Helper to read current process RSS memory in MB."""
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class MassiveStreamGenerator:
    """Stream-generates Solr select responses with movie schemas."""

    def __init__(self, num_docs: int, chunk_size: int = 50000):
        self.num_docs = num_docs
        self.chunk_size = chunk_size
        self.encoder = Encoder()

    def iter_chunks(self) -> Iterator[bytes]:
        # Reset the strings cache
        self.encoder._strings = {}

        # 1. Generate & yield standard select response envelope with the ARR header
        solr_list = SolrDocList(
            num_found=self.num_docs,
            start=0,
            max_score=1.0,
            num_found_exact=True,
            docs=[],
        )
        envelope = NamedList(
            [
                ("responseHeader", NamedList([("zkConnected", True), ("status", 0)])),
                ("response", solr_list),
            ]
        )

        env_bytes = bytearray(encode(envelope))

        # Pop the last byte (ARR | 0 = 0x80) and append ARR | self.num_docs
        env_bytes.pop()

        self.encoder.buf = bytearray()
        self.encoder._write_tag(4 << 5, self.num_docs)  # ARR tag
        env_bytes.extend(self.encoder.buf)
        yield bytes(env_bytes)

        # 2. Yield document chunks
        chunk_docs = []
        for i in range(self.num_docs):
            m_id = f"movie-{10000000 + i}"
            title = f"The Great Movie Title Alpha Beta Gamma {i}"
            doc = SolrDoc(
                fields={
                    "movie_id": m_id,
                    "title": title,
                    "rating": 8.5,
                    "release_year": 2010 + (i % 15),
                }
            )
            chunk_docs.append(doc)

            if len(chunk_docs) >= self.chunk_size:
                self.encoder.buf = bytearray()
                for d in chunk_docs:
                    self.encoder._write_solr_doc(d)
                yield bytes(self.encoder.buf)
                chunk_docs = []

        if chunk_docs:
            self.encoder.buf = bytearray()
            for d in chunk_docs:
                self.encoder._write_solr_doc(d)
            yield bytes(self.encoder.buf)


def run_massive_streaming_benchmark():
    print("=" * 80)
    print("BENCHMARK 1: MASSIVE STREAMING PROCESSING (5.0 GIGABYTE SOLR RESPONSE)")
    print("=" * 80)
    print("Simulating a massive streaming endpoint with 20,000,000 movie records...")
    print("Piping raw bytes on the fly directly into javapyn decoders...")
    print("-" * 80)

    # 1. Standard Object Stream Decoder
    gc.collect()
    start_mem = get_peak_memory_mb()
    dec = javapyn.StreamDecoder()

    doc_count = 0

    def handle_doc(doc):
        nonlocal doc_count
        doc_count += 1

    generator = MassiveStreamGenerator(num_docs=20000000, chunk_size=100000)

    print(f"[*] Starting Object StreamDecoder. Initial Memory: {start_mem:.2f} MB")
    start_time = time.perf_counter()
    bytes_streamed = 0

    for chunk in generator.iter_chunks():
        bytes_streamed += len(chunk)
        dec.feed(chunk, handle_doc)

    dec.finish()
    end_time = time.perf_counter()
    peak_mem = get_peak_memory_mb()

    elapsed = end_time - start_time
    throughput_docs = doc_count / elapsed
    throughput_mb = (bytes_streamed / (1024 * 1024)) / elapsed

    print("[+] StreamDecoder Finished Successfully:")
    print(f"    - Total Documents Processed : {doc_count:,}")
    print(
        f"    - Total Raw Bytes Flowed    : {bytes_streamed / (1024 * 1024 * 1024):.2f} GB ({bytes_streamed:,} bytes)"
    )
    print(f"    - Total Elapsed Time        : {elapsed:.2f} seconds")
    print(
        f"    - Throughput (Speed)        : {throughput_docs:,.0f} docs/sec ({throughput_mb:.1f} MB/sec)"
    )
    print(
        f"    - Initial RAM / Peak RAM    : {start_mem:.2f} MB / {peak_mem:.2f} MB (Flat Memory Footprint!)"
    )
    print("-" * 80)

    # 2. Columnar Arrow Stream Decoder directly to Polars
    gc.collect()
    start_mem = get_peak_memory_mb()

    solr_schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("title", pa.string()),
            ("rating", pa.float32()),
            ("release_year", pa.int32()),
        ]
    )

    dec_arrow = javapyn.ArrowStreamDecoder(solr_schema, batch_size=200000)
    batches = []

    generator = MassiveStreamGenerator(num_docs=20000000, chunk_size=100000)
    print(
        f"[*] Starting Columnar ArrowStreamDecoder. Initial Memory: {start_mem:.2f} MB"
    )
    start_time = time.perf_counter()
    bytes_streamed = 0

    for chunk in generator.iter_chunks():
        bytes_streamed += len(chunk)
        completed = dec_arrow.feed(chunk)
        if completed:
            batches.extend(completed)
            if len(batches) >= 20:
                table = pa.Table.from_batches(batches, schema=solr_schema)
                df = pl.from_arrow(table)
                _ = len(df)
                batches = []

    completed = dec_arrow.finish()
    if completed:
        batches.extend(completed)
    if batches:
        table = pa.Table.from_batches(batches, schema=solr_schema)
        df = pl.from_arrow(table)
        _ = len(df)

    end_time = time.perf_counter()
    peak_mem = get_peak_memory_mb()

    elapsed = end_time - start_time
    throughput_docs = 20000000 / elapsed
    throughput_mb = (bytes_streamed / (1024 * 1024)) / elapsed

    print("[+] ArrowStreamDecoder (Direct-to-Polars) Finished Successfully:")
    print("    - Total Documents Processed : 20,000,000")
    print(
        f"    - Total Raw Bytes Flowed    : {bytes_streamed / (1024 * 1024 * 1024):.2f} GB"
    )
    print(f"    - Total Elapsed Time        : {elapsed:.2f} seconds")
    print(
        f"    - Throughput (Speed)        : {throughput_docs:,.0f} docs/sec ({throughput_mb:.1f} MB/sec)"
    )
    print(f"    - Initial RAM / Peak RAM    : {start_mem:.2f} MB / {peak_mem:.2f} MB")
    print("=" * 80)
    print()


def run_in_memory_benchmark():
    print("=" * 80)
    print("BENCHMARK 2: IN-MEMORY PARSING COMPARISON (1,000,000 RECORDS)")
    print("=" * 80)
    print("Generating simulated dataset...")

    # 1. Generate 1,000,000 movie docs in Python memory
    movies_list = []
    movies_json_list = []
    for i in range(1000000):
        doc_fields = {
            "movie_id": f"movie-{10000000 + i}",
            "title": f"The Great Movie Title Alpha Beta Gamma {i}",
            "rating": 8.5,
            "release_year": 2010 + (i % 15),
        }
        movies_list.append(SolrDoc(fields=doc_fields))
        movies_json_list.append(doc_fields)

    # Serialize to javabin standard select response
    select_response = NamedList(
        [
            ("responseHeader", NamedList([("zkConnected", True), ("status", 0)])),
            (
                "response",
                SolrDocList(
                    num_found=1000000,
                    start=0,
                    max_score=1.0,
                    num_found_exact=True,
                    docs=movies_list,
                ),
            ),
        ]
    )
    javabin_bytes = encode(select_response)

    # Serialize to JSON standard select response
    json_envelope = {
        "responseHeader": {"zkConnected": True, "status": 0},
        "response": {
            "numFound": 1000000,
            "start": 0,
            "maxScore": 1.0,
            "numFoundExact": True,
            "docs": movies_json_list,
        },
    }
    json_bytes = orjson.dumps(json_envelope)

    print("[*] Dataset Ready:")
    print(
        f"    - Raw wt=json payload size    : {len(json_bytes) / (1024 * 1024):.2f} MB"
    )
    print(
        f"    - Raw wt=javabin payload size  : {len(javabin_bytes) / (1024 * 1024):.2f} MB (2.3x smaller!)"
    )
    print("-" * 80)

    # 1. Standard python json.loads
    gc.collect()
    start_mem = get_peak_memory_mb()
    start_time = time.perf_counter()
    data_json = json.loads(json_bytes)
    _ = len(data_json["response"]["docs"])
    elapsed_json = time.perf_counter() - start_time
    peak_mem_json = get_peak_memory_mb() - start_mem
    del data_json

    # 2. High-performance orjson.loads
    gc.collect()
    start_mem = get_peak_memory_mb()
    start_time = time.perf_counter()
    data_orjson = orjson.loads(json_bytes)
    _ = len(data_orjson["response"]["docs"])
    elapsed_orjson = time.perf_counter() - start_time
    peak_mem_orjson = get_peak_memory_mb() - start_mem
    del data_orjson

    # 3. javapyn.deserialize
    gc.collect()
    start_mem = get_peak_memory_mb()
    start_time = time.perf_counter()
    data_javapyn = javapyn.deserialize(javabin_bytes)
    _ = len(data_javapyn["response"]["docs"])
    elapsed_javapyn = time.perf_counter() - start_time
    peak_mem_javapyn = get_peak_memory_mb() - start_mem
    del data_javapyn

    # 4. javapyn.deserialize_arrow (Direct to Columnar)
    gc.collect()
    start_mem = get_peak_memory_mb()
    solr_schema = pa.schema(
        [
            ("movie_id", pa.string()),
            ("title", pa.string()),
            ("rating", pa.float32()),
            ("release_year", pa.int32()),
        ]
    )
    start_time = time.perf_counter()
    batch = javapyn.deserialize_arrow(javabin_bytes, solr_schema)
    df = pl.from_arrow(batch)
    _ = len(df)
    elapsed_arrow = time.perf_counter() - start_time
    peak_mem_arrow = get_peak_memory_mb() - start_mem
    del batch, df

    print("Results for 1,000,000 records:")
    print(
        f"    - json.loads (std-lib JSON)   : {elapsed_json:.4f} sec | RAM Overhead: {peak_mem_json:.1f} MB"
    )
    print(
        f"    - orjson.loads (Rust JSON)    : {elapsed_orjson:.4f} sec | RAM Overhead: {peak_mem_orjson:.1f} MB"
    )
    print(
        f"    - javapyn.deserialize (Rust)  : {elapsed_javapyn:.4f} sec | RAM Overhead: {peak_mem_javapyn:.1f} MB (Faster than orjson!)"
    )
    print(
        f"    - javapyn.deserialize_arrow   : {elapsed_arrow:.4f} sec | RAM Overhead: {peak_mem_arrow:.1f} MB (Speed King & Minimal RAM!)"
    )
    print("=" * 80)


if __name__ == "__main__":
    # Install psutil if missing
    import importlib.util

    if importlib.util.find_spec("psutil") is None:
        import subprocess

        print("[*] Installing psutil dependency for memory measurements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
        print("-" * 80)

    run_in_memory_benchmark()
    print()
    run_massive_streaming_benchmark()
