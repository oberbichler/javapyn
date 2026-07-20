import json
import time
from pathlib import Path
from statistics import median

import orjson

import javapyn as javabin

FIXTURES = Path("tests/fixtures")


def bench(fn, data, n=50):
    fn(data)  # warm-up
    times = []
    for _ in range(n):
        s = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - s)
    return median(times) * 1000


for name in [
    "solr_movies_sample",
    "solr_movies_all_fields",
    "solr_movies_deterministic",
]:
    data = (FIXTURES / f"{name}.bin").read_bytes()
    ref_json = (FIXTURES / f"{name}.json").read_bytes()

    def combo(d):
        return orjson.loads(javabin.deserialize_json(d))

    print(f"{name}  (javabin {len(data)}B / json {len(ref_json)}B)")
    print(f"  json.loads                    : {bench(json.loads, ref_json):.4f} ms")
    print(f"  orjson.loads                  : {bench(orjson.loads, ref_json):.4f} ms")
    print(
        f"  javabin.deserialize            : {bench(javabin.deserialize, data):.4f} ms"
    )
    print(
        f"  javabin.deserialize_json       : {bench(javabin.deserialize_json, data):.4f} ms"
    )
    print(f"  deserialize_json + orjson.loads: {bench(combo, data):.4f} ms")
    print()
