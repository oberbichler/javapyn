"""
Fetch a real ``wt=javabin`` sample response from a Solr collection and save
the raw bytes to disk, for use as a fixture in decoder tests / manual
verification against a live Solr instance.

Usage
-----
    uv run --with httpx python scripts/fetch_sample.py \\
        --base-url https://solr.example.com/solr \\
        --collection solr_movies \\
        --query "*:*" \\
        --fields movie_id,title,rating,genres \\
        --rows 3 \\
        --out tests/fixtures/solr_movies_sample

This writes both ``solr_movies_sample.bin`` (raw javabin bytes, wt=javabin) and
``solr_movies_sample.json`` (the same query with wt=json, for comparison).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def fetch(
    base_url: str,
    collection: str,
    query: str,
    fields: list[str] | None,
    rows: int,
    wt: str,
    auth: tuple[str, str] | None,
) -> bytes:
    params: dict[str, str] = {
        "q": query,
        "rows": str(rows),
        "wt": wt,
        "version": "2",
    }
    if fields:
        params["fl"] = ",".join(fields)

    url = f"{base_url.rstrip('/')}/{collection}/select"
    response = httpx.get(url, params=params, auth=auth, timeout=30.0)
    response.raise_for_status()
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="Solr base URL, e.g. https://host/solr"
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", default="*:*")
    parser.add_argument("--fields", default=None, help="Comma-separated field list")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--out", required=True, help="Output path prefix (without extension)"
    )
    args = parser.parse_args()

    fields = args.fields.split(",") if args.fields else None
    auth = (args.user, args.password) if args.user else None

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    javabin_bytes = fetch(
        args.base_url, args.collection, args.query, fields, args.rows, "javabin", auth
    )
    out_prefix.with_suffix(".bin").write_bytes(javabin_bytes)
    print(f"Wrote {len(javabin_bytes)} bytes to {out_prefix.with_suffix('.bin')}")

    json_bytes = fetch(
        args.base_url, args.collection, args.query, fields, args.rows, "json", auth
    )
    parsed = json.loads(json_bytes)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False)
    )
    print(f"Wrote reference JSON to {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
