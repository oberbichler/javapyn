"""
Helpers for asserting that a decoded ``wt=javabin`` response matches the
``wt=json`` response for the same query.

Shared by the fixture-replay tests (``test_live_fixtures.py``) and the live
Solr tests (``test_solr_container.py``, ``test_solr_conformance.py``), which
all use Solr's own JSON output as the reference for what the javabin bytes
must decode to.

The two renderings are not byte-identical by design, so the comparison knows
about every legitimate difference (dates, float32, binary, non-finite doubles,
NamedList rendering) and treats everything else as a defect. Each equivalence
is justified where it is implemented -- none of them is a tolerance fudge.
"""

from __future__ import annotations

import base64
import math
import re
import struct
from datetime import datetime, timezone
from typing import Any

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")

#: Keys whose values legitimately differ between two otherwise identical
#: requests: query timing, the echoed ``wt``/``version`` parameters, the
#: per-request id, and javabin's always-present ``maxScore`` slot (which the
#: JSON writer omits when there are no scores).
VOLATILE_KEYS = frozenset(
    {"QTime", "RESPONSE_TIME", "time", "wt", "version", "rid", "maxScore"}
)


def solr_date_to_millis(iso: str) -> int:
    """Parse a Solr ``tdate``/``pdate`` JSON string (``YYYY-MM-DDTHH:MM:SSZ``,
    optionally with fractional seconds) into milliseconds since the Unix epoch,
    matching the javabin ``DATE`` tag's representation.
    """
    if "." in iso:
        head, frac = iso[:-1].split(".")
        millis = int(round(float("0." + frac) * 1000))
    else:
        head, millis = iso[:-1], 0
    dt = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1000 + millis


def _as_float32(value: float) -> float:
    """Narrow a Python float to float32 precision and back."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error):
        return value


def floats_match(got: float, expected: float) -> bool:
    """Compare a decoded javabin float against its ``wt=json`` counterpart.

    A Solr ``pfloat``/``float`` field is a 4-byte value. ``wt=json`` prints the
    shortest decimal that round-trips through *float32* (``1.037``), while
    javabin transmits the raw 4 bytes, which widen to the exact float32 value
    (``1.0369999408721924``). Both denote the same float32, so narrow each side
    to float32 before comparing -- exactly, no tolerance fudge. Doubles are
    unaffected: they compare equal directly.
    """
    if math.isnan(got) and math.isnan(expected):
        return True
    return got == expected or _as_float32(got) == _as_float32(expected)


def _flat_pairs_to_dict(flat: list) -> dict | None:
    """Rebuild a dict from Solr's flat ``[k, v, k, v]`` JSON rendering.

    Solr's JSON writer renders some NamedLists as a flat array rather than an
    object -- facet field counts, facet range counts, stats percentiles. The
    javabin encoding is a NamedList either way, which this decoder maps to a
    dict, so the flat form has to be rebuilt before comparing.
    """
    if len(flat) % 2 or not all(isinstance(k, str) for k in flat[::2]):
        return None
    return dict(zip(flat[::2], flat[1::2]))


def _scalars_match(got: Any, ref: Any) -> bool:
    if isinstance(ref, str) and ISO_DATE_RE.match(ref) and isinstance(got, int):
        # javabin DATE carries millis since epoch; JSON carries an ISO string.
        return got == solr_date_to_millis(ref)
    if isinstance(got, bytes) and isinstance(ref, str):
        # javabin BYTEARR carries raw bytes; JSON carries base64.
        return got == base64.b64decode(ref)
    if isinstance(got, float) and isinstance(ref, float):
        return floats_match(got, ref)
    if isinstance(got, float) and isinstance(ref, str):
        # JSON has no non-finite literals, so Solr's writer stringifies them.
        if math.isinf(got):
            return ref == ("Infinity" if got > 0 else "-Infinity")
        return math.isnan(got) and ref == "NaN"
    return type(got) is type(ref) and got == ref


def diff_against_json(
    got: Any,
    ref: Any,
    path: str = "",
    out: list[str] | None = None,
    *,
    check_order: bool = True,
) -> list[str]:
    """Return every difference between a decoded javabin value and ``wt=json``.

    An empty list means the two agree modulo the documented equivalences.

    ``check_order`` compares the order of dict keys as well as their values. A
    NamedList is ordered and both writers preserve that order, so this catches a
    decoder that shuffles entries -- but a few sections (the collapse component's
    ``expanded``) are built from an unordered Java map and get a different order
    on every request, from either writer, so those have to opt out.
    """
    if out is None:
        out = []
    if isinstance(got, dict) and isinstance(ref, dict):
        got_keys = [k for k in got if k not in VOLATILE_KEYS and got[k] is not None]
        ref_keys = [k for k in ref if k not in VOLATILE_KEYS and ref[k] is not None]
        if set(got_keys) != set(ref_keys):
            out.append(
                f"{path or '<root>'}: javabin-only keys "
                f"{sorted(set(got_keys) - set(ref_keys))}, json-only keys "
                f"{sorted(set(ref_keys) - set(got_keys))}"
            )
        elif check_order and got_keys != ref_keys:
            out.append(f"{path or '<root>'}: key order {got_keys} != {ref_keys}")
        for key in ref_keys:
            if key in got:
                diff_against_json(
                    got[key], ref[key], f"{path}.{key}", out, check_order=check_order
                )
        return out
    if isinstance(got, dict) and isinstance(ref, list):
        rebuilt = _flat_pairs_to_dict(ref)
        if rebuilt is not None:
            return diff_against_json(
                got, rebuilt, f"{path}<flat>", out, check_order=check_order
            )
    if isinstance(got, list) and isinstance(ref, list):
        if len(got) != len(ref):
            out.append(f"{path}: length {len(got)} != {len(ref)}")
        for i, (g, r) in enumerate(zip(got, ref)):
            diff_against_json(g, r, f"{path}[{i}]", out, check_order=check_order)
        return out
    if not _scalars_match(got, ref):
        out.append(f"{path}: {got!r} != {ref!r}")
    return out


def assert_matches_json(
    got: Any, ref: Any, label: str = "", *, check_order: bool = True
) -> None:
    """Assert a decoded javabin response equals the ``wt=json`` response."""
    problems = diff_against_json(got, ref, check_order=check_order)
    assert not problems, (
        f"{label or 'response'} differs from its wt=json reference "
        f"({len(problems)} difference(s)):\n  " + "\n  ".join(problems[:15])
    )


def assert_docs_match(got_docs: list[dict], ref_docs: list[dict]) -> None:
    """Assert every field of every ``wt=json`` reference document is present and
    equal in the javabin-decoded documents.

    Unlike :func:`assert_matches_json` this ignores javabin-only fields, which
    the older fixtures rely on.
    """
    assert len(got_docs) == len(ref_docs)
    for got, expected in zip(got_docs, ref_docs):
        for key, expected_value in expected.items():
            got_value = got.get(key)
            if isinstance(expected_value, (dict, list)):
                assert_matches_json(got_value, expected_value, key)
            else:
                assert _scalars_match(got_value, expected_value), (
                    f"{key}: {got_value!r} != {expected_value!r}"
                )
