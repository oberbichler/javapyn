"""
Pytest fixtures for the opt-in live-Solr conformance tests.

The ``solr`` fixture starts one real Solr container per test session (see the
``solr`` marker in ``pyproject.toml``; these tests are deselected by default).
Its point is oracle independence: every other test in this repo validates the
decoder against our own reading of the javabin spec — either the hand-written
reference encoder or fixtures captured once from an internal Solr — while these
compare against bytes produced by Apache Solr's own ``JavaBinCodec``, on demand,
on any machine with Docker.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from solr_probe import SolrProbe

#: Pinned on purpose: a rolling ``solr:9`` tag would make CI non-reproducible.
#: Override to try another release, e.g. JAVAPYN_SOLR_IMAGE=solr:10.0.0.
SOLR_IMAGE = os.environ.get("JAVAPYN_SOLR_IMAGE", "solr:9.10.1")

SOLR_PORT = 8983


def _solr_command(image: str) -> str:
    """The container command that brings up SolrCloud for this image.

    SolrCloud mode with embedded ZooKeeper is required, not incidental: the
    ``/stream`` handler resolves collections through ZooKeeper and fails in
    standalone core mode. Solr 8 and 9 need ``-c`` to get there; Solr 10 removed
    that flag because SolrCloud became the default (``--user-managed`` opts out),
    and passing ``-c`` there makes the container exit with a usage message.
    """
    tag = image.rpartition(":")[2]
    major = tag.partition(".")[0]
    if major.isdigit() and int(major) >= 10:
        return "solr-foreground"
    return "solr-foreground -c"


def _docker_unavailable_reason() -> str | None:
    """Return why Docker can't be used, or None if it can."""
    try:
        from testcontainers.core.docker_client import DockerClient

        DockerClient().client.ping()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


@pytest.fixture(scope="session")
def solr() -> Iterator[SolrProbe]:
    """A provisioned Solr collection in a throwaway container.

    Skips (rather than fails) without a reachable Docker daemon, so
    ``pytest -m solr`` degrades gracefully on machines and CI runners without
    one -- notably the macOS and Windows GitHub runners.
    """
    reason = _docker_unavailable_reason()
    if reason is not None:
        pytest.skip(f"Docker not available: {reason}")

    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer(SOLR_IMAGE)
        .with_command(_solr_command(SOLR_IMAGE))
        .with_exposed_ports(SOLR_PORT)
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(SOLR_PORT)
        probe = SolrProbe(f"http://{host}:{port}/solr")
        try:
            probe.wait_until_ready()
            probe.provision()
            yield probe
        finally:
            probe.close()
    finally:
        container.stop()
