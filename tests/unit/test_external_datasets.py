"""The external-dataset registry and its fetchers.

Every test here is offline.  The download policy is the thing worth testing --
that a fetch is refused unless the licence and the permission are both recorded
as verified -- and that check must hold without a network.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import random
import threading
import urllib.error

import pytest

from intervene3d.data.external import (
    CONFIRM_THRESHOLD_GB,
    ExternalDataset,
    ExternalRegistry,
    FetchFailed,
    FetchRefused,
    RemoteFile,
    execute_fetch,
    fetchers,
    plan_fetch,
    validate_external_dataset,
    variants_for,
)
from intervene3d.data.external.fetchers import (
    FetchPlan,
    _download_resumable,
    _download_segmented,
    _hf_resolve_url,
    _next_link,
)


# --------------------------------------------------------------------- registry
def test_registry_loads_and_every_entry_has_the_required_fields():
    registry = ExternalRegistry()
    assert registry.keys(), "the registry is empty"
    for ds in registry.datasets.values():
        for field in ("title", "status", "licence_status", "priority", "instructions"):
            assert field in ds.payload, f"{ds.key} is missing {field!r}"
        assert ds.instructions().strip(), f"{ds.key} has empty instructions"


def test_by_priority_is_descending():
    priorities = [int(d.payload.get("priority", 0)) for d in ExternalRegistry().by_priority()]
    assert priorities == sorted(priorities, reverse=True)


def test_absent_dataset_reports_not_downloaded_rather_than_failing():
    ds = ExternalDataset("nowhere", {"expected_layout": "data/raw/__definitely_not_present__/"})
    report = validate_external_dataset(ds)
    assert report["present"] is False
    assert report["status"] == "NOT DOWNLOADED"


# ----------------------------------------------------------------------- policy
def _entry(**overrides):
    payload = {
        "expected_layout": "data/raw/policy_probe/",
        "licence": "CC0-1.0",
        "licence_status": "verified",
        "automated_download_permitted": True,
        "fetch": {
            "kind": "huggingface",
            "repo_id": "someone/something",
            "default_variant": "small",
            "variants": {"small": {"allow_patterns": ["*.parquet"], "approx_size_gb": 0.1}},
        },
    }
    payload.update(overrides)
    return ExternalDataset("policy_probe", payload)


def test_unverified_licence_is_refused():
    with pytest.raises(FetchRefused, match="requires both to be verified"):
        plan_fetch(_entry(licence_status="ACCESS UNVERIFIED"))


def test_permission_not_granted_is_refused_even_with_a_verified_licence():
    with pytest.raises(FetchRefused, match="requires both to be verified"):
        plan_fetch(_entry(automated_download_permitted=False))


def test_a_verified_entry_without_a_fetch_block_is_refused_not_guessed():
    entry = _entry()
    del entry.payload["fetch"]
    with pytest.raises(FetchRefused, match="no `fetch:` block"):
        plan_fetch(entry)


def test_unimplemented_backend_is_refused():
    entry = _entry()
    entry.payload["fetch"]["kind"] = "bittorrent"
    with pytest.raises(FetchRefused, match="no fetcher implemented"):
        plan_fetch(entry)


def test_unknown_variant_is_refused_and_lists_the_real_ones():
    with pytest.raises(FetchRefused, match="unknown variant"):
        plan_fetch(_entry(), "does_not_exist")


def test_plan_uses_the_default_variant_and_a_per_variant_subdirectory():
    plan = plan_fetch(_entry())
    assert plan.variant == "small"
    assert plan.dest.name == "small"
    assert plan.dest.parent.name == "policy_probe"


def test_large_transfer_is_held_until_explicitly_confirmed(tmp_path):
    big = int((CONFIRM_THRESHOLD_GB + 1) * 1e9)
    plan = FetchPlan(
        dataset_key="probe", variant="huge", kind="huggingface", repo_id="a/b",
        repo_type="dataset", revision="main", allow_patterns=("*",), dest=tmp_path / "huge",
        licence="CC0-1.0", source_url="https://example.invalid",
        files=(RemoteFile(path="x.bin", size=big, sha256=None),),
    )
    with pytest.raises(FetchRefused, match="confirmation threshold"):
        execute_fetch(plan, confirmed=False)


def test_every_permitted_entry_in_the_shipped_registry_actually_plans():
    """A `verified` + `permitted` entry must carry a usable fetch block."""
    for ds in ExternalRegistry().datasets.values():
        if not ds.may_auto_download:
            continue
        plan = plan_fetch(ds)  # must not raise
        assert plan.repo_id and plan.allow_patterns
        assert variants_for(ds), f"{ds.key} permits download but declares no variants"


def test_no_registry_entry_claims_permission_without_a_verified_licence():
    for ds in ExternalRegistry().datasets.values():
        if ds.payload.get("automated_download_permitted") is True:
            assert ds.licence_status == "verified", (
                f"{ds.key} permits automated download but its licence is {ds.licence_status!r}"
            )


# ------------------------------------------------------------------- mechanics
def test_resolve_url_escapes_the_path():
    url = _hf_resolve_url("princeton-vl/LayeredDepth", "dataset", "main", "data/train-0.parquet")
    assert url.endswith("/datasets/princeton-vl/LayeredDepth/resolve/main/data/train-0.parquet")


def test_pagination_link_header_is_followed():
    header = '<https://huggingface.co/api/x?cursor=abc>; rel="next", <https://y>; rel="prev"'
    assert _next_link(header) == "https://huggingface.co/api/x?cursor=abc"
    assert _next_link('<https://y>; rel="prev"') is None


# -------------------------------------------------------------------- manifest
def test_manifest_checksums_are_verified_and_a_corruption_is_caught(tmp_path):
    """A manifest written by a fetch must let validation detect later corruption."""
    root = tmp_path / "data" / "raw" / "probe"
    variant = root / "small"
    variant.mkdir(parents=True)
    payload = variant / "thing.bin"
    payload.write_bytes(b"intervene3d")
    import hashlib

    digest = hashlib.sha256(b"intervene3d").hexdigest()
    (variant / "manifest.json").write_text(json.dumps({"files": {"thing.bin": digest}}), encoding="utf-8")

    ds = ExternalDataset("probe", {"expected_layout": str(root)})
    report = validate_external_dataset(ds)
    assert report["present"] and report["checksums"] == "all 1 checksums match"
    assert report["variants"] == ["small"]

    payload.write_bytes(b"tampered")
    assert "mismatch" in validate_external_dataset(ds)["checksums"]


# ------------------------------------------------- transfer, against localhost
class _RangeServer(http.server.BaseHTTPRequestHandler):
    """Minimal static server that honours ``Range``; ``payload`` is set per test."""

    payload: bytes = b""
    honour_range: bool = True

    def do_GET(self):  # noqa: N802 - the stdlib spells it this way
        rng = self.headers.get("Range")
        if rng and self.honour_range:
            start, _, end = rng.removeprefix("bytes=").partition("-")
            lo = int(start)
            hi = int(end) if end else len(self.payload) - 1
            body = self.payload[lo : hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(self.payload)}")
        else:
            body = self.payload
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


@contextlib.contextmanager
def _serving(payload: bytes, *, honour_range: bool = True):
    handler = type("H", (_RangeServer,), {"payload": payload, "honour_range": honour_range})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def blob() -> bytes:
    return bytes(random.Random(0).getrandbits(8) for _ in range(300_000))


def test_segmented_download_reassembles_the_exact_bytes(tmp_path, blob):
    target = tmp_path / "out.bin"
    with _serving(blob) as url:
        _download_segmented(url, target, expected_size=len(blob), segments=5, progress=False, label="t")
    assert target.read_bytes() == blob
    assert not list(tmp_path.glob("*.part*")), "segment files were left behind"


def test_segmented_download_resumes_a_partial_segment(tmp_path, blob):
    """A half-written segment must be continued, not silently duplicated."""
    target = tmp_path / "out.bin"
    segments = 4
    span = -(-len(blob) // segments)
    (tmp_path / "out.bin.part1").write_bytes(blob[span : span + 100])  # partial segment 1
    with _serving(blob) as url:
        _download_segmented(url, target, expected_size=len(blob), segments=segments, progress=False, label="t")
    assert target.read_bytes() == blob


def test_segmented_download_falls_back_when_the_server_ignores_range(tmp_path, blob):
    target = tmp_path / "out.bin"
    with _serving(blob, honour_range=False) as url:
        _download_segmented(url, target, expected_size=len(blob), segments=4, progress=False, label="t")
    assert target.read_bytes() == blob


def test_a_truncated_response_is_rejected_rather_than_accepted(tmp_path, blob):
    with _serving(blob) as url:
        with pytest.raises(FetchFailed):
            _download_resumable(
                url, tmp_path / "out.bin", expected_size=len(blob) + 1_000,
                progress=False, label="t", retries=1,
            )
    assert not (tmp_path / "out.bin").exists()


def test_a_complete_file_is_not_refetched(tmp_path, blob):
    target = tmp_path / "out.bin"
    target.write_bytes(blob)
    # Point at a server serving different bytes: an unnecessary refetch would show.
    with _serving(b"x" * len(blob)) as url:
        _download_resumable(url, target, expected_size=len(blob), progress=False, label="t")
    assert target.read_bytes() == blob


# ------------------------------------------------- regressions (2026-08-29)
def test_expected_layout_placeholders_never_become_directories():
    """`.../{a,b}/...` and a trailing `...` are documentation, not path parts.

    Regression: `local_root` previously split only on `{`, so any entry whose
    layout had no brace alternation resolved to a directory literally named
    `...` -- 8 of the 14 shipped entries. A fetch then wrote to
    `data/raw/transphy3d/.../sample`.
    """
    for ds in ExternalRegistry().datasets.values():
        parts = ds.local_root.parts
        assert "..." not in parts, f"{ds.key} resolves to {ds.local_root}"
        assert not any(set(p) == {"."} for p in parts), f"{ds.key} resolves to {ds.local_root}"
        assert ds.local_root.parent.name == "raw", f"{ds.key} escaped data/raw: {ds.local_root}"


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("data/raw/thing/...", "thing"),
        ("data/raw/thing/{a,b}/...", "thing"),
        ("data/raw/thing/", "thing"),
        ("data/raw/thing", "thing"),
    ],
)
def test_local_root_ignores_every_placeholder_form(layout, expected):
    assert ExternalDataset("thing", {"expected_layout": layout}).local_root.name == expected


def test_local_root_falls_back_to_the_key_when_no_layout_is_recorded():
    assert ExternalDataset("mystery", {}).local_root.name == "mystery"


def test_a_variant_may_override_the_repository():
    """One entry can span two Hub repos (3D Visual Illusion: real vs virtual).

    Regression: `plan_fetch` read `fetch.repo_id` only, so a variant declaring
    its own `repo_id` silently fetched from the wrong repository.
    """
    entry = _entry()
    entry.payload["fetch"]["variants"]["elsewhere"] = {
        "repo_id": "other/repo",
        "allow_patterns": ["*.csv"],
    }
    assert plan_fetch(entry, "small").repo_id == "someone/something"
    assert plan_fetch(entry, "elsewhere").repo_id == "other/repo"


class _FlakyServer(http.server.BaseHTTPRequestHandler):
    """Fails `fail_first` requests, then serves `payload`. Counts attempts."""

    payload: bytes = b"[]"
    fail_first: int = 0
    attempts: list[int] = []

    def do_GET(self):  # noqa: N802
        type(self).attempts.append(1)
        if len(type(self).attempts) <= type(self).fail_first:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def _flaky(payload: bytes, fail_first: int):
    handler = type("F", (_FlakyServer,), {"payload": payload, "fail_first": fail_first, "attempts": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api", handler
    finally:
        server.shutdown()
        server.server_close()


def test_a_dropped_listing_request_is_retried(monkeypatch):
    """Regression: a dropped connection mid-pagination aborted the whole fetch.

    TransPhy3D needs 12 paginated calls; the Hub closed one of them and the
    listing raised, losing an otherwise valid download.
    """
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)  # no real backoff in tests
    with _flaky(b'{"ok": true}', fail_first=2) as (url, handler):
        assert fetchers._get_json(url) == {"ok": True}
    assert len(handler.attempts) == 3, "should have retried twice before succeeding"


def test_listing_gives_up_after_the_retry_budget(monkeypatch):
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)
    with _flaky(b"{}", fail_first=99) as (url, handler):
        with pytest.raises(FetchFailed, match="failed after 5 attempts"):
            fetchers._api_get(url)
    assert len(handler.attempts) == 5


def test_a_definitive_http_answer_is_not_retried(monkeypatch):
    """404 / 403 mean gated or missing. Retrying just wastes time and looks rude."""
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)

    class _NotFound(http.server.BaseHTTPRequestHandler):
        attempts: list[int] = []

        def do_GET(self):  # noqa: N802
            type(self).attempts.append(1)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError):
            fetchers._api_get(f"http://127.0.0.1:{server.server_address[1]}/api")
    finally:
        server.shutdown()
        server.server_close()
    assert len(_NotFound.attempts) == 1


@pytest.mark.parametrize(("code", "expect_retry"), [(429, True), (500, True), (503, True),
                                                    (400, False), (401, False), (403, False), (404, False)])
def test_only_rate_limits_and_server_errors_are_retried(monkeypatch, code, expect_retry):
    monkeypatch.setattr(fetchers.time, "sleep", lambda _s: None)

    class _Fixed(http.server.BaseHTTPRequestHandler):
        attempts: list[int] = []

        def do_GET(self):  # noqa: N802
            type(self).attempts.append(1)
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    handler = type("H", (_Fixed,), {"attempts": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises((FetchFailed, urllib.error.HTTPError)):
            fetchers._api_get(f"http://127.0.0.1:{server.server_address[1]}/api")
    finally:
        server.shutdown()
        server.server_close()
    assert len(handler.attempts) == (5 if expect_retry else 1)
