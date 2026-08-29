"""Licence-respecting fetchers for the external dataset registry.

The rules from ``registry.py`` still hold and are re-checked here rather than
trusted: a fetch runs only when the registry records **both** a ``verified``
licence and ``automated_download_permitted: true``.  Everything else prints
instructions and exits.

What a fetch does, in order:

1. resolve a :class:`FetchPlan` from the registry entry and the requested
   variant (a dataset may publish several splits of very different size);
2. list the remote files and report the exact byte count **before** anything is
   transferred -- ``docs/DATASET_MATRIX.md`` forbids downloading an unknown
   quantity;
3. transfer, resuming any partial file rather than restarting it;
4. verify each file against the publisher's own SHA-256 where the host exposes
   one (Hugging Face publishes it as ``lfs.oid``);
5. write ``manifest.json`` next to the data, with per-file SHA-256, the resolved
   commit revision, the source URL and the recorded licence, so
   ``validate_external_dataset`` can check integrity later and so a run is
   reproducible from the manifest alone.

Only the Hugging Face Hub backend is implemented.  It is deliberately pure
standard library: the preliminary pipeline stays NumPy-only, and adding a
download dependency for four registry entries is not worth it.  ``requirements``
therefore does not grow.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from intervene3d.data.external.registry import STATUS_VERIFIED, ExternalDataset
from intervene3d.reproducibility.hashing import sha256_file
from intervene3d.utils.io import dump_json

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = "intervene3d/0.1.0 (research; +https://github.com/)"

#: A transfer larger than this (decimal GB, as the registry states sizes)
#: needs an explicit confirmation flag.  The number
#: is a policy choice, not a technical limit: nothing large is ever pulled by
#: accident.
CONFIRM_THRESHOLD_GB = 1.0

_CHUNK = 1 << 20
#: Below this, splitting a file across connections costs more in round trips
#: than it saves in throughput.
_SEGMENT_MIN_BYTES = 256 << 20
#: Runaway guard for pagination. 1000 entries per page, so this covers a
#: 50k-file repository -- larger than anything in the registry.
_MAX_LIST_PAGES = 50


class FetchRefused(RuntimeError):
    """Raised when policy forbids the download.  Never a bug -- the normal path."""


class FetchFailed(RuntimeError):
    """Raised when a permitted download could not be completed."""


@dataclass(frozen=True)
class RemoteFile:
    """One file on the remote host."""

    path: str
    size: int
    sha256: str | None  # the publisher's checksum, when it exposes one

    @property
    def size_gb(self) -> float:
        return self.size / 1e9


@dataclass(frozen=True)
class FetchPlan:
    """Everything needed to run and record one acquisition."""

    dataset_key: str
    variant: str
    kind: str
    repo_id: str
    repo_type: str
    revision: str
    allow_patterns: tuple[str, ...]
    dest: Path
    licence: str
    source_url: str
    declared_size_gb: float | None = None
    files: tuple[RemoteFile, ...] = field(default=())

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1e9

    def describe(self) -> str:
        lines = [
            f"  backend               : {self.kind}",
            f"  repository            : {self.repo_id} ({self.repo_type})",
            f"  revision              : {self.revision}",
            f"  variant               : {self.variant}",
            f"  patterns              : {list(self.allow_patterns)}",
            f"  destination           : {self.dest}",
            f"  licence (as recorded) : {self.licence}",
        ]
        if self.files:
            lines.append(f"  remote files          : {len(self.files)}")
            lines.append(f"  transfer size         : {self.total_gb:.3f} GB ({self.total_bytes:,} bytes)")
        elif self.declared_size_gb is not None:
            lines.append(f"  transfer size         : ~{self.declared_size_gb:.3f} GB (declared; not yet listed)")
        return "\n".join(lines)


# --------------------------------------------------------------------- planning
def variants_for(dataset: ExternalDataset) -> dict[str, dict[str, Any]]:
    """The named subsets a registry entry offers, or ``{}`` if it has no fetcher."""
    fetch = dataset.payload.get("fetch") or {}
    return dict(fetch.get("variants") or {})


def plan_fetch(dataset: ExternalDataset, variant: str | None = None) -> FetchPlan:
    """Build a :class:`FetchPlan`, refusing anything policy does not allow.

    Raises :class:`FetchRefused` with a message suitable for printing verbatim.
    """
    if dataset.licence_status != STATUS_VERIFIED or not dataset.automated_download_permitted:
        raise FetchRefused(
            f"licence_status={dataset.licence_status!r}, "
            f"automated_download_permitted={dataset.payload.get('automated_download_permitted')!r}; "
            "automated download requires both to be verified"
        )
    fetch = dataset.payload.get("fetch")
    if not fetch:
        raise FetchRefused(
            "the registry entry permits automated download but records no `fetch:` block; "
            "add one naming the backend and the remote repository"
        )
    kind = str(fetch.get("kind", "")).lower()
    if kind != "huggingface":
        raise FetchRefused(f"no fetcher implemented for backend {kind!r} (only 'huggingface' exists)")

    variants = variants_for(dataset)
    if not variants:
        raise FetchRefused("the `fetch:` block declares no variants")
    if variant is None:
        variant = str(fetch.get("default_variant") or next(iter(variants)))
    if variant not in variants:
        raise FetchRefused(f"unknown variant {variant!r}; available: {sorted(variants)}")

    spec = variants[variant]
    # A variant may live in a different repository from the entry's default --
    # 3D Visual Illusion splits its real and virtual data across two Hub repos.
    repo_id = str(spec.get("repo_id") or fetch["repo_id"])
    repo_type = str(spec.get("repo_type") or fetch.get("repo_type", "dataset"))
    revision = str(spec.get("revision") or fetch.get("revision") or "main")
    patterns = tuple(spec.get("allow_patterns") or ["*"])
    subdir = str(spec.get("subdir") or variant)
    dest = dataset.local_root / subdir
    return FetchPlan(
        dataset_key=dataset.key,
        variant=variant,
        kind=kind,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=patterns,
        dest=dest,
        licence=str(dataset.payload.get("licence", "unknown")),
        source_url=_repo_url(repo_id, repo_type),
        declared_size_gb=spec.get("approx_size_gb"),
    )


def resolve_plan(plan: FetchPlan) -> FetchPlan:
    """Attach the remote file listing, so the exact size is known before transfer."""
    files = _hf_list_files(plan.repo_id, plan.repo_type, plan.revision, plan.allow_patterns)
    if not files:
        raise FetchFailed(
            f"no remote file under {plan.repo_id}@{plan.revision} matched {list(plan.allow_patterns)}"
        )
    return FetchPlan(**{**plan.__dict__, "files": tuple(files)})


# ------------------------------------------------------------------- execution
def execute_fetch(
    plan: FetchPlan,
    *,
    confirmed: bool = False,
    workers: int = 4,
    progress: bool = True,
) -> dict[str, Any]:
    """Run a resolved plan and write ``manifest.json``.  Returns the manifest."""
    if not plan.files:
        plan = resolve_plan(plan)
    if plan.total_gb > CONFIRM_THRESHOLD_GB and not confirmed:
        raise FetchRefused(
            f"this transfer is {plan.total_gb:.2f} GB, above the {CONFIRM_THRESHOLD_GB:.1f} GB "
            "confirmation threshold; re-run with --yes to proceed"
        )
    free_gb = shutil.disk_usage(_existing_ancestor(plan.dest)).free / 1e9
    if free_gb < plan.total_gb * 1.10:
        raise FetchRefused(
            f"insufficient free space: {free_gb:.1f} GB available, "
            f"{plan.total_gb * 1.10:.1f} GB needed (transfer + 10% headroom)"
        )

    plan.dest.mkdir(parents=True, exist_ok=True)
    commit = _hf_commit_sha(plan.repo_id, plan.repo_type, plan.revision) or plan.revision
    started = time.time()
    done: list[tuple[RemoteFile, Path]] = []
    errors: list[str] = []

    # One connection per file saturates nothing when a variant is a single large
    # tarball (3D Visual Illusion ships 8.1 GB as one file).  Spread the spare
    # workers across byte ranges of each file instead.
    segments_per_file = max(1, workers // max(1, len(plan.files)))

    def _one(rf: RemoteFile) -> tuple[RemoteFile, Path]:
        target = plan.dest / rf.path
        url = _hf_resolve_url(plan.repo_id, plan.repo_type, commit, rf.path)
        if segments_per_file > 1 and rf.size > _SEGMENT_MIN_BYTES:
            _download_segmented(
                url, target, expected_size=rf.size, segments=segments_per_file,
                progress=progress, label=rf.path,
            )
        else:
            _download_resumable(url, target, expected_size=rf.size, progress=progress, label=rf.path)
        if rf.sha256:
            actual = sha256_file(target)
            if actual != rf.sha256:
                target.unlink(missing_ok=True)
                raise FetchFailed(f"checksum mismatch for {rf.path}: expected {rf.sha256}, got {actual}")
        return rf, target

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_one, rf): rf for rf in plan.files}
        for i, fut in enumerate(as_completed(futures), start=1):
            rf = futures[fut]
            try:
                done.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                errors.append(f"{rf.path}: {type(exc).__name__}: {exc}")
                continue
            if progress:
                _status(f"  [{i}/{len(plan.files)}] {rf.path}  ({rf.size_gb:.3f} GB)")
    if progress:
        sys.stderr.write("\n")
    if errors:
        raise FetchFailed(f"{len(errors)} file(s) failed:\n  " + "\n  ".join(errors))

    manifest = {
        "dataset": plan.dataset_key,
        "variant": plan.variant,
        "source": {
            "backend": plan.kind,
            "repo_id": plan.repo_id,
            "repo_type": plan.repo_type,
            "url": plan.source_url,
            "revision_requested": plan.revision,
            "revision_resolved": commit,
            "allow_patterns": list(plan.allow_patterns),
        },
        "licence_as_recorded": plan.licence,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "n_files": len(done),
        "total_bytes": plan.total_bytes,
        "total_gb": round(plan.total_gb, 4),
        # `files` is the schema validate_external_dataset() checks: relpath -> sha256.
        "files": {rf.path: (rf.sha256 or sha256_file(path)) for rf, path in sorted(done, key=lambda t: t[0].path)},
        # How many checksums came from the publisher (Hugging Face `lfs.oid`)
        # rather than being computed here.  Small non-LFS files have none.
        "publisher_checksums": sum(1 for rf in plan.files if rf.sha256),
    }
    dump_json(plan.dest / "manifest.json", manifest)
    return manifest


# ----------------------------------------------------------- Hugging Face Hub
def _repo_url(repo_id: str, repo_type: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ("models/" if repo_type == "model" else f"{repo_type}s/")
    if repo_type == "model":
        prefix = ""
    return f"{HF_ENDPOINT}/{prefix}{repo_id}"


def _hf_api_url(repo_id: str, repo_type: str, suffix: str = "") -> str:
    kind = {"dataset": "datasets", "model": "models", "space": "spaces"}.get(repo_type, "datasets")
    return f"{HF_ENDPOINT}/api/{kind}/{repo_id}{suffix}"


def _hf_resolve_url(repo_id: str, repo_type: str, revision: str, path: str) -> str:
    base = _repo_url(repo_id, repo_type)
    return f"{base}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"


def _request(url: str, *, headers: dict[str, str] | None = None) -> urllib.request.Request:
    hdrs = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        # Present only when the user has already supplied credentials themselves.
        # Nothing here creates, requests or stores a token.
        hdrs["Authorization"] = f"Bearer {token}"
    hdrs.update(headers or {})
    return urllib.request.Request(url, headers=hdrs)


def _api_get(url: str, *, timeout: int = 120, retries: int = 5) -> tuple[Any, str]:
    """One JSON API call, retried with backoff.  Returns ``(payload, Link header)``.

    Listing a large repository takes many paginated calls, and the Hub will drop
    a connection partway through (``WinError 10054`` / ``RemoteDisconnected``)
    when they arrive in a burst.  A dropped listing used to abort the whole
    fetch, which is a silly way to lose an otherwise valid download.
    """
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            # 429 (rate limited) and 5xx are worth another try; every other 4xx is
            # a definitive answer -- gated, missing, forbidden -- and hammering it
            # would be both useless and rude.
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < retries:
            time.sleep(min(30, 2**attempt))
    raise FetchFailed(f"listing {url} failed after {retries} attempts: {type(last).__name__}: {last}")


def _get_json(url: str, *, timeout: int = 60) -> Any:
    return _api_get(url, timeout=timeout)[0]


def hf_repo_info(repo_id: str, repo_type: str = "dataset") -> dict[str, Any]:
    """Public metadata for a Hub repository (licence, gating, last modified)."""
    return _get_json(_hf_api_url(repo_id, repo_type))


def _hf_commit_sha(repo_id: str, repo_type: str, revision: str) -> str | None:
    try:
        info = _get_json(_hf_api_url(repo_id, repo_type, f"/revision/{urllib.parse.quote(revision, safe='')}"))
    except Exception:  # noqa: BLE001 - a missing sha is not fatal, it is just less precise
        return None
    return info.get("sha")


def _hf_list_files(
    repo_id: str, repo_type: str, revision: str, patterns: tuple[str, ...]
) -> list[RemoteFile]:
    """List matching files, following the Hub's cursor pagination."""
    # No `expand=true`: it caps a page at 50 entries instead of 1000 while
    # returning the same lfs.oid, which turns an 11k-file repository into 200+
    # round trips for nothing.
    url = _hf_api_url(
        repo_id, repo_type, f"/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    )
    out: list[RemoteFile] = []
    seen: set[str] = set()
    pages = 0
    while url:
        entries, link = _api_get(url)
        pages += 1
        if pages > _MAX_LIST_PAGES:
            raise FetchFailed(
                f"listing {repo_id} exceeded {_MAX_LIST_PAGES} pages; narrow `allow_patterns`"
            )
        for entry in entries:
            if entry.get("type") != "file":
                continue
            path = entry["path"]
            if path in seen or not any(fnmatch(path, p) for p in patterns):
                continue
            seen.add(path)
            lfs = entry.get("lfs") or {}
            out.append(
                RemoteFile(
                    path=path,
                    size=int(lfs.get("size") or entry.get("size") or 0),
                    sha256=lfs.get("oid") or lfs.get("sha256"),
                )
            )
        url = _next_link(link)
    return sorted(out, key=lambda f: f.path)


def _next_link(header: str) -> str | None:
    for part in header.split(","):
        if 'rel="next"' in part:
            start, end = part.find("<"), part.find(">")
            if 0 <= start < end:
                return part[start + 1 : end]
    return None


# ------------------------------------------------------------------ transfer
def _download_resumable(
    url: str, target: Path, *, expected_size: int, progress: bool, label: str, retries: int = 4
) -> Path:
    """Fetch ``url`` to ``target``, resuming a partial ``.part`` if one exists."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected_size and target.stat().st_size == expected_size:
        return target  # already complete
    part = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected_size and have == expected_size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(_request(url, headers=headers), timeout=120) as resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if have else "wb"
                with part.open(mode) as handle:
                    while True:
                        block = resp.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        have += len(block)
                        if progress and expected_size:
                            _status(f"  {label}  {100 * have / expected_size:5.1f}%  "
                                    f"({have / (1 << 20):,.0f}/{expected_size / (1 << 20):,.0f} MiB)")
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise FetchFailed(f"{label}: {type(exc).__name__}: {exc}") from exc
            time.sleep(min(30, 2**attempt))

    size = part.stat().st_size if part.exists() else 0
    if expected_size and size != expected_size:
        raise FetchFailed(f"{label}: expected {expected_size} bytes, got {size}")
    part.replace(target)
    return target


def _download_segmented(
    url: str, target: Path, *, expected_size: int, segments: int, progress: bool, label: str
) -> Path:
    """Fetch one large file over several concurrent byte ranges, then join them.

    Each segment is its own resumable ``.partN`` file, so an interrupted transfer
    restarts only the segments that were incomplete.  Falls back to the
    single-stream path if the server does not honour ``Range``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        return target
    span = -(-expected_size // segments)  # ceil
    bounds = [(i * span, min(expected_size, (i + 1) * span) - 1) for i in range(segments)]
    bounds = [(a, b) for a, b in bounds if a <= b]
    parts = [target.with_suffix(target.suffix + f".part{i}") for i in range(len(bounds))]
    fetched = [0] * len(bounds)

    def _segment(i: int) -> None:
        start, end = bounds[i]
        part = parts[i]
        want = end - start + 1
        have = part.stat().st_size if part.exists() else 0
        if have > want:  # a stale part from a different segmentation
            part.unlink()
            have = 0
        fetched[i] = have
        if have == want:
            return
        for attempt in range(1, 5):
            try:
                req = _request(url, headers={"Range": f"bytes={start + have}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise FetchFailed("server ignored Range")
                    with part.open("ab" if have else "wb") as handle:
                        while True:
                            block = resp.read(_CHUNK)
                            if not block:
                                break
                            handle.write(block)
                            have += len(block)
                            fetched[i] = have
                            if progress:
                                total = sum(fetched)
                                _status(f"  {label}  {100 * total / expected_size:5.1f}%  "
                                        f"({total / (1 << 20):,.0f}/{expected_size / (1 << 20):,.0f} MiB, "
                                        f"{len(bounds)} streams)")
                if have == want:
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 4:
                    raise FetchFailed(f"{label} segment {i}: {type(exc).__name__}: {exc}") from exc
                time.sleep(min(30, 2**attempt))
                have = part.stat().st_size if part.exists() else 0
        raise FetchFailed(f"{label} segment {i}: incomplete after retries")

    try:
        with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
            for fut in as_completed([pool.submit(_segment, i) for i in range(len(bounds))]):
                fut.result()
    except FetchFailed as exc:
        if "ignored Range" not in str(exc):
            raise
        for part in parts:
            part.unlink(missing_ok=True)
        return _download_resumable(
            url, target, expected_size=expected_size, progress=progress, label=label
        )

    joined = target.with_suffix(target.suffix + ".part")
    with joined.open("wb") as out:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, _CHUNK)
    size = joined.stat().st_size
    if size != expected_size:
        joined.unlink(missing_ok=True)
        raise FetchFailed(f"{label}: joined {size} bytes, expected {expected_size}")
    joined.replace(target)
    for part in parts:
        part.unlink(missing_ok=True)
    return target


def _existing_ancestor(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return Path.cwd()  # pragma: no cover


def _status(text: str) -> None:
    sys.stderr.write("\r\033[K" + text[:150])
    sys.stderr.flush()
