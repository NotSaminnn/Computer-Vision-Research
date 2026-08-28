"""Run manifests and the lightweight experiment registry.

The registry is a single append-only JSONL file at ``experiments/registry.jsonl``.
One line per run, never rewritten, so aggregation across runs and seeds is a
plain file read and a crashed run leaves a legible record rather than a hole.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intervene3d.utils.io import append_jsonl, dump_json, load_json, load_jsonl

REGISTRY_FILENAME = "registry.jsonl"


def registry_path(root: Path | str = "experiments") -> Path:
    return Path(root) / REGISTRY_FILENAME


def finalise_run(
    run_path: Path | str,
    *,
    status: str,
    metrics_file: str | None = None,
    figures: list[str] | None = None,
    tables: list[str] | None = None,
    summary: Mapping[str, Any] | None = None,
    error: str | None = None,
    duration_seconds: float | None = None,
    registry_root: Path | str = "experiments",
) -> dict[str, Any]:
    """Update ``run_manifest.json`` and append the run to the registry."""
    run_path = Path(run_path)
    manifest = load_json(run_path / "run_manifest.json")
    manifest.update(
        {
            "status": status,
            "metrics_file": metrics_file,
            "figures": figures or [],
            "tables": tables or [],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": duration_seconds,
        }
    )
    if summary is not None:
        manifest["summary"] = dict(summary)
    if error:
        manifest["error"] = error
    dump_json(run_path / "run_manifest.json", manifest)

    append_jsonl(
        registry_path(registry_root),
        {
            "experiment_name": manifest["experiment_name"],
            "run_id": manifest["run_id"],
            "run_path": str(run_path),
            "seed": manifest["seed"],
            "config_hash": manifest["config_hash"],
            "git_commit": manifest["git_commit"],
            "dataset_manifest": manifest.get("dataset_manifest"),
            "status": status,
            "metrics_file": metrics_file,
            "figures": figures or [],
            "created_utc": manifest["created_utc"],
            "finished_utc": manifest["finished_utc"],
            "duration_seconds": duration_seconds,
        },
    )
    return manifest


def load_registry(root: Path | str = "experiments") -> list[dict[str, Any]]:
    return load_jsonl(registry_path(root))


def runs_for_experiment(
    experiment_name: str, *, root: Path | str = "experiments", status: str | None = "success"
) -> list[dict[str, Any]]:
    """Registry entries for one experiment, newest last, optionally filtered by status."""
    rows = [r for r in load_registry(root) if r.get("experiment_name") == experiment_name]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return sorted(rows, key=lambda r: str(r.get("created_utc", "")))
