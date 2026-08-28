"""Query helpers over the experiment registry.

Thin wrappers around :mod:`intervene3d.reproducibility.manifest`, kept here so
``intervene3d.experiments`` is a complete surface for anything experiment-shaped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intervene3d.reproducibility.manifest import (
    load_registry,
    registry_path,
    runs_for_experiment,
)
from intervene3d.utils.io import load_json


def experiment_names(root: Path | str = "experiments") -> list[str]:
    return sorted({r["experiment_name"] for r in load_registry(root)})


def load_run_metrics(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Load ``metrics.json`` for one registry entry, or ``None`` if absent."""
    if not entry.get("metrics_file"):
        return None
    path = Path(entry["run_path"]) / entry["metrics_file"]
    return load_json(path) if path.exists() else None


def summarise_registry(root: Path | str = "experiments") -> list[dict[str, Any]]:
    """One row per experiment: run count, seeds, statuses."""
    rows = load_registry(root)
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = r["experiment_name"]
        item = out.setdefault(
            name, {"experiment_name": name, "runs": 0, "seeds": [], "success": 0, "failed": 0, "config_hashes": set()}
        )
        item["runs"] += 1
        item["seeds"].append(r.get("seed"))
        item["config_hashes"].add(r.get("config_hash"))
        item["success" if r.get("status") == "success" else "failed"] += 1
    for item in out.values():
        item["seeds"] = sorted({s for s in item["seeds"] if s is not None})
        item["config_hashes"] = sorted(h for h in item["config_hashes"] if h)
    return [out[k] for k in sorted(out)]


__all__ = [
    "experiment_names",
    "load_registry",
    "load_run_metrics",
    "registry_path",
    "runs_for_experiment",
    "summarise_registry",
]
