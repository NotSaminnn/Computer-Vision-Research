"""Deterministic, machine-readable I/O helpers.

Rule from the research plan (section 25): results must never live only in logs.
Everything written here is either JSON, JSONL or CSV so that figures can be
regenerated from files without re-running an experiment.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _default(obj: Any) -> Any:
    """JSON encoder fallback for NumPy scalars/arrays and Paths."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def dump_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as pretty, sorted, deterministic JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, sort_keys=True, default=_default)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_default) + "\n")
    return path


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_default) + "\n")
    return path


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str] | None = None) -> Path:
    """Write ``rows`` to CSV with a stable column order.

    Parquet is the format the plan mentions first, but it would add a pandas /
    pyarrow dependency for no benefit at this scale.  CSV keeps the preliminary
    codebase dependency-light and is trivially readable; see
    ``docs/SOFTWARE_ARCHITECTURE.md`` for the documented deviation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist())
    if value is None:
        return ""
    return value


def write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
