"""YAML configuration loading, inheritance, overrides and hashing.

Nothing in this repository hard-codes an experiment parameter.  A configuration
is a plain nested dictionary so that:

* it round-trips to YAML without loss and is archived verbatim in every run
  directory;
* ``--set a.b.c=value`` overrides are unambiguous;
* the *canonical* JSON serialisation gives a stable ``config_hash`` that appears
  in the run directory name, so two runs with the same configuration are
  immediately recognisable and two runs with different configurations can never
  collide.

A config may declare ``_base: path/to/other.yaml`` (relative to its own
directory or to the repository root) and the two are deep-merged, child wins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for any malformed or inconsistent configuration."""


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[3]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins.  Lists are replaced, not merged."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_base(base: str, config_path: Path) -> Path:
    candidates = [config_path.parent / base, repo_root() / base, Path(base)]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise ConfigError(f"_base {base!r} referenced by {config_path} was not found (tried {candidates})")


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration {path} must be a mapping at the top level")
    return payload


def load_config(
    path: Path | str,
    *,
    overrides: Iterable[str] | None = None,
    _seen: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base`` inheritance and applying overrides."""
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"circular _base inheritance detected at {path}")
    payload = load_yaml(path)

    base_ref = payload.pop("_base", None)
    if base_ref is not None:
        base_path = _resolve_base(str(base_ref), path)
        base_cfg = load_config(base_path, _seen=_seen + (path,))
        payload = deep_merge(base_cfg, payload)

    payload["_config_path"] = str(path)
    if overrides:
        payload = apply_overrides(payload, overrides)
    return payload


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides.  Values are parsed as YAML scalars."""
    out = json.loads(json.dumps(config, default=str)) if False else dict(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must be of the form key.path=value")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ConfigError(f"could not parse override value {raw!r}: {exc}") from exc
        _set_nested(out, key.strip().split("."), value)
    return out


def _set_nested(config: dict[str, Any], keys: list[str], value: Any) -> None:
    node = config
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


#: Keys excluded from the configuration hash.
#:
#: The hash must identify the *experimental configuration*, not one run of it.
#: The seed is recorded separately -- in the run directory name, in
#: ``run_manifest.json`` and in the registry -- so folding it into the hash would
#: give every seed a different hash and make multi-seed aggregation impossible,
#: which is exactly the mistake the research plan warns against ("do not report
#: one lucky seed"). Everything that changes what the experiment *is* still
#: contributes to the hash.
HASH_EXCLUDED_PATHS: tuple[str, ...] = ("experiment.seed",)


def _strip_paths(config: Mapping[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps({k: v for k, v in config.items() if not k.startswith("_")}, default=str))
    for path in paths:
        node: Any = cleaned
        parts = path.split(".")
        for key in parts[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return cleaned


def canonical_json(config: Mapping[str, Any], *, exclude: Iterable[str] | None = None) -> str:
    """Deterministic serialisation used for hashing.

    Transient ``_``-prefixed keys and :data:`HASH_EXCLUDED_PATHS` are removed.
    """
    cleaned = _strip_paths(config, HASH_EXCLUDED_PATHS if exclude is None else exclude)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any], *, length: int = 8) -> str:
    """Short, stable SHA-256 fingerprint of a configuration (seed-independent)."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:length]


def full_config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def dump_config(path: Path | str, config: Mapping[str, Any]) -> Path:
    """Write a config back to YAML (used to archive it inside a run directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v for k, v in config.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cleaned, handle, sort_keys=True, default_flow_style=False)
    return path
