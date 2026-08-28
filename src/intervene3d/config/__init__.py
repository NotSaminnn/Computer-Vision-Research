"""Configuration system: YAML loading, inheritance, overrides, hashing, validation."""

from intervene3d.config.loader import (
    ConfigError,
    apply_overrides,
    canonical_json,
    config_hash,
    deep_merge,
    dump_config,
    full_config_hash,
    load_config,
    load_yaml,
    repo_root,
)
from intervene3d.config.schema import (
    EXPERIMENT_DEFAULTS,
    SYNTHETIC_DEFAULTS,
    validate_experiment_config,
    validate_synthetic_config,
)

__all__ = [
    "ConfigError",
    "apply_overrides",
    "canonical_json",
    "config_hash",
    "deep_merge",
    "dump_config",
    "full_config_hash",
    "load_config",
    "load_yaml",
    "repo_root",
    "EXPERIMENT_DEFAULTS",
    "SYNTHETIC_DEFAULTS",
    "validate_experiment_config",
    "validate_synthetic_config",
]
