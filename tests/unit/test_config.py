"""Configuration loading, inheritance, overrides, hashing and validation."""

from __future__ import annotations

import pytest
import yaml

from intervene3d.config.loader import (
    ConfigError,
    apply_overrides,
    config_hash,
    deep_merge,
    dump_config,
    load_config,
)
from intervene3d.config.schema import validate_experiment_config, validate_synthetic_config


def test_deep_merge_child_wins_and_lists_are_replaced():
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2, 3]}
    out = deep_merge(base, {"a": {"c": 9}, "list": [7]})
    assert out == {"a": {"b": 1, "c": 9}, "list": [7]}
    assert base["a"]["c"] == 2  # the base is not mutated


def test_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"experiment": {"name": "b", "seed": 1}, "x": 1}))
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"_base": str(tmp_path / "base.yaml"), "experiment": {"seed": 7}})
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["experiment"] == {"name": "b", "seed": 7}
    assert cfg["x"] == 1


def test_circular_inheritance_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"_base": str(tmp_path / "b.yaml")}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"_base": str(tmp_path / "a.yaml")}))
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_config_and_missing_base(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"_base": "does/not/exist.yaml"}))
    with pytest.raises(ConfigError, match="was not found"):
        load_config(tmp_path / "c.yaml")


def test_overrides_parse_yaml_scalars_and_create_paths():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=5", "a.c=true", "d.e.f=0.25", "g=[1, 2]"])
    assert cfg["a"]["b"] == 5 and cfg["a"]["c"] is True
    assert cfg["d"]["e"]["f"] == 0.25 and cfg["g"] == [1, 2]
    with pytest.raises(ConfigError, match="key.path=value"):
        apply_overrides({}, ["bad-override"])


def test_config_hash_is_stable_and_ignores_transient_keys():
    a = {"x": 1, "y": {"z": 2}}
    b = {"y": {"z": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash({**a, "_config_path": "/tmp/x.yaml"}) == config_hash(a)
    assert config_hash({"x": 2, "y": {"z": 2}}) != config_hash(a)


def test_dump_config_strips_transient_keys(tmp_path):
    path = dump_config(tmp_path / "out.yaml", {"a": 1, "_config_path": "x"})
    assert yaml.safe_load(path.read_text()) == {"a": 1}


# ------------------------------------------------------------ synthetic schema
def test_synthetic_defaults_fill_in():
    cfg = validate_synthetic_config({})
    assert cfg["mechanisms"] == ["direct", "emissive", "reflection"]
    assert cfg["identifiability"]["epsilon_px"] == 1.0


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"dataset": {"num_base_scenes": 0}}, "num_base_scenes"),
        ({"mechanisms": ["direct"]}, "at least two"),
        ({"mechanisms": ["direct", "hologram"]}, "unknown mechanism"),
        ({"mechanisms": ["direct", "direct", "emissive"]}, "unique"),
        ({"camera": {"fx": -1}}, "must be positive"),
        ({"scene": {"content_fill": 0.0}}, "content_fill"),
        ({"scene": {"num_content_landmarks": 2}}, "num_content_landmarks"),
        ({"scene": {"interface_distance": [3.0]}}, "two-element"),
        ({"display": {"mode": "hologram"}}, "display.mode"),
        ({"display": {"view_tracked_probability": 2.0}}, "view_tracked_probability"),
        ({"identifiability": {"epsilon_px": 0.0}}, "epsilon_px"),
        ({"identifiability": {"distance": {"lambda_feature": 1.0}}}, "NOT IMPLEMENTED"),
        ({"splits": {"train": 0.9, "val": 0.2, "test": 0.3}}, "sum to 1.0"),
        ({"splits": {"policy": "frame"}}, "base_scene"),
    ],
)
def test_synthetic_validation_rejects_bad_configs(bad, match):
    with pytest.raises(ConfigError, match=match):
        validate_synthetic_config(bad)


# ----------------------------------------------------------- experiment schema
def _minimal_experiment(**over):
    cfg = {
        "experiment": {"name": "unit_test", "seed": 1},
        "data": {"synthetic_config": "configs/synthetic/smoke.yaml", "split": "test"},
        "methods": [{"name": "m1"}],
    }
    cfg.update(over)
    return cfg


def test_experiment_defaults_and_valid_config():
    cfg = validate_experiment_config(_minimal_experiment())
    assert cfg["model"]["transition"]["name"] == "analytical"
    assert cfg["visualization"]["formats"] == ["pdf", "png"]


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"experiment": {"name": "", "seed": 1}}, "experiment.name"),
        ({"experiment": {"name": "bad name!", "seed": 1}}, "alphanumerics"),
        ({"data": {"split": "test"}}, "dataset_dir or data.synthetic_config"),
        ({"data": {"synthetic_config": "x", "split": "nope"}}, "data.split"),
        ({"methods": []}, "at least one entry"),
        ({"methods": [{"name": "a"}, {"name": "a"}]}, "duplicate method"),
        ({"methods": [{"name": "a", "selector": "telepathy"}]}, "selector"),
        ({"methods": [{"name": "a", "selector": "fixed"}]}, "fixed_action"),
        ({"methods": [{"name": "a", "transition": "magic"}]}, "invalid"),
    ],
)
def test_experiment_validation_rejects_bad_configs(bad, match):
    with pytest.raises(ConfigError, match=match):
        validate_experiment_config(_minimal_experiment(**bad))


def test_experiment_validation_rejects_unimplemented_distance_term():
    cfg = _minimal_experiment()
    cfg["model"] = {"distance": {"lambda_feature": 0.5}}
    with pytest.raises(ConfigError, match="NOT IMPLEMENTED"):
        validate_experiment_config(cfg)


def test_shipped_configs_are_all_valid(repo_root):
    """Every config in the repository must load and validate."""
    for path in sorted((repo_root / "configs" / "synthetic").glob("*.yaml")):
        validate_synthetic_config(load_config(path))
    for path in sorted((repo_root / "configs" / "experiments").glob("*.yaml")):
        validate_experiment_config(load_config(path))
    validate_experiment_config(load_config(repo_root / "configs" / "smoke_test.yaml"))


def test_config_hash_is_seed_independent():
    """Runs differing only in seed must share a hash, or aggregation breaks."""
    a = validate_experiment_config(_minimal_experiment(experiment={"name": "x", "seed": 1}))
    b = validate_experiment_config(_minimal_experiment(experiment={"name": "x", "seed": 2}))
    assert config_hash(a) == config_hash(b)
    c = validate_experiment_config(_minimal_experiment(experiment={"name": "y", "seed": 1}))
    assert config_hash(a) != config_hash(c)
