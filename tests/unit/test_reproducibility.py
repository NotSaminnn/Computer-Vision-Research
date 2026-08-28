"""Determinism: the same seed must produce the same everything."""

from __future__ import annotations

import numpy as np

from intervene3d.config import validate_synthetic_config
from intervene3d.config.loader import config_hash
from intervene3d.data.splits import assign_split, build_splits, check_no_leakage
from intervene3d.data.synthetic import (
    action_space_from_config,
    build_hypothesis_set,
    generate_base_scene,
    reference_observation,
)
from intervene3d.reproducibility.environment import environment_metadata, git_metadata
from intervene3d.reproducibility.seeds import rng_for, set_global_seed


def test_same_seed_produces_the_same_synthetic_sample(synth_config):
    a = generate_base_scene(np.random.default_rng([5, 3]), synth_config, 3)
    b = generate_base_scene(np.random.default_rng([5, 3]), synth_config, 3)
    assert np.array_equal(a.content.points, b.content.points)
    assert np.array_equal(a.content.colors, b.content.colors)
    assert np.array_equal(a.content.observer_markers_cam, b.content.observer_markers_cam)
    assert np.array_equal(a.content.reference_camera.T_wc, b.content.reference_camera.T_wc)
    assert a.base_scene_id == b.base_scene_id


def test_different_seeds_produce_different_samples(synth_config):
    a = generate_base_scene(np.random.default_rng([5, 3]), synth_config, 3)
    b = generate_base_scene(np.random.default_rng([6, 3]), synth_config, 3)
    assert not np.array_equal(a.content.points, b.content.points)


def test_same_config_produces_the_same_deterministic_result(synth_config):
    """The full oracle computation is a pure function of (config, seed)."""
    from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
    from intervene3d.models.transition import AnalyticalTransitionModel

    def compute():
        base = generate_base_scene(np.random.default_rng([11, 2]), synth_config, 2)
        hyps = build_hypothesis_set(base.content.interface, synth_config, np.random.default_rng([11, 2, 7]))
        f0 = reference_observation(base.content, hyps[0]).feature
        est = GeometrySeparabilityEstimator(
            AnalyticalTransitionModel(),
            DistanceWeights.from_dict(synth_config["identifiability"]["distance"]),
        )
        return est.pairwise_over_actions(
            f0, hyps, action_space_from_config(synth_config["action_space"]),
            markers_cam=base.content.observer_markers_cam,
        )

    assert np.array_equal(compute(), compute())


def test_rng_streams_are_independent():
    """Adding a consumer must not perturb an existing one."""
    a1 = rng_for(42, 1).normal(size=5)
    a2 = rng_for(42, 1).normal(size=5)
    b = rng_for(42, 2).normal(size=5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_set_global_seed_reports_what_it_configured():
    report = set_global_seed(7)
    assert report["seed"] == 7 and report["numpy"] and report["deterministic"]
    assert "torch" in report  # present or explicitly "not installed"


def test_config_hash_changes_with_the_configuration():
    a = validate_synthetic_config({"dataset": {"seed": 1}})
    b = validate_synthetic_config({"dataset": {"seed": 2}})
    assert config_hash(a) != config_hash(b)
    assert config_hash(a) == config_hash(validate_synthetic_config({"dataset": {"seed": 1}}))


# ------------------------------------------------------------------- splits
def test_split_assignment_is_deterministic_and_leak_free():
    fractions = {"train": 0.5, "val": 0.2, "test": 0.3}
    ids = [f"base_{i:05d}" for i in range(200)]
    first = build_splits(ids, fractions)
    second = build_splits(ids, fractions)
    assert first["assignment"] == second["assignment"]
    for name, frac in fractions.items():
        assert abs(first["realised_fractions"][name] - frac) < 0.10


def test_split_is_stable_when_the_dataset_grows():
    """Regenerating with more scenes must not move an existing scene."""
    fractions = {"train": 0.5, "val": 0.2, "test": 0.3}
    small = build_splits([f"base_{i:05d}" for i in range(20)], fractions)["assignment"]
    large = build_splits([f"base_{i:05d}" for i in range(200)], fractions)["assignment"]
    for key, value in small.items():
        assert large[key] == value


def test_all_variants_of_a_base_scene_share_a_split():
    fractions = {"train": 0.5, "val": 0.2, "test": 0.3}
    for base in ("base_00007", "base_00042"):
        splits = {assign_split(base, fractions) for _ in range(5)}
        assert len(splits) == 1


def test_leakage_detector_flags_a_split_base_scene():
    records = [
        {"base_scene_id": "b1", "split": "train"},
        {"base_scene_id": "b1", "split": "test"},
        {"base_scene_id": "b2", "split": "train"},
    ]
    assert check_no_leakage(records) == ["b1"]
    assert check_no_leakage(records[1:]) == []


def test_split_fractions_must_sum_to_one():
    import pytest

    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_split("b", {"train": 0.5, "val": 0.2, "test": 0.1})


# -------------------------------------------------------------- environment
def test_environment_metadata_is_complete():
    meta = environment_metadata()
    for key in ("python_version", "os", "hardware", "packages", "environment_variables", "git"):
        assert key in meta
    assert "numpy" in meta["packages"]
    assert "cuda_available" in meta["hardware"]


def test_git_metadata_never_raises():
    meta = git_metadata()
    assert isinstance(meta["commit"], str) and meta["commit"]


def test_no_generated_scene_ever_violates_the_matched_property(synth_config):
    """Swept across many seeds, not just the fixture's.

    A single scene whose mirror reveals itself at C_0 would let a single-frame
    classifier beat chance and invalidate the benchmark's premise, so this is
    checked exhaustively rather than on one example.
    """
    from intervene3d.data.synthetic.optical_variants import observer_reflection_visible_at_reference

    for seed in (1, 12345, 20260828, 20260829, 99):
        for b in range(8):
            base = generate_base_scene(np.random.default_rng([seed, b]), synth_config, b)
            assert not observer_reflection_visible_at_reference(base.content), (
                f"seed={seed}, base={b}: the observer's reflection is visible at C_0"
            )
            hyps = build_hypothesis_set(
                base.content.interface, synth_config, np.random.default_rng([seed, b, 7])
            )
            refs = [reference_observation(base.content, h) for h in hyps]
            f0 = refs[0].feature
            for r in refs[1:]:
                assert np.array_equal(r.feature.visible, f0.visible)
