"""Integration: the full inference loop and the dataset round trip."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.data.dataset import SyntheticDataset
from intervene3d.data.synthetic import generate_dataset, simulate, validate_dataset
from intervene3d.inference.engine import AbstentionPolicy, Intervene3DEngine
from intervene3d.inference.result import UNRESOLVED_MESSAGE
from intervene3d.models.belief import LikelihoodBeliefUpdater
from intervene3d.models.encoders import build_geometry_encoder
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.selector import build_selector


def _engine(estimator, *, epsilon=1.0, abstention=True, selector="max_separability", seed=0):
    return Intervene3DEngine(
        estimator=estimator,
        selector=build_selector(selector, estimator, seed=seed),
        belief=LikelihoodBeliefUpdater(beta=1.0),
        identifiability=EpsilonIdentifiabilityEstimator(epsilon=epsilon),
        abstention=AbstentionPolicy(enabled=abstention, tau=0.8),
    )


def test_full_loop_identifies_the_true_mechanism_when_resolvable(
    base_scene, hypotheses, actions, estimator
):
    """Scientific loop end to end, for every mechanism in the family."""
    from intervene3d.data.synthetic import reference_observation

    engine = _engine(estimator)
    markers = base_scene.content.observer_markers_cam
    for h in hypotheses:
        ref = reference_observation(base_scene.content, h)

        def observe(action, index, _h=h, _ref=ref):
            return simulate(base_scene.content, _h, action, reference_feature=_ref.feature)

        result = engine.run(f"it_{h.symbol}", ref.feature, hypotheses, actions, observe, markers_cam=markers)

        assert np.isclose(result.hypothesis_probabilities.sum(), 1.0)
        assert result.belief_trajectory.shape == (2, len(hypotheses))
        assert result.selected_action in actions.names
        assert result.contact_geometry.shape[0] == len(ref.feature)

        if result.resolvable:
            assert not result.abstained
            assert result.predicted_mechanism == h.mechanism.value, (
                f"resolvable scene misclassified: predicted {result.predicted_mechanism}, "
                f"truth {h.mechanism.value}"
            )
        else:
            assert result.abstained
            assert result.message == UNRESOLVED_MESSAGE
            assert result.predicted_mechanism == "abstain"


def test_abstention_can_be_disabled_and_then_forces_a_label(base_scene, hypotheses, actions, estimator):
    from intervene3d.data.synthetic import reference_observation

    markers = base_scene.content.observer_markers_cam
    ref = reference_observation(base_scene.content, hypotheses[0])

    def observe(action, index):
        return simulate(base_scene.content, hypotheses[0], action, reference_feature=ref.feature)

    forced = _engine(estimator, abstention=False).run(
        "forced", ref.feature, hypotheses, actions, observe, markers_cam=markers
    )
    assert not forced.abstained
    assert forced.predicted_mechanism in [h.mechanism.value for h in hypotheses]
    assert "abstention disabled" in forced.reason


def test_impossible_epsilon_forces_abstention_everywhere(base_scene, hypotheses, actions, estimator):
    """With an unreachable threshold nothing is identifiable, so nothing is claimed."""
    from intervene3d.data.synthetic import reference_observation

    ref = reference_observation(base_scene.content, hypotheses[0])

    def observe(action, index):
        return simulate(base_scene.content, hypotheses[0], action, reference_feature=ref.feature)

    result = _engine(estimator, epsilon=1e9).run(
        "impossible", ref.feature, hypotheses, actions, observe,
        markers_cam=base_scene.content.observer_markers_cam,
    )
    assert result.abstained and not result.resolvable
    assert result.identifiability_uncertainty > 0.99


def test_two_uncertainties_are_reported_separately(base_scene, hypotheses, actions, estimator):
    from intervene3d.data.synthetic import reference_observation

    ref = reference_observation(base_scene.content, hypotheses[0])

    def observe(action, index):
        return simulate(base_scene.content, hypotheses[0], action, reference_feature=ref.feature)

    result = _engine(estimator).run(
        "u", ref.feature, hypotheses, actions, observe,
        markers_cam=base_scene.content.observer_markers_cam,
    )
    assert 0.0 <= result.prediction_uncertainty <= 1.0
    assert 0.0 <= result.identifiability_uncertainty <= 1.0
    record = result.to_record()
    assert "prediction_uncertainty" in record and "identifiability_uncertainty" in record


@pytest.mark.parametrize("selector", ["max_separability", "random", "max_baseline", "entropy_nbv", "null"])
def test_every_selector_runs(selector, base_scene, hypotheses, actions, estimator):
    from intervene3d.data.synthetic import reference_observation

    ref = reference_observation(base_scene.content, hypotheses[0])

    def observe(action, index):
        return simulate(base_scene.content, hypotheses[0], action, reference_feature=ref.feature)

    engine = _engine(estimator, selector=selector)
    result = engine.run("s", ref.feature, hypotheses, actions, observe,
                        markers_cam=base_scene.content.observer_markers_cam)
    assert result.selected_action in actions.names
    if selector == "null":
        # Doing nothing yields no evidence at all: every hypothesis predicts the
        # matched reference view, so the posterior must stay exactly uniform.
        assert result.selected_action == "stay"
        assert np.allclose(result.hypothesis_probabilities, 1.0 / len(hypotheses), atol=1e-9)
    else:
        assert not actions[result.selected_action_index].is_null


def test_noisy_encoder_degrades_gracefully(base_scene, hypotheses, actions, estimator):
    """With measurement noise the loop must still produce a valid, normalised posterior."""
    from intervene3d.data.synthetic import reference_observation
    from intervene3d.data.types import Observation

    encoder = build_geometry_encoder({"name": "mock", "pixel_noise_std": 1.0, "seed": 3})
    ref = reference_observation(base_scene.content, hypotheses[1])
    feature = encoder.encode(ref)

    def observe(action, index):
        obs = simulate(base_scene.content, hypotheses[1], action, reference_feature=ref.feature)
        return Observation(encoder.encode(obs), obs.contact_depth, obs.mechanism)

    result = _engine(estimator).run(
        "noisy", feature, hypotheses, actions, observe, markers_cam=base_scene.content.observer_markers_cam
    )
    assert np.isclose(result.hypothesis_probabilities.sum(), 1.0)
    assert np.all(np.isfinite(result.hypothesis_probabilities))


def test_unimplemented_encoders_raise_with_instructions():
    for name in ("moge", "vggt_like"):
        enc = build_geometry_encoder({"name": name})
        with pytest.raises(NotImplementedError, match="NOT IMPLEMENTED"):
            enc.encode(None)
    with pytest.raises(ValueError, match="unknown geometry encoder"):
        build_geometry_encoder({"name": "telepathy"})


# ------------------------------------------------------------ dataset round trip
def test_dataset_generation_load_and_validation(tmp_path, synth_config):
    cfg = dict(synth_config)
    cfg["dataset"] = {**cfg["dataset"], "name": "it_roundtrip", "num_base_scenes": 6}
    manifest = generate_dataset(cfg, output_root=tmp_path)
    root = tmp_path / "it_roundtrip"

    assert manifest["statistics"]["n_scene_variants"] == 6 * len(cfg["mechanisms"])
    assert manifest["config_hash"] and manifest["files"]

    report = validate_dataset(root)
    assert report.passed, report.render()

    ds = SyntheticDataset(root)
    assert len(ds) == manifest["statistics"]["n_scene_variants"]
    assert set(s.split for s in ds) <= {"train", "val", "test"}

    scene = ds.scenes("all")[0]
    ref = scene.reference_observation()
    assert ref.feature.uv.shape[0] == scene.record["n_landmarks"]
    hyps = scene.hypothesis_set()
    assert len(hyps) == len(cfg["mechanisms"])
    assert hyps[scene.true_index].mechanism.value == scene.mechanism

    # Stored observations must match a fresh simulation exactly.
    from intervene3d.data.types import SceneContent

    arrays = scene.arrays()
    content = SceneContent(
        points=arrays["content_points"], colors=arrays["content_colors"],
        reference_camera=scene.camera, interface=scene.interface,
        observer_markers_cam=arrays["markers_cam"],
    )
    idx = len(ds.action_space) // 2
    fresh = simulate(content, hyps[scene.true_index], ds.action_space[idx], reference_feature=ref.feature)
    stored = scene.observation_with_camera(idx, scene.camera.moved(ds.action_space[idx].delta_T()))
    m = stored.feature.visible & fresh.feature.visible
    assert np.array_equal(stored.feature.visible, fresh.feature.visible)
    assert np.allclose(stored.feature.uv[m], fresh.feature.uv[m], atol=1e-12)


def test_dataset_regeneration_is_bit_identical(tmp_path, synth_config):
    cfg = dict(synth_config)
    cfg["dataset"] = {**cfg["dataset"], "name": "it_determinism", "num_base_scenes": 3}
    a = generate_dataset(cfg, output_root=tmp_path / "a")
    b = generate_dataset(cfg, output_root=tmp_path / "b")
    assert a["files"] == b["files"], "regenerating with the same seed must reproduce identical files"


def test_validator_detects_a_corrupted_file(tmp_path, synth_config):
    cfg = dict(synth_config)
    cfg["dataset"] = {**cfg["dataset"], "name": "it_corrupt", "num_base_scenes": 3}
    generate_dataset(cfg, output_root=tmp_path)
    root = tmp_path / "it_corrupt"
    (root / "scenes.jsonl").write_text((root / "scenes.jsonl").read_text() + "\n")
    report = validate_dataset(root)
    checks = {c["check"]: c for c in report.checks}
    assert not checks["checksums"]["ok"]
    assert not report.passed
