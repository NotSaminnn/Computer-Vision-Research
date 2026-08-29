r"""Phase 1 -- the problem-existence experiment.

Research question
-----------------
Is the proposed problem experimentally *observable*?  Concretely, does

.. math::
    \text{single-frame} < \text{passive multi-view} < \text{intervention-aware}

hold on appearance-matched causal variants?

The objective is **not** impressive numbers.  A benchmark whose variants are
pixel-identical at the reference view makes the single-frame result a
near-certainty (chance), and an oracle encoder paired with analytical optics
that match the simulator makes the intervention-aware result near-perfect.  Both
facts are stated in the run summary so no reader mistakes an upper bound for a
scientific claim.  The informative quantities are the *gaps* between the tiers,
the behaviour on non-identifiable cases (FPCR), and the noisy-encoder condition.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.config.loader import ConfigError, load_config
from intervene3d.config.schema import validate_synthetic_config
from intervene3d.data.dataset import Scene, SyntheticDataset
from intervene3d.data.synthetic.dataset_writer import generate_dataset
from intervene3d.data.synthetic.trajectory_generator import mcrb_baseline_sweep
from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature, Observation
from intervene3d.experiments.learned import (
    BASE_FOR_TRANSITION,
    required_bases,
    train_residual_transition,
)
from intervene3d.experiments.methods import MethodSpec, build_engine
from intervene3d.hypotheses.base import HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.metrics.classification import (
    causal_explanation_accuracy,
    confusion_matrix,
    false_physical_certainty_rate,
)
from intervene3d.metrics.depth import contact_depth_metrics
from intervene3d.metrics.identifiability import binary_decision_metrics, identifiability_auroc
from intervene3d.metrics.mcrb import mcrb_numeric
from intervene3d.metrics.regret import intervention_regret
from intervene3d.models.baselines import reference_features, response_features, train_baseline
from intervene3d.models.encoders import build_geometry_encoder
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import AnalyticalTransitionModel
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ------------------------------------------------------------------ dataset
def load_or_generate_dataset(config: Mapping[str, Any], seed: int | None = None) -> SyntheticDataset:
    """Attach an existing dataset, or generate one from the referenced config.

    ``data.reseed_with_experiment_seed`` selects the multi-seed protocol:

    ``false`` (default)
        One fixed benchmark; the experiment seed varies only model-side
        stochasticity.  Deterministic methods then show zero variance across
        seeds, which is a determinism check rather than an error bar.
    ``true``
        The benchmark itself is **redrawn** for each experiment seed, so the
        aggregate confidence intervals measure variation over the data-generating
        process.  This is the protocol whose error bars are worth reporting.
    """
    data_cfg = config["data"]
    if data_cfg.get("dataset_dir"):
        return SyntheticDataset(data_cfg["dataset_dir"])
    synth_path = data_cfg.get("synthetic_config")
    if not synth_path:
        raise ConfigError("data.dataset_dir or data.synthetic_config must be set")
    synth_cfg = validate_synthetic_config(load_config(synth_path))
    if data_cfg.get("reseed_with_experiment_seed") and seed is not None:
        synth_cfg["dataset"] = {
            **synth_cfg["dataset"],
            "seed": int(synth_cfg["dataset"]["seed"]) + int(seed),
            "name": f"{synth_cfg['dataset']['name']}_s{int(seed)}",
        }
        LOGGER.info(
            "reseeding the benchmark for experiment seed %d -> dataset %s (seed %d)",
            seed, synth_cfg["dataset"]["name"], synth_cfg["dataset"]["seed"],
        )
    root = Path(synth_cfg["output"]["root"]) / synth_cfg["dataset"]["name"]
    if (root / "manifest.json").exists():
        LOGGER.info("using existing dataset at %s", root)
        return SyntheticDataset(root)
    LOGGER.info("dataset not found at %s -- generating it now", root)
    generate_dataset(synth_cfg)
    return SyntheticDataset(root)


# ------------------------------------------------------- per-scene evaluation
def _observe_factory(scene: Scene, actions: ActionSpace, encoder, noise_rng: np.random.Generator | None,
                     action_noise: Mapping[str, Any]):
    """Build the ``observe(action, index)`` callback for one scene.

    Observations come from the dataset's pre-simulated arrays, so an experiment
    can never silently depend on a changed simulator.  When *action-execution
    noise* is enabled the executed pose differs from the commanded one and the
    observation is re-simulated for the perturbed action -- the generalisation
    condition ``Delta C + epsilon``.
    """
    ref_cam = scene.camera

    def observe(action, index: int) -> Observation:
        if action_noise.get("enabled") and noise_rng is not None:
            from intervene3d.data.synthetic.optical_variants import simulate
            from intervene3d.data.types import SceneContent

            perturbed = action.perturbed(
                noise_rng,
                translation_std=float(action_noise.get("translation_std", 0.0)),
                rotation_std=np.radians(float(action_noise.get("rotation_std_deg", 0.0))),
            )
            arrays = scene.arrays()
            content = SceneContent(
                points=arrays["content_points"], colors=arrays["content_colors"],
                reference_camera=ref_cam, interface=scene.interface,
                observer_markers_cam=arrays["markers_cam"],
            )
            hypotheses = scene.hypothesis_set()
            obs = simulate(content, hypotheses[scene.true_index], perturbed,
                           reference_feature=scene.reference_observation().feature)
            return obs
        cam = ref_cam.moved(actions[index].delta_T())
        return scene.observation_with_camera(index, cam)

    return observe


def _encoded(encoder, observation: Observation) -> GeometryFeature:
    return encoder.encode(observation)


def _predicted_mcrb(
    estimator: GeometrySeparabilityEstimator,
    feature: GeometryFeature,
    hypotheses: HypothesisSet,
    beliefs: np.ndarray,
    markers_cam: np.ndarray,
    baselines: np.ndarray,
    lateral_space: ActionSpace,
    epsilon: float,
) -> float | None:
    """The model's own estimate of the minimum lateral resolving baseline.

    Computed for the top-two hypotheses under the current posterior -- the pair
    the system would actually still have to separate.
    """
    if len(hypotheses) < 2:
        return None
    order = np.argsort(-np.asarray(beliefs))
    i, j = int(order[0]), int(order[1])
    lat = estimator.pairwise_over_actions(feature, hypotheses, lateral_space, markers_cam=markers_cam)
    return mcrb_numeric(baselines, lat[:, i, j], epsilon).value


def evaluate_engine_method(
    spec: MethodSpec,
    scenes: Sequence[Scene],
    actions: ActionSpace,
    model_config: Mapping[str, Any],
    *,
    seed: int,
    baselines: np.ndarray,
    lateral_space: ActionSpace,
    action_noise: Mapping[str, Any],
    collect_examples: int = 1,
    learned_model=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the inference loop over every scene and collect per-scene records."""
    engine, encoder = build_engine(spec, model_config, seed=seed, learned_model=learned_model)
    epsilon = engine.identifiability.epsilon
    noise_rng = np.random.default_rng([seed, 991]) if action_noise.get("enabled") else None

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for scene in scenes:
        reference = scene.reference_observation()
        feature = _encoded(encoder, reference)
        hypotheses = scene.hypothesis_set()
        markers = scene.markers_cam
        observe = _observe_factory(scene, actions, encoder, noise_rng, action_noise)

        def observe_encoded(action, index, _observe=observe):
            obs = _observe(action, index)
            return Observation(_encoded(encoder, obs), obs.contact_depth, obs.mechanism)

        result = engine.run(
            scene.scene_id, feature, hypotheses, actions, observe_encoded, markers_cam=markers
        )

        oracle_utility = scene.oracle_utility()
        regret = intervention_regret(oracle_utility, result.selected_action_index)

        content_mask = (reference.feature.channel == CHANNEL_CONTENT) & reference.feature.visible
        contact = contact_depth_metrics(result.contact_geometry, reference.contact_depth, content_mask)

        pred_mcrb = _predicted_mcrb(
            engine.estimator, feature, hypotheses, result.hypothesis_probabilities,
            markers, baselines, lateral_space, epsilon,
        )

        rows.append(
            {
                "method": spec.name,
                "true_mechanism": scene.mechanism,
                "base_scene_id": scene.base_scene_id,
                "split": scene.split,
                "resolvable_gt": bool(scene.resolvable),
                "mcrb_gt": scene.record.get("mcrb"),
                "mcrb_analytic": scene.record.get("mcrb_analytic"),
                "oracle_best_action": scene.record.get("oracle_best_action"),
                "identifiability_score_gt": scene.record.get("identifiability_score"),
                "display_mode": scene.display_mode,
                **result.to_record(),
                "predicted_mcrb": pred_mcrb,
                **{f"regret_{k}": v for k, v in regret.items()},
                **contact,
            }
        )

        if len(examples) < collect_examples:
            examples.append(_build_example(scene, result, actions, engine, feature, hypotheses, markers))

    return rows, {"examples": examples, "spec": spec.to_dict(), "encoder": getattr(encoder, "to_dict", lambda: {"name": encoder.name})()}


def _build_example(scene, result, actions, engine, feature, hypotheses, markers) -> dict[str, Any]:
    """Rich per-scene payload used by the figure generator."""
    action = actions[result.selected_action_index]
    predictions = engine.estimator.predict_all(feature, hypotheses, action, markers_cam=markers)
    observed = scene.observation_with_camera(
        result.selected_action_index, scene.camera.moved(action.delta_T())
    ).feature
    return {
        "scene_id": scene.scene_id,
        "true_mechanism": scene.mechanism,
        "mechanisms": [h.mechanism.value for h in hypotheses],
        "identifiability_matrix": result.identifiability_matrix.tolist(),
        "belief_trajectory": result.belief_trajectory.tolist(),
        "abstained": bool(result.abstained),
        "action_names": actions.names,
        "utility": result.action_utility.tolist(),
        "selected_index": int(result.selected_action_index),
        "oracle_index": int(np.argmax(scene.oracle_utility())),
        "candidate_translations": [a.translation.tolist() for a in actions],
        "executed_translations": [actions[result.selected_action_index].translation.tolist()],
        "reference_uv": feature.uv.tolist(),
        "observed_uv": observed.uv.tolist(),
        "reference_channel": feature.channel.tolist(),
        "reference_visible": feature.visible.tolist(),
        "observed_visible": observed.visible.tolist(),
        "predictions": [
            {"label": h.mechanism.value, "uv": p.uv.tolist()} for h, p in zip(hypotheses, predictions, strict=True)
        ],
    }


def evaluate_classifier_method(
    spec: MethodSpec,
    train_scenes: Sequence[Scene],
    eval_scenes: Sequence[Scene],
    actions: ActionSpace,
    mechanisms: Sequence[str],
    *,
    seed: int,
    fixed_action: str,
    epochs: int = 400,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train and evaluate a discriminative control baseline."""
    encoder = build_geometry_encoder({**spec.encoder, "seed": seed})
    uses_response = spec.classifier == "passive_multiview"
    action_index = actions.index_of(fixed_action) if uses_response else -1

    def featurise(scene: Scene) -> np.ndarray:
        ref = _encoded(encoder, scene.reference_observation())
        base = reference_features(ref)
        if not uses_response:
            return base
        cam = scene.camera.moved(actions[action_index].delta_T())
        obs = _encoded(encoder, scene.observation_with_camera(action_index, cam))
        return np.concatenate([base, response_features(ref, obs)])

    label_of = {m: i for i, m in enumerate(mechanisms)}
    x_train = np.stack([featurise(s) for s in train_scenes])
    y_train = np.array([label_of[s.mechanism] for s in train_scenes], dtype=int)
    model = train_baseline(
        spec.name, x_train, y_train, mechanisms, uses_response=uses_response,
        fixed_action=fixed_action if uses_response else None, seed=seed, epochs=epochs,
    )

    x_eval = np.stack([featurise(s) for s in eval_scenes])
    probs = model.predict_proba(x_eval)

    rows: list[dict[str, Any]] = []
    for scene, p in zip(eval_scenes, probs, strict=True):
        k = int(np.argmax(p))
        rows.append(
            {
                "method": spec.name,
                "scene_id": scene.scene_id,
                "true_mechanism": scene.mechanism,
                "base_scene_id": scene.base_scene_id,
                "split": scene.split,
                "resolvable_gt": bool(scene.resolvable),
                "mcrb_gt": scene.record.get("mcrb"),
                "display_mode": scene.display_mode,
                "predicted_mechanism": mechanisms[k],
                "committed_mechanism": mechanisms[k],
                "max_probability": float(np.max(p)),
                "abstained": False,
                "reason": "discriminative baseline: forced choice, no abstention mechanism",
                # A confidence proxy: these baselines have no notion of identifiability,
                # which is exactly what H2 predicts should make them worse at it.
                "identifiability_score": float(np.max(p)),
                "resolvable_pred": True,
                "selected_action": fixed_action if uses_response else "stay",
                "prediction_uncertainty": float(-np.sum(p * np.log(np.clip(p, 1e-12, None))) / np.log(len(p))),
                "identifiability_uncertainty": float(1.0 - np.max(p)),
                "motion_cost": float(actions[action_index].translation_magnitude) if uses_response else 0.0,
                "steps": 1 if uses_response else 0,
                "predicted_mcrb": None,
                **{f"p_{m}": float(v) for m, v in zip(mechanisms, p, strict=True)},
            }
        )
    return rows, {"spec": spec.to_dict(), "model": model.to_dict()}


# --------------------------------------------------------------------- metrics
def compute_method_metrics(
    rows: Sequence[Mapping[str, Any]], mechanisms: Sequence[str], *, tau: float
) -> dict[str, Any]:
    """All headline metrics for one method."""
    predicted = [str(r["predicted_mechanism"]) for r in rows]
    truth = [str(r["true_mechanism"]) for r in rows]
    abstained = [bool(r["abstained"]) for r in rows]

    cea = causal_explanation_accuracy(predicted, truth, abstained=abstained)
    max_p = [float(r["max_probability"]) for r in rows]
    resolvable_gt = [bool(r["resolvable_gt"]) for r in rows]
    fpcr = false_physical_certainty_rate(max_p, resolvable_gt, tau=tau, abstained=abstained)
    # A single tau can be uninformative: with K hypotheses and one genuinely
    # unresolvable pair the posterior saturates near 1/2, so FPCR at tau=0.8 can be
    # zero for reasons that have nothing to do with the abstention mechanism.
    # Reporting a sweep makes that visible instead of hiding it.
    fpcr["sweep"] = {
        f"{t:.2f}": false_physical_certainty_rate(max_p, resolvable_gt, tau=t, abstained=abstained)["fpcr"]
        for t in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    }
    auroc = identifiability_auroc(
        [float(r["identifiability_score"]) for r in rows], [bool(r["resolvable_gt"]) for r in rows]
    )
    decision = binary_decision_metrics(
        [bool(r["resolvable_pred"]) for r in rows], [bool(r["resolvable_gt"]) for r in rows]
    )

    absrel = [float(r["abs_rel_contact"]) for r in rows if "abs_rel_contact" in r]
    rmse = [float(r["rmse_contact"]) for r in rows if "rmse_contact" in r]
    regret = [float(r["regret_normalised_regret"]) for r in rows if "regret_normalised_regret" in r]
    motion = [float(r.get("motion_cost", np.nan)) for r in rows]

    mcrb_errors = [
        abs(float(r["predicted_mcrb"]) - float(r["mcrb_gt"]))
        for r in rows
        if r.get("predicted_mcrb") is not None and r.get("mcrb_gt") is not None
    ]

    return {
        "n": len(rows),
        "cea": cea,
        "fpcr": fpcr,
        "identifiability": {**auroc, **decision},
        "contact_depth": {
            "abs_rel_contact": float(np.nanmean(absrel)) if absrel else float("nan"),
            "rmse_contact": float(np.nanmean(rmse)) if rmse else float("nan"),
            "n_scored": len(absrel),
        },
        "intervention": {
            "normalised_regret": float(np.nanmean(regret)) if regret else float("nan"),
            "motion_cost": float(np.nanmean(motion)) if motion else float("nan"),
        },
        "mcrb": {
            "mae": float(np.mean(mcrb_errors)) if mcrb_errors else float("nan"),
            "n_scored": len(mcrb_errors),
            "note": (
                "NOT SCORED: this method does not predict a resolving baseline"
                if not mcrb_errors
                else "predicted from the model's own separability sweep"
            ),
        },
        "confusion_matrix": confusion_matrix(predicted, truth, list(mechanisms) + ["abstain"]).tolist(),
        "confusion_labels": list(mechanisms) + ["abstain"],
    }


# ------------------------------------------------------- figure input assembly
def _initial_view_data(
    scenes: Sequence[Scene], actions: ActionSpace, hypotheses: HypothesisSet, epsilon: float
) -> dict[str, Any]:
    """Oracle separability at ``C_0`` versus after the best allowed action."""
    null_idx = next((i for i, a in enumerate(actions) if a.is_null), 0)
    pairs = hypotheses.pairs()
    ref_vals = {p: [] for p in pairs}
    best_vals = {p: [] for p in pairs}
    for scene in scenes:
        sep = scene.oracle_separability()
        for i, j in pairs:
            ref_vals[(i, j)].append(float(sep[null_idx, i, j]))
            best_vals[(i, j)].append(float(np.max(sep[:, i, j])))
    labels = [f"{hypotheses[i].symbol}/{hypotheses[j].symbol}" for i, j in pairs]
    return {
        "pairs": labels,
        "reference_distance_px": [float(np.mean(ref_vals[p])) for p in pairs],
        "best_action_distance_px": [float(np.mean(best_vals[p])) for p in pairs],
        "epsilon_px": float(epsilon),
        "n_scenes": len(scenes),
    }


def _baseline_curve_data(
    scenes: Sequence[Scene],
    hypotheses: HypothesisSet,
    estimator: GeometrySeparabilityEstimator,
    baselines: np.ndarray,
    lateral_space: ActionSpace,
    epsilon: float,
    *,
    max_scenes: int = 24,
) -> dict[str, Any]:
    """Mean separability-versus-lateral-baseline curve for every hypothesis pair."""
    pairs = hypotheses.pairs()
    acc = {p: [] for p in pairs}
    for scene in scenes[:max_scenes]:
        feature = scene.reference_observation().feature
        hyps = scene.hypothesis_set()
        lat = estimator.pairwise_over_actions(
            feature, hyps, lateral_space, markers_cam=scene.markers_cam
        )
        for i, j in pairs:
            acc[(i, j)].append(lat[:, i, j])
    curves = []
    for i, j in pairs:
        if not acc[(i, j)]:
            continue
        mean_curve = np.mean(np.stack(acc[(i, j)]), axis=0)
        mcrb = mcrb_numeric(baselines, mean_curve, epsilon)
        curves.append(
            {
                "label": f"{hypotheses[i].symbol} vs {hypotheses[j].symbol}",
                "separability": mean_curve.tolist(),
                "mcrb": mcrb.value,
            }
        )
    return {
        "baselines": np.asarray(baselines).tolist(),
        "curves": curves,
        "epsilon_px": float(epsilon),
        "n_scenes": min(len(scenes), max_scenes),
    }


def _theory_validation(scenes: Sequence[Scene]) -> dict[str, Any]:
    r"""Does the measured resolving baseline obey the MCRB law?

    The derivation predicts ``B_min = delta / (f |1/Z_1 - 1/Z_2|)``, i.e.
    ``1/B_min`` linear in ``f |1/Z_1 - 1/Z_2|``.  The measurement is the
    *homography-compensated* resolving baseline, because compensating the screen
    plane's projective warp is exactly the assumption the derivation makes.  The
    fitted slope is not expected to be ``f/delta``: the theory uses the extremal
    depth pair while the measurement is an RMS residual over all landmarks, so
    the informative quantity is the **linearity** (``R^2``), not the constant.
    """
    x, y, scene_ids = [], [], []
    for scene in scenes:
        if scene.mechanism != "emissive" or scene.display_mode != "static":
            continue
        b = scene.record.get("mcrb_compensated")
        z_near, z_far = scene.record.get("z_near"), scene.record.get("z_far")
        if not b or z_near is None or z_far is None:
            continue
        if not (np.isfinite(z_near) and np.isfinite(z_far)) or abs(1 / z_near - 1 / z_far) < 1e-9:
            continue
        x.append(scene.camera.intrinsics.fx * abs(1.0 / z_near - 1.0 / z_far))
        y.append(1.0 / float(b))
        scene_ids.append(scene.scene_id)
    if len(x) < 3:
        return {
            "f_inv_depth_difference": x, "inverse_measured_baseline": y,
            "slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan"),
            "n": len(x), "status": "NOT SCORED: fewer than 3 applicable static-display scenes",
        }
    xa, ya = np.asarray(x), np.asarray(y)
    A = np.vstack([xa, np.ones_like(xa)]).T
    coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ya - pred) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    return {
        "f_inv_depth_difference": xa.tolist(),
        "inverse_measured_baseline": ya.tolist(),
        "scene_ids": scene_ids,
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": len(x),
        "measurement": "homography-compensated lateral resolving baseline",
        "status": "scored",
    }


def _matched_strip_data(
    scenes: Sequence[Scene], actions: ActionSpace, oracle_action: str | None
) -> dict[str, Any]:
    """One base scene shown across its causal variants, before and after one action."""
    by_base: dict[str, list[Scene]] = {}
    for scene in scenes:
        by_base.setdefault(scene.base_scene_id, []).append(scene)
    group = next((v for v in by_base.values() if len(v) >= 2), None)
    if group is None:
        return {}
    action_name = oracle_action or group[0].record.get("oracle_best_action") or actions.non_null()[0].name
    try:
        idx = actions.index_of(action_name)
    except KeyError:
        idx = actions.index_of(actions.non_null()[0].name)
    variants = []
    for scene in sorted(group, key=lambda s: s.mechanism):
        ref = scene.reference_observation().feature
        after = scene.observation_with_camera(idx, scene.camera.moved(actions[idx].delta_T())).feature
        variants.append(
            {
                "label": scene.mechanism,
                "reference": {"uv": ref.uv.tolist(), "visible": ref.visible.tolist(), "channel": ref.channel.tolist()},
                "after": {"uv": after.uv.tolist(), "visible": after.visible.tolist(), "channel": after.channel.tolist()},
            }
        )
    intr = group[0].camera.intrinsics
    return {"variants": variants, "image_width": intr.width, "image_height": intr.height,
            "action": actions[idx].name}


def _contact_vs_apparent_data(scenes: Sequence[Scene], *, per_mechanism: int = 4) -> dict[str, Any]:
    """Apparent versus contact depth, a few scenes per mechanism."""
    taken: dict[str, int] = {}
    entries = []
    max_depth = 1.0
    for scene in scenes:
        if taken.get(scene.mechanism, 0) >= per_mechanism:
            continue
        taken[scene.mechanism] = taken.get(scene.mechanism, 0) + 1
        obs = scene.reference_observation()
        m = (obs.feature.channel == CHANNEL_CONTENT) & obs.feature.visible
        apparent = obs.feature.depth[m]
        contact = obs.contact_depth[m]
        finite = np.isfinite(apparent) & np.isfinite(contact)
        if not np.any(finite):
            continue
        max_depth = max(max_depth, float(np.nanmax(apparent[finite])), float(np.nanmax(contact[finite])))
        entries.append(
            {"mechanism": scene.mechanism, "apparent_depth": apparent[finite].tolist(),
             "contact_depth": contact[finite].tolist()}
        )
    return {"scenes": entries, "max_depth": max_depth * 1.05}


# ------------------------------------------------- learned transition (Gate 7)
def _residual_key(spec: MethodSpec) -> str:
    """The trained-model key a method needs, or ``""`` if it needs none."""
    base = BASE_FOR_TRANSITION.get(spec.transition)
    if base is None:
        return ""
    return f"{base}:{int(bool(spec.hypothesis_conditioning))}"


def _train_learned_transitions(
    specs: Sequence[MethodSpec],
    train_scenes: Sequence[Scene],
    actions: ActionSpace,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one residual model per (base, conditioning) a method asks for.

    Returns ``({key: model}, {key: report})``.  Both are empty when no method
    uses a learned transition, so the ordinary analytical run pays nothing.
    """
    needed = required_bases(specs)
    if not needed:
        return {}, {}
    if config.get("experiment", {}).get("train_learned_transition") is False:
        raise ConfigError(
            "methods request a learned transition ("
            + ", ".join(sorted({s.transition for s in specs if _residual_key(s)}))
            + ") but experiment.train_learned_transition is false; "
            "enable it or drop those methods"
        )
    cfg = config.get("learned_transition", {})
    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for key, conditioning in sorted(needed.items()):
        base = key.split(":")[0]
        LOGGER.info(
            "training the %s residual transition on %d train scenes (hypothesis_conditioning=%s)",
            base, len(train_scenes), conditioning,
        )
        model, report = train_residual_transition(
            train_scenes, actions, base=base, seed=seed, config=cfg,
            hypothesis_conditioning=conditioning,
        )
        models[key] = model
        reports[key] = report
    return models, reports


# ---------------------------------------------------------------------- driver
def run(config: Mapping[str, Any], run_dir, seed: int) -> dict[str, Any]:
    """Execute the Phase 1 experiment and write every artefact into ``run_dir``."""
    from intervene3d.experiments.figures import build_figure_data, generate_figures
    from intervene3d.visualization.ieee_style import style_provenance

    started = time.time()
    dataset = load_or_generate_dataset(config, seed)
    actions = dataset.action_space
    model_cfg = config["model"]
    epsilon = float(model_cfg["identifiability"]["epsilon_px"])
    tau = float(model_cfg["abstention"]["tau"])
    action_noise = config.get("action_noise", {"enabled": False})

    split = config["data"]["split"]
    eval_scenes = dataset.scenes(split)
    train_scenes = dataset.scenes("train")
    if not eval_scenes:
        raise ValueError(f"no scenes in split {split!r}; check the dataset splits")
    if not train_scenes:
        raise ValueError("no training scenes: the discriminative baselines need a train split")

    mechanisms = list(dict.fromkeys(s.mechanism for s in dataset))
    hypotheses = eval_scenes[0].hypothesis_set()
    LOGGER.info(
        "evaluating on split=%s: %d scenes (%d base scenes), %d train scenes, |A|=%d",
        split, len(eval_scenes), len({s.base_scene_id for s in eval_scenes}), len(train_scenes), len(actions),
    )

    fixed_action = str(config.get("passive_action") or _default_passive_action(actions))
    baselines, lateral_space = mcrb_baseline_sweep(
        config.get("mcrb", {"num_baseline_samples": 41, "max_baseline": 0.35})
    )

    specs = [MethodSpec.from_dict(m, config) for m in config["methods"]]

    # Gate 7: any method asking for a learned transition needs a model trained
    # first, on the TRAIN split only. Nothing runs when no method asks.
    learned_models, learned_reports = _train_learned_transitions(
        specs, train_scenes, actions, config, seed
    )

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    method_details: dict[str, Any] = {}
    examples: dict[str, Any] = {}

    for spec in specs:
        t0 = time.time()
        if spec.kind == "classifier":
            rows, detail = evaluate_classifier_method(
                spec, train_scenes, eval_scenes, actions, mechanisms,
                seed=seed, fixed_action=spec.fixed_action or fixed_action,
                epochs=int(config.get("classifier_epochs", 400)),
            )
        else:
            rows, detail = evaluate_engine_method(
                spec, eval_scenes, actions, model_cfg, seed=seed,
                baselines=baselines, lateral_space=lateral_space, action_noise=action_noise,
                learned_model=learned_models.get(_residual_key(spec)),
            )
            if detail["examples"] and "primary" not in examples:
                examples["primary"] = detail["examples"][0]
            if detail["examples"]:
                examples[spec.name] = detail["examples"][0]
        rows_by_method[spec.name] = rows
        method_details[spec.name] = {k: v for k, v in detail.items() if k != "examples"}
        LOGGER.info("  method %-28s : %d scenes in %.2f s", spec.name, len(rows), time.time() - t0)

    metrics: dict[str, Any] = {
        "experiment": config["experiment"]["name"],
        "seed": int(seed),
        "split": split,
        "epsilon_px": epsilon,
        "tau": tau,
        "chance_level": 1.0 / len(mechanisms),
        # Empty unless a method requested a learned transition. Recording it
        # means a hybrid/learned_only result can never be quoted without its
        # training provenance alongside.
        "learned_transition_training": learned_reports,
        "mechanisms": mechanisms,
        "n_eval_scenes": len(eval_scenes),
        "n_train_scenes": len(train_scenes),
        "n_eval_base_scenes": len({s.base_scene_id for s in eval_scenes}),
        "resolvable_fraction_eval": float(np.mean([s.resolvable for s in eval_scenes])),
        "action_space_size": len(actions),
        "passive_action": fixed_action,
        "action_noise": dict(action_noise),
        "dataset": dataset.manifest_summary(),
        "distance_weights": DistanceWeights.from_dict(model_cfg["distance"]).to_dict(),
        "style": style_provenance(),
        "methods": {
            name: compute_method_metrics(rows, mechanisms, tau=tau) for name, rows in rows_by_method.items()
        },
        "method_specs": {s.name: s.to_dict() for s in specs},
        "method_details": method_details,
    }

    oracle_estimator = GeometrySeparabilityEstimator(
        AnalyticalTransitionModel(), DistanceWeights.from_dict(model_cfg["distance"])
    )
    theory = _theory_validation(eval_scenes)
    metrics["theory_validation"] = {k: v for k, v in theory.items() if not isinstance(v, list)}

    figure_data = build_figure_data(
        rows_by_method=rows_by_method,
        metrics=metrics,
        mechanisms=mechanisms,
        epsilon=epsilon,
        tau=tau,
        examples=examples,
        initial_view=_initial_view_data(eval_scenes, actions, hypotheses, epsilon),
        baseline_curves=_baseline_curve_data(
            eval_scenes, hypotheses, oracle_estimator, baselines, lateral_space, epsilon
        ),
        theory=theory,
        matched_strip=_matched_strip_data(eval_scenes, actions, None),
        contact_vs_apparent=_contact_vs_apparent_data(eval_scenes),
    )
    intr = eval_scenes[0].camera.intrinsics
    figure_data["image_size"] = {"image_width": intr.width, "image_height": intr.height}

    # ---- artefacts -------------------------------------------------------
    all_rows = [r for rows in rows_by_method.values() for r in rows]
    write_csv(run_dir.predictions / "predictions.csv", all_rows)
    dump_json(run_dir.metrics / "metrics.json", metrics)
    dump_json(run_dir.metrics / "figure_data.json", figure_data)

    tables = _write_tables(run_dir, metrics, mechanisms)
    vis_cfg = config.get("visualization", {})
    figures = generate_figures(
        figure_data, run_dir.figures,
        formats=vis_cfg.get("formats", ["pdf", "png"]), dpi=int(vis_cfg.get("dpi", 400)),
    )

    summary = _build_summary(metrics, figure_data, elapsed=time.time() - started)
    dump_json(run_dir.metrics / "summary.json", summary)
    (run_dir.path / "summary.md").write_text(_render_summary_md(metrics, summary, figures, tables), encoding="utf-8")

    return {
        "summary": summary,
        "metrics_file": str((run_dir.metrics / "metrics.json").relative_to(run_dir.path)),
        "figures": [str(Path(f).relative_to(run_dir.path)) for f in figures],
        "tables": [str(Path(t).relative_to(run_dir.path)) for t in tables],
    }


def _default_passive_action(actions: ActionSpace) -> str:
    """A mid-size lateral translation, chosen without any hypothesis reasoning."""
    lateral = [a for a in actions if a.kind.value == "translate_x" and a.translation[0] > 0]
    if lateral:
        return sorted(lateral, key=lambda a: a.translation_magnitude)[len(lateral) // 2].name
    return actions.non_null()[0].name


def _write_tables(run_dir, metrics: Mapping[str, Any], mechanisms: Sequence[str]) -> list[str]:
    """Machine-readable CSV plus a human-readable Markdown table."""
    rows = []
    for name, m in metrics["methods"].items():
        rows.append(
            {
                "method": name,
                "n": m["n"],
                "CEA": m["cea"]["cea_all"],
                "CEA_committed": m["cea"]["cea_committed"],
                "abstention_rate": m["cea"]["abstention_rate"],
                "identifiability_auroc": m["identifiability"]["identifiability_auroc"],
                "resolvability_accuracy": m["identifiability"]["resolvability_accuracy"],
                "FPCR": m["fpcr"]["fpcr"],
                "AbsRel_contact": m["contact_depth"]["abs_rel_contact"],
                "RMSE_contact": m["contact_depth"]["rmse_contact"],
                "MAE_MCRB": m["mcrb"]["mae"],
                "normalised_regret": m["intervention"]["normalised_regret"],
                "motion_cost": m["intervention"]["motion_cost"],
                **{f"CEA_{mech}": m["cea"]["by_mechanism"].get(mech, {}).get("cea_all", float("nan"))
                   for mech in mechanisms},
            }
        )
    csv_path = write_csv(run_dir.tables / "method_comparison.csv", rows)

    cols = list(rows[0]) if rows else []
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append(
            "| " + " | ".join(
                (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else ("n/a" if isinstance(v, float) else str(v)))
                for v in (r[c] for c in cols)
            ) + " |"
        )
    md_path = run_dir.tables / "method_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(csv_path), str(md_path)]


def _build_summary(metrics: Mapping[str, Any], figure_data: Mapping[str, Any], *, elapsed: float) -> dict[str, Any]:
    """The compact, machine-readable verdict for this run."""
    per = metrics["methods"]
    names = list(per)

    def _cea(name: str) -> float:
        return float(per[name]["cea"]["cea_all"])

    headline = next((n for n in names if n == "intervene3d"), None) or next(
        (n for n in names if n.startswith("intervene3d") and "no_" not in n), None
    )
    tiers = {
        "single_frame": next((n for n in names if "single_frame" in n), None),
        "passive_multiview": next((n for n in names if "passive" in n), None),
        "intervention_aware": headline,
        # The discriminative controls are forced-choice, so the like-for-like
        # comparison uses the forced-choice ablation of the full method.  CEA_all
        # counts an abstention as incorrect, which would otherwise penalise the
        # only method capable of abstaining for doing the right thing.
        "intervention_aware_forced_choice": next((n for n in names if "no_abstention" in n), None),
    }
    ordering = None
    if tiers["single_frame"] and tiers["passive_multiview"] and headline:
        forced = tiers["intervention_aware_forced_choice"] or headline
        ordering = {
            "comparison_basis": (
                "CEA with abstention counted as incorrect; the intervention-aware tier uses the "
                f"forced-choice variant {forced!r} so all three tiers are forced-choice"
            ),
            "single_frame_cea": _cea(tiers["single_frame"]),
            "passive_multiview_cea": _cea(tiers["passive_multiview"]),
            "intervention_aware_cea": _cea(forced),
            "intervention_aware_with_abstention_cea_all": _cea(headline),
            "intervention_aware_with_abstention_cea_committed": float(per[headline]["cea"]["cea_committed"]),
            "intervention_aware_abstention_rate": float(per[headline]["cea"]["abstention_rate"]),
        }
        ordering["hypothesis_supported"] = bool(
            ordering["single_frame_cea"] <= ordering["passive_multiview_cea"] + 1e-9
            <= ordering["intervention_aware_cea"] + 1e-9
        )
        ordering["strictly_increasing"] = bool(
            ordering["single_frame_cea"] < ordering["passive_multiview_cea"] < ordering["intervention_aware_cea"]
        )
    return {
        "experiment": metrics["experiment"],
        "seed": metrics["seed"],
        "n_eval_scenes": metrics["n_eval_scenes"],
        "resolvable_fraction_eval": metrics["resolvable_fraction_eval"],
        "chance_level": metrics["chance_level"],
        "tiers": tiers,
        "ordering": ordering,
        "best_cea": max(((n, _cea(n)) for n in names), key=lambda kv: kv[1]),
        "theory_validation_r2": metrics["theory_validation"].get("r_squared"),
        "elapsed_seconds": round(elapsed, 2),
    }


def _render_summary_md(metrics, summary, figures, tables) -> str:
    lines = [
        f"# {metrics['experiment']} -- run summary",
        "",
        f"- seed: `{metrics['seed']}`",
        f"- evaluation split: `{metrics['split']}` "
        f"({metrics['n_eval_scenes']} scene variants over {metrics['n_eval_base_scenes']} base scenes)",
        f"- resolvable fraction (ground truth): **{metrics['resolvable_fraction_eval']:.3f}** "
        f"({int(round((1 - metrics['resolvable_fraction_eval']) * metrics['n_eval_scenes']))} scenes are "
        "non-identifiable under the allowed action set)",
        f"- chance-level accuracy: {metrics['chance_level']:.3f}",
        f"- action set: {metrics['action_space_size']} candidate interventions; "
        f"epsilon = {metrics['epsilon_px']:g} px-equivalent; tau = {metrics['tau']:g}",
        f"- dataset: `{metrics['dataset']['name']}` (config hash `{metrics['dataset']['config_hash']}`)",
        "",
        "## Method comparison",
        "",
    ]
    md_table = Path(tables[1]).read_text(encoding="utf-8") if len(tables) > 1 else ""
    lines.append(md_table)
    lines += ["", "## Phase 1 verdict", ""]
    ordering = summary.get("ordering")
    if ordering:
        lines += [
            f"Basis: {ordering['comparison_basis']}.",
            "",
            f"- single-frame CEA        : **{ordering['single_frame_cea']:.3f}** "
            f"(chance = {metrics['chance_level']:.3f})",
            f"- passive multi-view CEA  : **{ordering['passive_multiview_cea']:.3f}**",
            f"- intervention-aware CEA  : **{ordering['intervention_aware_cea']:.3f}**",
            "",
            f"Ordering `single-frame <= passive <= intervention-aware` holds: "
            f"**{ordering['hypothesis_supported']}** "
            f"(strictly increasing: {ordering['strictly_increasing']}).",
            "",
            "With abstention enabled the full method reports "
            f"CEA(all) = {ordering['intervention_aware_with_abstention_cea_all']:.3f}, "
            f"CEA(committed) = {ordering['intervention_aware_with_abstention_cea_committed']:.3f} "
            f"at an abstention rate of {ordering['intervention_aware_abstention_rate']:.3f}. "
            "The gap between those two numbers is the abstention mechanism declining to name a "
            "mechanism on cases no allowed action can resolve.",
        ]
    else:
        lines.append("Ordering NOT EVALUATED: the three tiers were not all present in `methods`.")

    theory = metrics["theory_validation"]
    lines += ["", "## MCRB theory validation", ""]
    if theory.get("status") == "scored":
        lines += [
            f"- measurement: {theory['measurement']}",
            f"- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = {theory['slope']:.4f}, "
            f"intercept = {theory['intercept']:.4f}, **R^2 = {theory['r_squared']:.4f}** (n = {theory['n']})",
            "- the theory predicts linearity; the slope is not expected to equal `f/delta` because the "
            "derivation uses the extremal depth pair while the measurement is an RMS residual.",
        ]
    else:
        lines.append(f"- {theory.get('status', 'NOT SCORED')}")

    lines += ["", "## Interpretation limits (read before quoting any number)", ""]
    lines += [
        "- The causal variants are **pixel-identical at the reference view by construction**, so a "
        "near-chance single-frame result is a property of the benchmark design, not a discovery.",
        "- Methods using the `ground_truth` encoder together with the `analytical` transition share the "
        "simulator's optics exactly. Their results are an **upper bound and a pipeline check**, not evidence "
        "about real imagery. Compare against the noisy-encoder condition for a non-degenerate reading.",
        "- `MAE_MCRB` is near zero for oracle-encoder methods for the same reason.",
        "- No external dataset was used in this run; see `docs/DATASET_MATRIX.md` for their status.",
    ]
    lines += ["", "## Artefacts", "", "- metrics: `metrics/metrics.json`", "- figure inputs: `metrics/figure_data.json`",
              "- predictions: `predictions/predictions.csv`", "- tables: " + ", ".join(f"`{t}`" for t in tables), ""]
    lines.append(f"- figures ({len(figures)} files):")
    for f in sorted({str(Path(x).with_suffix('.pdf')) for x in figures}):
        lines.append(f"  - `{Path(f).name}`")
    return "\n".join(lines) + "\n"
