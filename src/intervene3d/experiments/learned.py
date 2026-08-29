"""Training stage for the learned transition model (Gate 7).

Before this module existed, ``configs/models/hybrid.yaml`` could not be run at
all: :func:`intervene3d.experiments.methods.build_transition` refused
``transition: hybrid`` with an error telling the reader to *"set
experiment.train_learned_transition: true"* -- a key nothing read.  That is what
``docs/EXPERIMENT_PLAN.md`` E9 means by *learned-only transition: IMPLEMENTED,
NOT RUN -- needs a training stage.*

What is learned
---------------
A per-landmark **residual** on the content channel::

    target = F_{t+1}^observed  -  Phi_base(F_t, H_k, a)

with ``Phi_base`` the analytical optics (``hybrid``) or the identity
(``learned_only``).  The residual is ``(du, dv, d_depth)``.

The two bases need different residuals -- against the analytical base the target
is near zero, against the identity it is the whole optical effect -- so a model
is trained **per base**, never shared.

Supervision is honest
---------------------
* Only the **train** split is used.  Splits are by base scene, so no matched
  counterfactual of a training scene can appear at evaluation.
* Targets come from the simulator's **pre-simulated observation for that exact
  action**, not from re-running the model being trained.
* A landmark contributes only when it is visible and finite in *both* the base
  prediction and the observation.  Invisible landmarks are dropped, never
  zero-filled into the loss -- a zero residual on an invisible point is not a
  free correct answer, it is a fabricated one.
* Each scene is paired with **its own true hypothesis**.  Training a transition
  on a mechanism the scene does not have would teach it the wrong physics.

Scope: this trains the small NumPy MLP that ships with the repository, on the
synthetic benchmark.  It is a Gate-7 *placeholder*, not a world model, and it is
not a substitute for training on external sequences -- see
``intervene3d.models.torch_transition`` for the Gate-7 stage that trains on
acquired external (rendered, not photographic) data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature
from intervene3d.hypotheses.base import Hypothesis
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.interventions.actions import Action
from intervene3d.models.learned_transition import (
    INPUT_DIM,
    OUTPUT_DIM,
    ResidualMLP,
    build_inputs,
)
from intervene3d.models.transition import AnalyticalTransitionModel

LOGGER = logging.getLogger(__name__)

#: Bases a residual can be trained against, and the transition names that use them.
BASE_FOR_TRANSITION = {"hybrid": "analytical", "learned_only": "identity"}


def _identity_base(state: GeometryFeature, action: Action) -> GeometryFeature:
    """The ``learned_only`` base: reproject nothing, just move the camera."""
    cam = state.camera.moved(action.delta_T())
    return GeometryFeature(state.uv.copy(), state.depth.copy(), state.visible.copy(), state.channel, cam)


def base_prediction(
    base: str,
    state: GeometryFeature,
    hypothesis: Hypothesis,
    action: Action,
    *,
    markers_cam: np.ndarray | None = None,
    analytical: AnalyticalTransitionModel | None = None,
) -> GeometryFeature:
    """``Phi_base(F_t, H_k, a)`` for the named base."""
    if base == "identity":
        return _identity_base(state, action)
    if base == "analytical":
        model = analytical or AnalyticalTransitionModel()
        return model.predict(state, hypothesis, action, markers_cam=markers_cam)
    raise ValueError(f"unknown residual base {base!r}; expected 'analytical' or 'identity'")


def build_training_pairs(
    scenes: Sequence[Any],
    actions: ActionSpace,
    *,
    base: str,
    hypothesis_conditioning: bool = True,
    max_actions: int | None = None,
    encoder=None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Assemble ``(X, Y, report)`` from real simulated observations.

    ``X`` is ``(N, INPUT_DIM)``, ``Y`` is ``(N, OUTPUT_DIM)``.  ``report`` records
    how many landmark-action pairs were offered and how many survived the
    visibility filter, so a caller can see what fraction of the data was usable
    rather than assuming it was all of it.
    """
    analytical = AnalyticalTransitionModel()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    offered = kept = 0
    n_actions_used = 0

    for scene in scenes:
        reference = scene.reference_observation()
        state = encoder.encode(reference) if encoder is not None else reference.feature
        hypotheses = scene.hypothesis_set()
        hypothesis = hypotheses[scene.true_index]  # the scene's OWN mechanism
        markers = scene.markers_cam

        indices = list(range(len(actions)))
        if max_actions is not None and max_actions < len(indices):
            picker = rng or np.random.default_rng(0)
            indices = sorted(picker.choice(len(indices), size=max_actions, replace=False).tolist())
        n_actions_used = len(indices)

        for ai in indices:
            action = actions[ai]
            observation = scene.observation_for_action(ai)
            observed = encoder.encode(observation) if encoder is not None else observation.feature
            predicted = base_prediction(
                base, state, hypothesis, action, markers_cam=markers, analytical=analytical
            )

            x = build_inputs(state, hypothesis, action, hypothesis_conditioning=hypothesis_conditioning)
            if x.shape[0] == 0:
                continue

            mask = state.channel == CHANNEL_CONTENT
            du = observed.uv[mask, 0] - predicted.uv[mask, 0]
            dv = observed.uv[mask, 1] - predicted.uv[mask, 1]
            dz = observed.depth[mask] - predicted.depth[mask]
            y = np.stack([du, dv, dz], axis=1)
            # NOTE: channels are (pixels, pixels, metres). An MSE over the raw
            # stack is ~10^6 times more sensitive to the pixel channels, so the
            # depth channel would never be learned and any single "mse" would be
            # a pixel metric wearing a unit-free name. Per-channel figures are
            # reported separately below; see `target_rms_*`.

            # A landmark supervises only where BOTH the base and the observation
            # are real. Zero-filling an invisible point would train the model to
            # answer confidently where it has no evidence.
            usable = (
                observed.visible[mask]
                & predicted.visible[mask]
                & np.all(np.isfinite(y), axis=1)
            )
            offered += int(y.shape[0])
            kept += int(np.count_nonzero(usable))
            if not np.any(usable):
                continue
            xs.append(x[usable])
            ys.append(y[usable])

    if not xs:
        raise ValueError(
            "no usable (landmark, action) supervision was produced. Every candidate was "
            "invisible or non-finite in either the base prediction or the observation."
        )
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    assert X.shape[1] == INPUT_DIM and Y.shape[1] == OUTPUT_DIM
    report = {
        "base": base,
        "hypothesis_conditioning": hypothesis_conditioning,
        "n_scenes": len(scenes),
        "n_actions_per_scene": n_actions_used,
        "landmark_actions_offered": offered,
        "landmark_actions_kept": kept,
        "kept_fraction": round(kept / max(offered, 1), 4),
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "target_rms_uv_px": float(np.sqrt(np.mean(Y[:, :2] ** 2))),
        "target_rms_depth_m": float(np.sqrt(np.mean(Y[:, 2] ** 2))),
        "units": ["px", "px", "m"],
        "units_note": (
            "the three residual channels are in different units; a pooled MSE is dominated "
            "by the pixel channels and is not a depth metric"
        ),
    }
    return X, Y, report


def train_residual_transition(
    scenes: Sequence[Any],
    actions: ActionSpace,
    *,
    base: str,
    seed: int,
    config: Mapping[str, Any] | None = None,
    hypothesis_conditioning: bool = True,
    encoder=None,
    val_fraction: float = 0.15,
) -> tuple[ResidualMLP, dict[str, Any]]:
    """Train one residual model and report what actually happened.

    A held-out slice of the *training* split is kept back so the report contains
    a validation loss the model never saw.  That slice is carved by **base
    scene**, not by row: rows from one scene are highly correlated, and splitting
    them would report an optimistic number.
    """
    cfg = dict(config or {})
    epochs = int(cfg.get("epochs", 200))
    hidden = int(cfg.get("hidden_dim", 64))
    lr = float(cfg.get("learning_rate", 0.01))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    batch_size = int(cfg.get("batch_size", 256))
    max_actions = cfg.get("max_actions_per_scene")
    rng = np.random.default_rng([seed, 0xE5])

    scenes = list(scenes)
    holdout_ids: set[str] = set()
    if 0.0 < val_fraction < 1.0:
        base_ids = sorted({s.base_scene_id for s in scenes})
        n_hold = max(1, int(round(len(base_ids) * val_fraction))) if len(base_ids) > 1 else 0
        if n_hold:
            holdout_ids = set(rng.permutation(np.array(base_ids, dtype=object))[:n_hold].tolist())
    fit_scenes = [s for s in scenes if s.base_scene_id not in holdout_ids]
    val_scenes = [s for s in scenes if s.base_scene_id in holdout_ids]
    if not fit_scenes:  # pragma: no cover - only if val_fraction is absurd
        fit_scenes, val_scenes, holdout_ids = scenes, [], set()

    t0 = time.time()
    X, Y, data_report = build_training_pairs(
        fit_scenes, actions, base=base, hypothesis_conditioning=hypothesis_conditioning,
        max_actions=int(max_actions) if max_actions else None, encoder=encoder, rng=rng,
    )
    model = ResidualMLP(hidden_dim=hidden, seed=seed, learning_rate=lr, weight_decay=weight_decay)
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, seed=seed)
    train_seconds = time.time() - t0

    report: dict[str, Any] = {
        "trained": True,
        "base": base,
        "transition": "hybrid" if base == "analytical" else "learned_only",
        "hypothesis_conditioning": hypothesis_conditioning,
        "seed": seed,
        "hyperparameters": {
            "hidden_dim": hidden, "epochs": epochs, "learning_rate": lr,
            "weight_decay": weight_decay, "batch_size": batch_size,
            "max_actions_per_scene": max_actions,
        },
        "data": data_report,
        "split_protocol": (
            "held out by BASE SCENE from the train split; the evaluation split is never touched"
        ),
        "n_fit_scenes": len(fit_scenes),
        "n_val_scenes": len(val_scenes),
        "loss_initial": float(history[0]) if history else None,
        "loss_final": float(history[-1]) if history else None,
        "loss_history": [float(h) for h in history],
        "train_seconds": round(train_seconds, 2),
    }

    # On the synthetic benchmark the analytical transition reproduces the
    # simulator bit-for-bit (README limitation 1: they share forward optics), so
    # the residual target is identically zero and `hybrid` is EQUAL to
    # `analytical` rather than an improvement on it. Flag it, loudly, so a run
    # cannot be read as "the learned residual added nothing" when in truth it
    # was never given anything to learn.
    # Tolerance, not exact equality: if the base matched the simulator to
    # floating-point rounding (RMS ~1e-16) rather than bit-for-bit, an `== 0.0`
    # test would silently pass and a model trained on float noise would be
    # reported as a result.
    degenerate = (
        data_report["target_rms_uv_px"] < 1e-9 and data_report["target_rms_depth_m"] < 1e-12
    )
    report["degenerate_zero_target"] = bool(degenerate)
    if degenerate:
        report["degenerate_note"] = (
            "The residual target is identically zero: the base model reproduces the simulator "
            "exactly, so there is nothing to learn and this transition is equivalent to its base. "
            "This is a property of the synthetic benchmark (the simulator and the analytical "
            "transition share the same forward optics), not a result about learned transitions. "
            "A meaningful hybrid residual requires data the analytical model does NOT generate -- "
            "i.e. real sequences. See docs/commands.md section 7."
        )
        LOGGER.warning(
            "%s residual has an identically zero target: '%s' is equivalent to its base on this "
            "dataset and must not be reported as a learned-transition result",
            base, report["transition"],
        )

    if val_scenes:
        Xv, Yv, val_data = build_training_pairs(
            val_scenes, actions, base=base, hypothesis_conditioning=hypothesis_conditioning,
            max_actions=int(max_actions) if max_actions else None, encoder=encoder, rng=rng,
        )
        pred = model.predict(Xv)
        baseline_mse = float(np.mean(Yv**2))          # what predicting zero would score
        model_mse = float(np.mean((pred - Yv) ** 2))
        per_channel = {}
        for i, (label, unit) in enumerate((("du", "px"), ("dv", "px"), ("depth", "m"))):
            base_i = float(np.mean(Yv[:, i] ** 2))
            mod_i = float(np.mean((pred[:, i] - Yv[:, i]) ** 2))
            per_channel[label] = {
                "unit": unit,
                "zero_residual_mse": base_i,
                "model_mse": mod_i,
                "ratio_vs_base": round(mod_i / base_i, 6) if base_i > 0 else None,
            }
        report["validation"] = {
            "n_samples": int(Xv.shape[0]),
            "n_scenes": len(val_scenes),
            # Pooled across channels of DIFFERENT UNITS -- kept for continuity but
            # it is a pixel metric, not a geometry one. Read `per_channel`.
            "mse_POOLED_MIXED_UNITS": model_mse,
            "zero_residual_mse_POOLED": baseline_mse,
            "mse_ratio_vs_base_POOLED": round(model_mse / baseline_mse, 6) if baseline_mse > 0 else None,
            "per_channel": per_channel,
            "kept_fraction": val_data["kept_fraction"],
        }
    else:
        report["validation"] = None

    LOGGER.info(
        "trained %s residual: %d samples from %d scenes (%.1f%% of landmark-actions usable), "
        "loss %.6g -> %.6g in %.1f s",
        base, data_report["n_samples"], len(fit_scenes), 100 * data_report["kept_fraction"],
        report["loss_initial"], report["loss_final"], train_seconds,
    )
    return model, report


def required_bases(specs: Sequence[Any]) -> dict[str, bool]:
    """Which residual bases the method list needs → hypothesis-conditioning flag.

    Returns ``{}`` when no method uses a learned transition, which is the common
    case and means no training stage runs at all.
    """
    needed: dict[str, bool] = {}
    for spec in specs:
        base = BASE_FOR_TRANSITION.get(getattr(spec, "transition", ""))
        if base is None:
            continue
        # A base shared by two methods with different conditioning must be
        # trained twice; key on the pair rather than collapsing them.
        needed[f"{base}:{int(bool(spec.hypothesis_conditioning))}"] = bool(spec.hypothesis_conditioning)
    return needed
