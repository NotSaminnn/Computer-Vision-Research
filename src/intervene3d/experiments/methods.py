"""The baseline / ablation framework.

A *method* is a named configuration of the six components.  Everything the
research specification lists as a baseline family or an ablation is expressible
as one entry, so activating or deactivating a component is a configuration
change rather than a code change.

Two kinds:

``engine``
    The hypothesis-conditioned inference loop, parameterised by selector,
    transition model, encoder and abstention policy.
``classifier``
    A trained discriminative control (single-frame, or passive multi-view).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from intervene3d.inference.engine import AbstentionPolicy, Intervene3DEngine
from intervene3d.models.belief import LikelihoodBeliefUpdater
from intervene3d.models.encoders import build_geometry_encoder
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.selector import build_selector
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import (
    AnalyticalTransitionModel,
    HybridTransitionModel,
    LearnedOnlyTransitionModel,
    NoHypothesisConditioningTransition,
)


@dataclass
class MethodSpec:
    """One named row of the baseline / ablation table."""

    name: str
    kind: str = "engine"  # "engine" | "classifier"
    selector: str = "max_separability"
    transition: str = "analytical"
    encoder: dict[str, Any] = field(default_factory=lambda: {"name": "ground_truth"})
    abstention: bool = True
    abstain_on_low_confidence: bool = False
    hypothesis_conditioning: bool = True
    fixed_action: str | None = None
    max_steps: int = 1
    classifier: str = "single_frame"  # "single_frame" | "passive_multiview"
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], defaults: Mapping[str, Any]) -> MethodSpec:
        model_defaults = defaults.get("model", {})
        encoder = dict(model_defaults.get("geometry_encoder", {"name": "ground_truth"}))
        encoder.update(payload.get("encoder", {}))
        return cls(
            name=str(payload["name"]),
            kind=str(payload.get("kind", "engine")),
            selector=str(payload.get("selector", "max_separability")),
            transition=str(payload.get("transition", model_defaults.get("transition", {}).get("name", "analytical"))),
            encoder=encoder,
            abstention=bool(payload.get("abstention", model_defaults.get("abstention", {}).get("enabled", True))),
            abstain_on_low_confidence=bool(payload.get("abstain_on_low_confidence", False)),
            hypothesis_conditioning=bool(payload.get("hypothesis_conditioning", True)),
            fixed_action=payload.get("fixed_action"),
            max_steps=int(payload.get("max_steps", 1)),
            classifier=str(payload.get("classifier", "single_frame")),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "selector": self.selector,
            "transition": self.transition, "encoder": self.encoder, "abstention": self.abstention,
            "abstain_on_low_confidence": self.abstain_on_low_confidence,
            "hypothesis_conditioning": self.hypothesis_conditioning,
            "fixed_action": self.fixed_action, "max_steps": self.max_steps,
            "classifier": self.classifier, "description": self.description,
        }


def build_transition(spec: MethodSpec, *, learned_model=None):
    """Instantiate the transition model named by ``spec``."""
    if spec.transition == "analytical":
        return AnalyticalTransitionModel()
    if spec.transition == "no_hypothesis_conditioning":
        return NoHypothesisConditioningTransition()
    if spec.transition in ("hybrid", "learned_only"):
        if learned_model is None:
            raise ValueError(
                f"method {spec.name!r} requests transition={spec.transition!r}, which needs a trained "
                "residual model, but none was supplied.\n"
                "The training stage runs automatically when a method requests 'hybrid' or "
                "'learned_only' and is driven by the optional `learned_transition:` config block "
                "(epochs, hidden_dim, learning_rate, batch_size, max_actions_per_scene). Set "
                "`experiment.train_learned_transition: false` to disable it -- in which case this "
                "method cannot run.\n"
                "Reaching this error means build_engine() was called directly without a "
                "learned_model; see intervene3d.experiments.learned.train_residual_transition."
            )
        cls = HybridTransitionModel if spec.transition == "hybrid" else LearnedOnlyTransitionModel
        return cls(learned_model, hypothesis_conditioning=spec.hypothesis_conditioning)
    raise ValueError(f"unknown transition {spec.transition!r}")


def build_engine(
    spec: MethodSpec,
    model_config: Mapping[str, Any],
    *,
    seed: int,
    learned_model=None,
) -> tuple[Intervene3DEngine, Any]:
    """Assemble the engine and encoder for one method."""
    weights = DistanceWeights.from_dict(model_config.get("distance"))
    transition = build_transition(spec, learned_model=learned_model)
    estimator = GeometrySeparabilityEstimator(transition, weights)
    epsilon = float(model_config.get("identifiability", {}).get("epsilon_px", 1.0))
    belief_cfg = model_config.get("belief", {})
    tau = float(model_config.get("abstention", {}).get("tau", 0.8))

    selector = build_selector(
        spec.selector, estimator, seed=seed, fixed_action=spec.fixed_action, allow_null=False
    )
    engine = Intervene3DEngine(
        estimator=estimator,
        selector=selector,
        belief=LikelihoodBeliefUpdater(
            beta=float(belief_cfg.get("beta", 1.0)), floor=float(belief_cfg.get("floor", 1e-6))
        ),
        identifiability=EpsilonIdentifiabilityEstimator(epsilon=epsilon),
        abstention=AbstentionPolicy(
            enabled=spec.abstention, tau=tau, on_low_confidence=spec.abstain_on_low_confidence
        ),
        max_steps=spec.max_steps,
    )
    encoder = build_geometry_encoder({**spec.encoder, "seed": int(spec.encoder.get("seed", seed))})
    return engine, encoder
