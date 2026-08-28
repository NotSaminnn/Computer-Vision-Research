r"""The Intervene3D inference loop.

::

    reference observation
        -> candidate hypotheses
        -> candidate camera actions
        -> analytical hypothesis-conditioned prediction   (H x A matrix)
        -> pairwise separability
        -> choose the intervention
        -> observe
        -> compare observation with predictions
        -> belief update
        -> identifiability decision
             resolved   -> physical explanation + contact geometry
             unresolved -> explicit abstention

Observation is supplied by a callable so the same engine drives the benchmark
(reading pre-simulated observations from disk) and any live simulator, without
the engine knowing which.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.data.types import GeometryFeature, Observation
from intervene3d.hypotheses.base import HypothesisSet
from intervene3d.inference.result import InferenceResult
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.interventions.actions import Action
from intervene3d.metrics.regret import motion_cost
from intervene3d.models.belief import LikelihoodBeliefUpdater, normalised_entropy
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.separability import GeometrySeparabilityEstimator, feature_distance

ObserveFn = Callable[[Action, int], Observation]


@dataclass
class AbstentionPolicy:
    """When the system refuses to name a physical mechanism.

    ``enabled=False`` reproduces an ordinary forced-choice classifier and is one
    of the ablations the research specification asks for ("no unresolvable
    state").
    """

    enabled: bool = True
    tau: float = 0.8
    on_low_confidence: bool = False

    def decide(self, resolvable: bool, max_probability: float) -> tuple[bool, str]:
        if not self.enabled:
            return False, "abstention disabled (forced choice)"
        if not resolvable:
            return True, "identifiability: I_A < epsilon under the allowed action set"
        if self.on_low_confidence and max_probability < self.tau:
            return True, f"prediction uncertainty: max p = {max_probability:.3f} < tau = {self.tau:g}"
        return False, "resolved"


@dataclass
class Intervene3DEngine:
    """Assembles the six components into the scientific loop."""

    estimator: GeometrySeparabilityEstimator
    selector: Any
    belief: LikelihoodBeliefUpdater
    identifiability: EpsilonIdentifiabilityEstimator
    abstention: AbstentionPolicy = field(default_factory=AbstentionPolicy)
    max_steps: int = 1
    rotation_weight_m_per_rad: float = 0.5

    def run(
        self,
        scene_id: str,
        reference_feature: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        observe: ObserveFn,
        *,
        markers_cam: np.ndarray | None = None,
        prior: np.ndarray | None = None,
    ) -> InferenceResult:
        beliefs = hypotheses.uniform_prior() if prior is None else np.asarray(prior, dtype=np.float64)
        trajectory = [beliefs.copy()]

        sep = self.estimator.pairwise_over_actions(
            reference_feature, hypotheses, actions, markers_cam=markers_cam
        )
        I_matrix, _, _ = self.identifiability.evaluate(sep)

        remaining = list(range(len(actions)))
        executed: list[str] = []
        total_motion = 0.0
        utility = np.zeros(len(actions))
        errors = np.zeros(len(hypotheses))
        chosen_index = -1
        chosen_action: Action | None = None
        steps = 0

        for _ in range(max(self.max_steps, 1)):
            if not remaining:
                break
            sub_actions = ActionSpace(tuple(actions[i] for i in remaining), actions.config)
            action, sub_utility = self.selector.select(
                reference_feature, hypotheses, sub_actions, beliefs, markers_cam=markers_cam
            )
            chosen_index = actions.index_of(action.name)
            utility = np.zeros(len(actions))
            for local, global_idx in enumerate(remaining):
                utility[global_idx] = sub_utility[local]
            chosen_action = action
            remaining = [i for i in remaining if i != chosen_index]
            executed.append(action.name)
            total_motion += motion_cost(
                action.translation_magnitude,
                action.rotation_magnitude,
                rotation_weight_m_per_rad=self.rotation_weight_m_per_rad,
            )
            steps += 1

            observation = observe(action, chosen_index)
            predictions = self.estimator.predict_all(
                reference_feature, hypotheses, action, markers_cam=markers_cam
            )
            errors = np.array(
                [feature_distance(observation.feature, p, self.estimator.weights).total for p in predictions]
            )
            beliefs = self.belief.update(beliefs, errors)
            trajectory.append(beliefs.copy())

            if float(np.max(beliefs)) >= self.abstention.tau:
                break

        # --- identifiability decision, weighted by what the posterior still allows
        score = self.identifiability.weighted_score(I_matrix, beliefs)
        resolvable = bool(score >= self.identifiability.epsilon)
        max_p = float(np.max(beliefs))
        abstained, reason = self.abstention.decide(resolvable, max_p)

        map_idx = int(np.argmax(beliefs))
        contact = _contact_geometry(
            self.estimator, reference_feature, hypotheses, map_idx, chosen_action, markers_cam
        )

        return InferenceResult(
            scene_id=scene_id,
            hypothesis_names=tuple(hypotheses.names),
            hypothesis_mechanisms=tuple(h.mechanism.value for h in hypotheses),
            hypothesis_probabilities=beliefs,
            identifiability_score=float(score),
            resolvable=resolvable,
            selected_action=executed[-1] if executed else "stay",
            selected_action_index=chosen_index,
            contact_geometry=contact,
            abstained=abstained,
            reason=reason,
            prediction_uncertainty=normalised_entropy(beliefs),
            identifiability_uncertainty=float(max(0.0, 1.0 - score / self.identifiability.epsilon)),
            errors=errors,
            action_utility=utility,
            identifiability_matrix=I_matrix,
            belief_trajectory=np.stack(trajectory),
            executed_actions=tuple(executed),
            motion_cost=float(total_motion),
            steps=steps,
        )


def _contact_geometry(
    estimator: GeometrySeparabilityEstimator,
    reference_feature: GeometryFeature,
    hypotheses: HypothesisSet,
    map_index: int,
    action: Action | None,
    markers_cam: np.ndarray | None,
) -> np.ndarray:
    """Contact depth under the MAP hypothesis, evaluated at the reference view.

    Reported even when the system abstains: knowing *which* surface you would
    touch is often still actionable, and reporting it separately from the
    mechanism label is precisely the point of the abstention mechanism.
    """
    from intervene3d.interventions.actions import null_action
    from intervene3d.optics.registry import get_transition

    hypothesis = hypotheses[map_index]
    transition = get_transition(hypothesis.mechanism)
    return transition.predict_contact_depth(
        reference_feature, hypothesis, null_action(), markers_cam=markers_cam
    )
