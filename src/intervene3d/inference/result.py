r"""The inference output object.

Abstention is a first-class outcome, not a threshold applied after the fact, and
the result object therefore keeps **two distinct uncertainties** apart:

``prediction_uncertainty`` (``U_prediction``)
    Normalised posterior entropy.  "I am not sure which hypothesis it is."
``identifiability_uncertainty`` (``U_identifiability``)
    How far the belief-weighted ``I_A`` falls short of ``epsilon``.  "No allowed
    intervention *can* tell these hypotheses apart."

A system can be perfectly confident that an apparent corridor has depth
structure while being fundamentally unable to decide whether that structure is a
real corridor or a view-conditioned display.  Conflating the two is exactly the
failure mode the project exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The message a system must be able to emit instead of a physical label.
UNRESOLVED_MESSAGE = "physical explanation unresolved under available visual evidence"


@dataclass(frozen=True)
class InferenceResult:
    """Everything one inference episode produced."""

    scene_id: str
    hypothesis_names: tuple[str, ...]
    hypothesis_mechanisms: tuple[str, ...]
    hypothesis_probabilities: np.ndarray
    identifiability_score: float
    resolvable: bool
    selected_action: str
    selected_action_index: int
    contact_geometry: np.ndarray
    abstained: bool
    reason: str
    prediction_uncertainty: float
    identifiability_uncertainty: float
    errors: np.ndarray
    action_utility: np.ndarray
    identifiability_matrix: np.ndarray
    belief_trajectory: np.ndarray
    executed_actions: tuple[str, ...] = ()
    motion_cost: float = 0.0
    predicted_mcrb: float | None = None
    steps: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def map_index(self) -> int:
        return int(np.argmax(self.hypothesis_probabilities))

    @property
    def max_probability(self) -> float:
        return float(np.max(self.hypothesis_probabilities))

    @property
    def predicted_mechanism(self) -> str:
        """The named mechanism, or ``"abstain"`` when the system declined to commit.

        Deliberately returns ``"abstain"`` rather than the argmax: a caller that
        forgets to check :attr:`abstained` must not silently receive a physical
        label the system did not stand behind.
        """
        return "abstain" if self.abstained else self.hypothesis_mechanisms[self.map_index]

    @property
    def committed_mechanism(self) -> str:
        """The MAP mechanism regardless of abstention (for diagnostics only)."""
        return self.hypothesis_mechanisms[self.map_index]

    @property
    def message(self) -> str:
        if self.abstained:
            return UNRESOLVED_MESSAGE
        return f"physical explanation: {self.committed_mechanism} (p={self.max_probability:.3f})"

    def to_record(self) -> dict[str, Any]:
        """Flat, CSV-friendly record for ``predictions.csv``."""
        return {
            "scene_id": self.scene_id,
            "predicted_mechanism": self.predicted_mechanism,
            "committed_mechanism": self.committed_mechanism,
            "max_probability": self.max_probability,
            "abstained": bool(self.abstained),
            "reason": self.reason,
            "identifiability_score": self.identifiability_score,
            "resolvable_pred": bool(self.resolvable),
            "selected_action": self.selected_action,
            "prediction_uncertainty": self.prediction_uncertainty,
            "identifiability_uncertainty": self.identifiability_uncertainty,
            "motion_cost": self.motion_cost,
            "steps": self.steps,
            "predicted_mcrb": self.predicted_mcrb,
            **{f"p_{n}": float(p) for n, p in zip(self.hypothesis_names, self.hypothesis_probabilities, strict=True)},
        }
