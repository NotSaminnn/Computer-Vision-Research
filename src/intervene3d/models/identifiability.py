r"""Action-set identifiability and the epsilon decision.

.. math::
    \mathcal I_{\mathcal A}(H_i, H_j) = \max_{a \in \mathcal A} \Delta_{ij}(a)

.. math::
    \mathcal I_{\mathcal A} < \epsilon \Rightarrow \text{not identifiable under } \mathcal A
    \qquad
    \mathcal I_{\mathcal A} \ge \epsilon \Rightarrow \text{potentially resolvable}

Identifiability is a property of the triple ``(hypothesis set, action set,
perceptual threshold)``.  None of those three may be dropped when reporting a
resolvability label -- the estimator therefore records all of them.

The scene-level score is the **worst** (minimum) pairwise identifiability: a
scene is resolvable only if *every* competing pair can be told apart.  A
belief-weighted variant is provided for inference time, when the system does not
yet know which hypotheses are still in play.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpsilonIdentifiabilityEstimator:
    """``max`` over actions, ``min`` over pairs, thresholded at ``epsilon``."""

    epsilon: float = 1.0
    name: str = "epsilon_threshold"

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    @staticmethod
    def identifiability_matrix(separability_by_action: np.ndarray) -> np.ndarray:
        """``(K, K)`` matrix ``I_A(H_i, H_j) = max_a Delta_ij(a)``."""
        s = np.asarray(separability_by_action, dtype=np.float64)
        if s.ndim != 3 or s.shape[1] != s.shape[2]:
            raise ValueError(f"expected (A, K, K) tensor, got shape {s.shape}")
        return np.max(s, axis=0)

    def evaluate(self, separability_by_action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Return ``(I_matrix, scene_score, resolvable)``.

        ``scene_score`` is the minimum off-diagonal entry of ``I_matrix``.
        """
        I = self.identifiability_matrix(separability_by_action)
        k = I.shape[0]
        if k < 2:  # pragma: no cover - HypothesisSet forbids this
            return I, float("inf"), True
        off = I[~np.eye(k, dtype=bool)]
        score = float(np.min(off))
        return I, score, bool(score >= self.epsilon)

    def weighted_score(
        self, identifiability_matrix: np.ndarray, beliefs: np.ndarray, *, weight_threshold: float = 1e-3
    ) -> float:
        """Belief-weighted worst-pair identifiability.

        Two regimes, and the distinction matters:

        **Live ambiguity.**  If some pair still carries joint posterior mass
        ``p_i p_j > weight_threshold``, the score is the smallest ``I_A`` among
        those pairs: that is the ambiguity the evidence has *not* resolved.

        **Near-point-mass posterior.**  If no pair carries meaningful joint mass
        the system has committed to one hypothesis, and the question becomes
        whether that commitment was *earned*: is the MAP hypothesis separable
        from every alternative under ``A``?  The score is therefore the minimum
        over the MAP hypothesis' own row.

        Taking a global minimum over all pairs in that second regime would be
        wrong: it would let a hard pair between two hypotheses the evidence has
        already eliminated force an abstention on a scene that is, for the MAP
        claim, perfectly identifiable.
        """
        I = np.asarray(identifiability_matrix, dtype=np.float64)
        p = np.asarray(beliefs, dtype=np.float64)
        k = I.shape[0]
        live = [float(I[i, j]) for i in range(k) for j in range(i + 1, k) if p[i] * p[j] > weight_threshold]
        if live:
            return float(np.min(live))
        map_index = int(np.argmax(p))
        row = np.delete(I[map_index], map_index)
        return float(np.min(row))

    def to_dict(self) -> dict[str, float | str]:
        return {"estimator": self.name, "epsilon": self.epsilon}
