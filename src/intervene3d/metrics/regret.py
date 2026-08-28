r"""Intervention regret -- did the system choose a near-optimal experiment?

.. math:: \text{Regret} = \Delta(a^*) - \Delta(\hat a)

where ``Delta(a)`` is the oracle belief-weighted separability of action ``a``
and ``a*`` is the oracle-optimal action.  Regret is zero for an optimal choice
and never negative.  A normalised variant divides by ``Delta(a*)`` so scenes of
different difficulty can be averaged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def intervention_regret(oracle_utility: np.ndarray, chosen_index: int) -> dict[str, Any]:
    """Absolute and normalised regret for one scene."""
    u = np.asarray(oracle_utility, dtype=np.float64)
    if u.size == 0:
        return {"regret": float("nan"), "normalised_regret": float("nan"), "best_utility": float("nan")}
    best = float(np.max(u))
    chosen = float(u[int(chosen_index)])
    regret = max(best - chosen, 0.0)
    norm = regret / best if best > 1e-12 else 0.0
    return {
        "regret": regret,
        "normalised_regret": float(norm),
        "best_utility": best,
        "chosen_utility": chosen,
        "optimal_action_index": int(np.argmax(u)),
    }


def motion_cost(translation_m: float, rotation_rad: float, *, rotation_weight_m_per_rad: float = 0.5) -> float:
    """Scalar motion budget used by RQ3 ("resolve with less motion").

    Rotation is converted to a translation-equivalent by a configurable factor;
    the factor is a modelling choice and is recorded in the run manifest.
    """
    return float(translation_m + rotation_weight_m_per_rad * rotation_rad)
