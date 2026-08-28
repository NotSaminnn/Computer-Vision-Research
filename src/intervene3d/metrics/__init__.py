"""Reusable metric functions.

Implemented (all covered by ``tests/unit/test_metrics.py``):

* ``CEA``                  -- causal explanation accuracy, overall and per mechanism
* ``AbsRel/RMSE_contact``  -- physical contact-depth error
* ``Identifiability AUROC``-- can resolvability under ``A`` be predicted?
* ``MCRB`` / ``MAE_MCRB``  -- minimum causal resolving baseline and its error
* ``FPCR``                 -- false physical certainty rate
* ``Intervention regret``  -- optional; how good was the chosen experiment?
"""

from intervene3d.metrics.aggregate import aggregate_runs, flatten_metrics, format_pm, summarise
from intervene3d.metrics.classification import (
    causal_explanation_accuracy,
    confusion_matrix,
    false_physical_certainty_rate,
)
from intervene3d.metrics.depth import abs_rel, contact_depth_metrics, delta_threshold, rmse
from intervene3d.metrics.identifiability import (
    auroc,
    binary_decision_metrics,
    identifiability_auroc,
    roc_curve,
)
from intervene3d.metrics.mcrb import (
    MCRBResult,
    analytic_pair_is_applicable,
    differential_parallax,
    mcrb_absolute_error,
    mcrb_analytic,
    mcrb_numeric,
)
from intervene3d.metrics.regret import intervention_regret, motion_cost

__all__ = [
    "aggregate_runs",
    "flatten_metrics",
    "format_pm",
    "summarise",
    "causal_explanation_accuracy",
    "confusion_matrix",
    "false_physical_certainty_rate",
    "abs_rel",
    "contact_depth_metrics",
    "delta_threshold",
    "rmse",
    "auroc",
    "binary_decision_metrics",
    "identifiability_auroc",
    "roc_curve",
    "MCRBResult",
    "analytic_pair_is_applicable",
    "differential_parallax",
    "mcrb_absolute_error",
    "mcrb_analytic",
    "mcrb_numeric",
    "intervention_regret",
    "motion_cost",
]
