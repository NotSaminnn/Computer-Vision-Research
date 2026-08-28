"""Identifiability AUROC and intervention regret.

AUROC is computed with the rank-based (Mann-Whitney U) formulation, which
handles ties exactly and needs no threshold sweep; it agrees with the
trapezoidal ROC integral.  Implemented here rather than pulled from scikit-learn
so the preliminary codebase keeps a minimal dependency set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared -- equivalent to scipy's 'average'."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve for a binary target.

    Returns NaN when one class is absent (AUROC is undefined then, and
    reporting 0.5 would be a fabricated result).
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=bool)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((np.sum(ranks[y]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def identifiability_auroc(
    identifiability_scores: Sequence[float], resolvable_labels: Sequence[bool]
) -> dict[str, Any]:
    """Can the system predict whether the ambiguity is resolvable under ``A``?"""
    value = auroc(identifiability_scores, resolvable_labels)
    y = np.asarray(list(resolvable_labels), dtype=bool)
    return {
        "identifiability_auroc": value,
        "n_resolvable": int(np.count_nonzero(y)),
        "n_non_resolvable": int(np.count_nonzero(~y)),
        "n": int(y.size),
    }


def binary_decision_metrics(predicted: Sequence[bool], truth: Sequence[bool]) -> dict[str, float]:
    """Accuracy / precision / recall / F1 of the hard resolvability decision."""
    p = np.asarray(list(predicted), dtype=bool)
    t = np.asarray(list(truth), dtype=bool)
    tp = float(np.count_nonzero(p & t))
    fp = float(np.count_nonzero(p & ~t))
    fn = float(np.count_nonzero(~p & t))
    tn = float(np.count_nonzero(~p & ~t))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {
        "resolvability_accuracy": (tp + tn) / total if total else float("nan"),
        "resolvability_precision": precision,
        "resolvability_recall": recall,
        "resolvability_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def roc_curve(scores: Sequence[float], labels: Sequence[bool]) -> dict[str, list[float]]:
    """ROC points for plotting.  Returns ``{"fpr": [...], "tpr": [...], "thresholds": [...]}``.

    Thresholds are the sorted unique scores, so the curve is exact rather than a
    fixed-grid approximation.
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=bool)
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n_pos, n_neg = int(np.count_nonzero(y)), int(np.count_nonzero(~y))
    if n_pos == 0 or n_neg == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": []}
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    # Keep only the last index of each run of equal scores.
    distinct = np.nonzero(np.diff(s_sorted))[0]
    idx = np.r_[distinct, s_sorted.size - 1]
    tpr = np.r_[0.0, tps[idx] / n_pos]
    fpr = np.r_[0.0, fps[idx] / n_neg]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": np.r_[np.inf, s_sorted[idx]].tolist()}
