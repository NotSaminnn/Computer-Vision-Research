r"""Causal Explanation Accuracy and the False Physical Certainty Rate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def causal_explanation_accuracy(
    predicted: Sequence[str], truth: Sequence[str], *, abstained: Sequence[bool] | None = None
) -> dict[str, Any]:
    r"""``CEA = P(\hat H = H^*)``, reported overall and per true mechanism.

    Abstention is scored explicitly rather than swept under the rug:

    ``cea_all``
        counts abstained samples as incorrect (the strict reading: the system
        did not name the mechanism).
    ``cea_committed``
        computed over non-abstained samples only (how good the system is *when*
        it commits).
    ``abstention_rate``
        the fraction abstained, so the two numbers can never be read in isolation.
    """
    predicted = list(predicted)
    truth = list(truth)
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth must have the same length")
    n = len(truth)
    if n == 0:
        return {"cea_all": float("nan"), "cea_committed": float("nan"), "abstention_rate": float("nan"), "n": 0, "by_mechanism": {}}

    abst = np.zeros(n, dtype=bool) if abstained is None else np.asarray(list(abstained), dtype=bool)
    correct = np.array([p == t for p, t in zip(predicted, truth, strict=True)], dtype=bool)

    cea_all = float(np.mean(correct & ~abst))
    committed = ~abst
    cea_committed = float(np.mean(correct[committed])) if np.any(committed) else float("nan")

    by_mech: dict[str, dict[str, float]] = {}
    for mech in sorted(set(truth)):
        m = np.array([t == mech for t in truth], dtype=bool)
        by_mech[mech] = {
            "cea_all": float(np.mean((correct & ~abst)[m])),
            "cea_committed": float(np.mean(correct[m & committed])) if np.any(m & committed) else float("nan"),
            "abstention_rate": float(np.mean(abst[m])),
            "n": int(np.count_nonzero(m)),
        }

    return {
        "cea_all": cea_all,
        "cea_committed": cea_committed,
        "abstention_rate": float(np.mean(abst)),
        "n": n,
        "by_mechanism": by_mech,
    }


def false_physical_certainty_rate(
    max_probability: Sequence[float],
    resolvable: Sequence[bool],
    *,
    tau: float = 0.8,
    abstained: Sequence[bool] | None = None,
) -> dict[str, Any]:
    r"""``FPCR = P(\max_k p(H_k) > \tau \mid y_{id} = 0)``.

    The rate at which the system asserts confident physical certainty on cases
    that *no allowed intervention can resolve*.  A system that abstains on such
    a case does not count as confidently certain, which is exactly the behaviour
    the abstention mechanism exists to produce.
    """
    p = np.asarray(list(max_probability), dtype=np.float64)
    y = np.asarray(list(resolvable), dtype=bool)
    if p.shape != y.shape:
        raise ValueError("max_probability and resolvable must have the same length")
    abst = np.zeros(p.shape, dtype=bool) if abstained is None else np.asarray(list(abstained), dtype=bool)

    non_identifiable = ~y
    n_ni = int(np.count_nonzero(non_identifiable))
    if n_ni == 0:
        return {"fpcr": float("nan"), "n_non_identifiable": 0, "tau": tau, "note": "no non-identifiable samples"}

    confident = (p > tau) & ~abst
    fpcr = float(np.mean(confident[non_identifiable]))

    identifiable = y
    fpr_on_identifiable = (
        float(np.mean(confident[identifiable])) if np.any(identifiable) else float("nan")
    )
    return {
        "fpcr": fpcr,
        "n_non_identifiable": n_ni,
        "tau": float(tau),
        "confident_rate_on_identifiable": fpr_on_identifiable,
        "note": "abstained samples are not counted as confidently certain",
    }


def confusion_matrix(predicted: Sequence[str], truth: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Rows = true label, columns = predicted label; unknown predictions ignored."""
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for p, t in zip(predicted, truth, strict=True):
        if t in index and p in index:
            m[index[t], index[p]] += 1
    return m
