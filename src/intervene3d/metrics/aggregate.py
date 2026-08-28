"""Multi-seed aggregation: mean, standard deviation and confidence intervals.

The research plan forbids reporting one lucky seed.  These helpers turn a list
of per-run metric dictionaries into mean / std / 95% CI summaries, using a
Student-t critical value so small seed counts are not reported as if they were
Gaussian.  The t table is inlined to keep SciPy out of the dependency set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

#: Two-sided 95% Student-t critical values indexed by degrees of freedom.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical_95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if dof in _T95:
        return _T95[dof]
    keys = sorted(_T95)
    for k in keys:
        if dof < k:
            return _T95[k]
    return 1.96


def summarise(values: Sequence[float]) -> dict[str, Any]:
    """Mean / std / n / 95% CI half-width for one metric across seeds."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "ci95": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = float(np.mean(v))
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1, "ci95": float("nan"), "min": mean, "max": mean,
                "note": "single seed -- no confidence interval; re-run with --seed to aggregate"}
    std = float(np.std(v, ddof=1))
    ci = t_critical_95(n - 1) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "n": n, "ci95": float(ci), "min": float(np.min(v)), "max": float(np.max(v))}


def flatten_metrics(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten a nested metrics dict into ``dotted.key -> scalar``."""
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[name] = float(value)
    return out


def aggregate_runs(run_metrics: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate a collection of per-run metric dictionaries across seeds."""
    flattened = [flatten_metrics(m) for m in run_metrics]
    keys: list[str] = []
    for f in flattened:
        for k in f:
            if k not in keys:
                keys.append(k)
    return {k: summarise([f.get(k, float("nan")) for f in flattened]) for k in keys}


def format_pm(summary: Mapping[str, Any], *, digits: int = 3) -> str:
    """``"0.812 +/- 0.021"`` -- a stable string for tables."""
    mean = summary.get("mean", float("nan"))
    if not np.isfinite(mean):
        return "n/a"
    if summary.get("n", 0) <= 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {summary.get('std', float('nan')):.{digits}f}"
