"""Physical contact-depth error.

``D_contact`` is the depth of the *first physical surface* along a ray -- the
mirror glass, the screen plane, the window pane -- rather than the apparent
content behind it.  These are the standard depth metrics restricted to that
quantity.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _valid(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    ok = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    return ok


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute relative error ``mean(|d - d*| / d*)``."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = _valid(pred, gt, mask)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(np.abs(pred[ok] - gt[ok]) / gt[ok]))


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root mean squared error in metres."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = _valid(pred, gt, mask)
    if not np.any(ok):
        return float("nan")
    return float(np.sqrt(np.mean((pred[ok] - gt[ok]) ** 2)))


def delta_threshold(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, threshold: float = 1.25) -> float:
    """Fraction of pixels with ``max(d/d*, d*/d) < threshold``."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = _valid(pred, gt, mask) & (pred > 0)
    if not np.any(ok):
        return float("nan")
    ratio = np.maximum(pred[ok] / gt[ok], gt[ok] / pred[ok])
    return float(np.mean(ratio < threshold))


def contact_depth_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, Any]:
    """``AbsRel_contact``, ``RMSE_contact`` and ``delta<1.25``."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = _valid(pred, gt, mask)
    return {
        "abs_rel_contact": abs_rel(pred, gt, mask),
        "rmse_contact": rmse(pred, gt, mask),
        "delta_1_25_contact": delta_threshold(pred, gt, mask),
        "n_valid": int(np.count_nonzero(ok)),
    }
