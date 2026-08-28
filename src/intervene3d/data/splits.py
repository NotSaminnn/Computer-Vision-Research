"""Split policy -- leakage prevention.

**The rule: split by underlying base scene, never by frame, pose or rendering.**

Every causal variant of one base scene depicts the same apparent geometry.  If a
mirror variant were in train and its matched display variant in test, a model
could recognise the scene rather than reason about the mechanism, and the
benchmark would measure memorisation.  Assignment is therefore a deterministic
function of ``base_scene_id`` alone, computed from a stable SHA-256 hash so that
regenerating a dataset with more scenes never moves an existing scene between
splits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

SPLIT_ORDER = ("train", "val", "test")


def _stable_unit(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(base_scene_id: str, fractions: Mapping[str, float], *, salt: str = "intervene3d") -> str:
    """Deterministically assign one base scene to train / val / test."""
    total = sum(float(fractions[k]) for k in SPLIT_ORDER)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    u = _stable_unit(base_scene_id, salt)
    acc = 0.0
    for name in SPLIT_ORDER:
        acc += float(fractions[name])
        if u < acc:
            return name
    return SPLIT_ORDER[-1]


def build_splits(
    base_scene_ids: Iterable[str], fractions: Mapping[str, float], *, salt: str = "intervene3d"
) -> dict[str, Any]:
    """Map every base scene to a split and report the realised proportions."""
    ids = list(dict.fromkeys(base_scene_ids))
    assignment = {sid: assign_split(sid, fractions, salt=salt) for sid in ids}
    counts = {name: sum(1 for v in assignment.values() if v == name) for name in SPLIT_ORDER}
    n = max(len(ids), 1)
    return {
        "policy": "base_scene",
        "salt": salt,
        "requested_fractions": {k: float(fractions[k]) for k in SPLIT_ORDER},
        "realised_fractions": {k: counts[k] / n for k in SPLIT_ORDER},
        "counts": counts,
        "assignment": assignment,
        "note": (
            "All causal variants of a base scene share its split. "
            "Splitting by frame, pose or rendering would leak matched counterfactuals."
        ),
    }


def check_no_leakage(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return a list of base scenes that appear in more than one split."""
    seen: dict[str, set[str]] = {}
    for rec in records:
        seen.setdefault(str(rec["base_scene_id"]), set()).add(str(rec["split"]))
    return sorted(base for base, splits in seen.items() if len(splits) > 1)
