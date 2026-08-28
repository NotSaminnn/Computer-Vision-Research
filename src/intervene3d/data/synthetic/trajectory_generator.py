"""Action sets and the lateral baseline sweeps used for MCRB estimation."""

from __future__ import annotations

from typing import Any

import numpy as np

from intervene3d.interventions.action_space import ActionSpace, ActionSpaceConfig


def action_space_from_config(config: dict[str, Any]) -> ActionSpace:
    """The allowed intervention set ``A``.

    Every identifiability label in the dataset is relative to *this* set, so it
    is serialised with the dataset manifest.
    """
    return ActionSpace.from_config(ActionSpaceConfig.from_dict(config))


def mcrb_baseline_sweep(config: dict[str, Any]) -> tuple[np.ndarray, ActionSpace]:
    """A pure lateral sweep ``B in [0, max_baseline]`` for resolving-baseline search.

    Returns ``(baselines, action_space)`` where the action ordering matches the
    baseline ordering, so a separability curve can be indexed directly.
    """
    n = int(config["num_baseline_samples"])
    b_max = float(config["max_baseline"])
    if n < 2:
        raise ValueError("mcrb.num_baseline_samples must be >= 2")
    baselines = np.linspace(0.0, b_max, n)
    return baselines, ActionSpace.lateral_sweep(baselines, max_translation=b_max)
