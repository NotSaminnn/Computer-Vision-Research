"""Seed handling and deterministic settings.

The preliminary pipeline is pure NumPy, so determinism is achieved by seeding
rather than by disabling nondeterministic kernels.  Where PyTorch is present (it
is optional) its deterministic flags are set too, so the guarantee does not
quietly weaken when a GPU model is plugged in later.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every RNG this process might use and return what was configured."""
    seed = int(seed)
    report: dict[str, Any] = {"seed": seed, "python_random": True, "numpy": True, "deterministic": deterministic}

    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:  # pragma: no cover - torch is optional
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # noqa: BLE001
                report["torch_deterministic_algorithms_error"] = str(exc)
        report["torch"] = torch.__version__
    except ImportError:
        report["torch"] = "not installed"
    return report


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    """A named, reproducible sub-stream of the master seed.

    Using ``default_rng([seed, *stream])`` rather than deriving generators
    sequentially means adding a new consumer never perturbs the draws of an
    existing one, so a run's results stay stable as the code grows.
    """
    return np.random.default_rng([int(seed), *[int(s) for s in stream]])
