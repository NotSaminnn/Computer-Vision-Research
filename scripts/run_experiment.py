#!/usr/bin/env python3
"""Run an Intervene3D experiment.

    python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed 42
    python scripts/run_experiment.py --config configs/smoke_test.yaml --seed 0
    python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml \
        --seed 3 --set model.abstention.tau=0.9

Creates a unique, immutable run directory under ``experiments/<name>/`` holding
the config, command, git commit, environment, dataset manifest, run manifest,
logs, metrics, predictions, tables and figures, and prints the exact command
that reproduces it.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from intervene3d.experiments.runner import main

if __name__ == "__main__":
    sys.exit(main())
