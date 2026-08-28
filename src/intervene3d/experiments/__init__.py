"""Experiment definitions, the runner and the registry.

**All experiments live under one top-level directory, ``experiments/``, and every
run gets its own immutable directory that is never overwritten.**
"""

from intervene3d.experiments.methods import MethodSpec, build_engine, build_transition
from intervene3d.experiments.registry import (
    experiment_names,
    load_registry,
    load_run_metrics,
    runs_for_experiment,
    summarise_registry,
)
from intervene3d.experiments.runner import EXPERIMENT_KINDS, main, run_experiment

__all__ = [
    "MethodSpec",
    "build_engine",
    "build_transition",
    "experiment_names",
    "load_registry",
    "load_run_metrics",
    "runs_for_experiment",
    "summarise_registry",
    "EXPERIMENT_KINDS",
    "main",
    "run_experiment",
]
