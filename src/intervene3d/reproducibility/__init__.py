"""Reproducibility: seeds, environment capture, run directories, registry, hashing.

One utility, used everywhere.  Every experiment records the seed, Python
version, OS, GPU/CUDA availability, package versions, Git commit and dirty
state, the configuration and its hash, the dataset manifest and its checksums,
the exact command, a UTC timestamp, hostname, the relevant environment
variables, and the deterministic settings that were applied.
"""

from intervene3d.reproducibility.environment import (
    environment_metadata,
    frozen_environment,
    git_metadata,
    hardware_metadata,
    package_versions,
    render_environment_text,
)
from intervene3d.reproducibility.hashing import sha256_file, sha256_text, sha256_tree
from intervene3d.reproducibility.manifest import (
    finalise_run,
    load_registry,
    registry_path,
    runs_for_experiment,
)
from intervene3d.reproducibility.run_dir import (
    RunDirectory,
    create_run_directory,
    reproduction_command,
)
from intervene3d.reproducibility.seeds import rng_for, set_global_seed

__all__ = [
    "environment_metadata",
    "frozen_environment",
    "git_metadata",
    "hardware_metadata",
    "package_versions",
    "render_environment_text",
    "sha256_file",
    "sha256_text",
    "sha256_tree",
    "finalise_run",
    "load_registry",
    "registry_path",
    "runs_for_experiment",
    "RunDirectory",
    "create_run_directory",
    "reproduction_command",
    "rng_for",
    "set_global_seed",
]
