"""Immutable run directories.

    experiments/<experiment_name>/run_<UTC timestamp>_seed<N>_<config hash>/

Every run gets its own directory and **an existing run is never overwritten**;
if a directory with the same name somehow exists, a numeric suffix is appended
rather than any file being replaced.  A run directory holds everything another
researcher needs to understand exactly how its numbers were produced.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intervene3d.config.loader import config_hash, dump_config, full_config_hash
from intervene3d.reproducibility.environment import environment_metadata, render_environment_text
from intervene3d.utils.io import dump_json, write_text
from intervene3d.utils.logging import setup_logging

SUBDIRS = ("logs", "checkpoints", "predictions", "metrics", "tables", "figures")


@dataclass
class RunDirectory:
    """A created run directory and the metadata written into it."""

    path: Path
    run_id: str
    experiment_name: str
    seed: int
    config_hash: str
    created_utc: str

    # ------------------------------------------------------------- convenience
    @property
    def logs(self) -> Path:
        return self.path / "logs"

    @property
    def metrics(self) -> Path:
        return self.path / "metrics"

    @property
    def figures(self) -> Path:
        return self.path / "figures"

    @property
    def tables(self) -> Path:
        return self.path / "tables"

    @property
    def predictions(self) -> Path:
        return self.path / "predictions"

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "created_utc": self.created_utc,
        }


def reproduction_command(config_path: str | Path, seed: int, overrides: list[str] | None = None) -> str:
    """The exact command that reproduces a run."""
    parts = ["python scripts/run_experiment.py", f"  --config {config_path}", f"  --seed {seed}"]
    for item in overrides or []:
        parts.append(f"  --set {item}")
    return " \\\n".join(parts)


def create_run_directory(
    config: Mapping[str, Any],
    *,
    seed: int,
    root: Path | str = "experiments",
    command: str | None = None,
    overrides: list[str] | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RunDirectory:
    """Create a unique run directory and write all reproducibility metadata."""
    experiment_name = str(config["experiment"]["name"])
    chash = config_hash(config)
    created = time.gmtime()
    stamp = time.strftime("%Y%m%d_%H%M%S", created)
    run_id = f"run_{stamp}_seed{seed}_{chash}"

    base = Path(root) / experiment_name
    path = base / run_id
    suffix = 1
    while path.exists():  # never overwrite an existing run
        path = base / f"{run_id}_{suffix}"
        suffix += 1
    for sub in SUBDIRS:
        (path / sub).mkdir(parents=True, exist_ok=True)

    setup_logging(path / "logs" / "run.log")

    dump_config(path / "config.yaml", config)
    cmd = command or " ".join([sys.executable.split("/")[-1], *sys.argv])
    repro = reproduction_command(config.get("_config_path", "<config>"), seed, overrides)
    write_text(path / "command.txt", f"{cmd}\n\n# reproduction command\n{repro}\n")

    env = environment_metadata()
    write_text(path / "git_commit.txt", f"{env['git']['commit']}\n")
    write_text(path / "environment.txt", render_environment_text(env))
    dump_json(path / "environment.json", env)
    dump_json(
        path / "dataset_manifest.json",
        dataset_manifest if dataset_manifest is not None else {"status": "NOT RUN -- no dataset attached"},
    )

    manifest = {
        "experiment_name": experiment_name,
        "run_id": path.name,
        "seed": int(seed),
        "config_hash": chash,
        "config_hash_full": full_config_hash(config),
        "config_path": config.get("_config_path"),
        "git_commit": env["git"]["commit"],
        "git_branch": env["git"]["branch"],
        "git_dirty": env["git"]["dirty"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", created),
        "command": cmd,
        "reproduction_command": repro,
        "overrides": list(overrides or []),
        "status": "running",
        "dataset_manifest": "dataset_manifest.json",
        "metrics_file": None,
        "figures": [],
        "environment": {
            "python": env["python_version"].splitlines()[0],
            "os": f"{env['os']} {env['os_release']}",
            "hostname": env["hardware"]["hostname"],
            "cuda_available": env["hardware"]["cuda_available"],
            "packages": env["packages"],
        },
        **dict(extra or {}),
    }
    dump_json(path / "run_manifest.json", manifest)

    return RunDirectory(
        path=path,
        run_id=path.name,
        experiment_name=experiment_name,
        seed=int(seed),
        config_hash=chash,
        created_utc=manifest["created_utc"],
    )
