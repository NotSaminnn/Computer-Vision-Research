"""The generic experiment runner.

    python scripts/run_experiment.py --config configs/experiments/<name>.yaml --seed 42

Creates a unique, immutable run directory, executes the experiment named by
``experiment.kind``, records the outcome in the registry and prints the exact
command that reproduces the run.  Failures are recorded too: a crashed run
leaves a ``run_manifest.json`` with ``status: "failed"`` and the traceback,
rather than an unexplained empty directory.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intervene3d.config.loader import load_config
from intervene3d.config.schema import validate_experiment_config
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory, reproduction_command
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: experiment.kind -> callable(config, run_dir, seed) -> dict
EXPERIMENT_KINDS: dict[str, str] = {
    "phase1_problem_existence": "intervene3d.experiments.phase1:run",
}


def _resolve(kind: str):
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(
            f"unknown experiment.kind {kind!r}; available: {sorted(EXPERIMENT_KINDS)}"
        )
    module_name, func_name = EXPERIMENT_KINDS[kind].split(":")
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def run_experiment(
    config_path: str | Path,
    *,
    seed: int | None = None,
    overrides: list[str] | None = None,
    root: str | Path = "experiments",
) -> dict[str, Any]:
    """Load, validate, execute and record one experiment run."""
    raw = load_config(config_path, overrides=overrides)
    config = validate_experiment_config(raw)
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
    seed = int(config["experiment"]["seed"])
    kind = str(config["experiment"].get("kind", config["experiment"]["name"]))

    seed_report = set_global_seed(seed)
    run_dir = create_run_directory(
        config,
        seed=seed,
        root=config["experiment"].get("root", root),
        command=" ".join([Path(sys.executable).name, *sys.argv]),
        overrides=overrides,
        extra={"experiment_kind": kind, "seed_report": seed_report},
    )
    LOGGER.info("run directory: %s", run_dir.path)
    LOGGER.info("config hash  : %s", run_dir.config_hash)

    started = time.time()
    try:
        func = _resolve(kind)
        outcome = func(config, run_dir, seed)
        # Attach the dataset manifest that the experiment actually used.
        _attach_dataset_manifest(run_dir, outcome)
        manifest = finalise_run(
            run_dir.path,
            status="success",
            metrics_file=outcome.get("metrics_file"),
            figures=outcome.get("figures", []),
            tables=outcome.get("tables", []),
            summary=outcome.get("summary"),
            duration_seconds=round(time.time() - started, 3),
            registry_root=config["experiment"].get("root", root),
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        LOGGER.error("experiment FAILED: %s\n%s", exc, tb)
        finalise_run(
            run_dir.path,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{tb}",
            duration_seconds=round(time.time() - started, 3),
            registry_root=config["experiment"].get("root", root),
        )
        raise

    repro = reproduction_command(config.get("_config_path", str(config_path)), seed, overrides)
    LOGGER.info("experiment finished in %.2f s", time.time() - started)
    print("\n" + "=" * 72)
    print(f"RUN SUCCEEDED: {run_dir.path}")
    print("=" * 72)
    print(f"metrics : {run_dir.path / 'metrics' / 'metrics.json'}")
    print(f"summary : {run_dir.path / 'summary.md'}")
    print(f"figures : {len(outcome.get('figures', []))} files in {run_dir.figures}")
    print("\nreproduce with:\n" + repro + "\n")

    return {"run_dir": str(run_dir.path), "manifest": manifest, **outcome}


def _attach_dataset_manifest(run_dir, outcome: Mapping[str, Any]) -> None:
    """Copy the dataset manifest the experiment reported into the run directory."""
    from intervene3d.utils.io import dump_json, load_json

    metrics_file = outcome.get("metrics_file")
    if not metrics_file:
        return
    metrics = load_json(run_dir.path / metrics_file)
    if "dataset" in metrics:
        dump_json(run_dir.path / "dataset_manifest.json", metrics["dataset"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an Intervene3D experiment in a fresh, immutable run directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed 42\n"
            "  python scripts/run_experiment.py --config configs/smoke_test.yaml --seed 0\n"
            "  python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml \\\n"
            "      --seed 3 --set model.abstention.tau=0.9\n"
        ),
    )
    parser.add_argument("--config", required=True, help="path to the experiment YAML config")
    parser.add_argument("--seed", type=int, default=None, help="random seed (overrides experiment.seed)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="config override of the form dotted.key=value (repeatable)",
    )
    parser.add_argument("--root", default=None, help="experiment root directory (default: experiments/)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs: dict[str, Any] = {"seed": args.seed, "overrides": args.overrides}
    if args.root:
        kwargs["root"] = args.root
    run_experiment(args.config, **kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
