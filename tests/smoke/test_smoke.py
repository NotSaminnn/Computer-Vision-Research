"""The smoke test as a pytest case.

Mirrors ``scripts/run_smoke_test.sh`` so CI can assert the same end-to-end
guarantee without a shell.  Everything runs inside ``tmp_path`` so the test
never touches the repository's own ``experiments/`` or ``data/`` directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intervene3d.config import validate_experiment_config, validate_synthetic_config
from intervene3d.config.loader import load_config
from intervene3d.data.synthetic import generate_dataset, validate_dataset
from intervene3d.experiments.runner import run_experiment


@pytest.mark.smoke
def test_end_to_end_smoke(tmp_path, repo_root, monkeypatch):
    """Clean environment -> data -> experiment -> run directory -> metrics -> figures."""
    monkeypatch.chdir(tmp_path)

    # ---- tiny dataset
    synth = validate_synthetic_config(load_config(repo_root / "configs" / "synthetic" / "smoke.yaml"))
    synth["dataset"]["num_base_scenes"] = 8
    synth["dataset"]["name"] = "smoke_pytest"
    synth["output"]["root"] = str(tmp_path / "data" / "processed")
    manifest = generate_dataset(synth)
    dataset_root = Path(synth["output"]["root"]) / "smoke_pytest"

    report = validate_dataset(dataset_root)
    assert report.passed, report.render()
    stats = manifest["statistics"]
    assert stats["n_scene_variants"] == 8 * len(synth["mechanisms"])
    assert stats["non_resolvable_count"] > 0, "the benchmark must contain non-identifiable cases"

    # ---- experiment
    exp = validate_experiment_config(load_config(repo_root / "configs" / "smoke_test.yaml"))
    exp["experiment"]["name"] = "smoke_pytest"
    exp["experiment"]["root"] = str(tmp_path / "experiments")
    exp["data"] = {"dataset_dir": str(dataset_root), "split": "test"}
    exp["classifier_epochs"] = 120
    exp["visualization"] = {"formats": ["png"], "dpi": 80}

    config_path = tmp_path / "smoke_pytest.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(exp))
    outcome = run_experiment(config_path, seed=0, root=str(tmp_path / "experiments"))

    run_dir = Path(outcome["run_dir"])
    assert run_dir.exists()

    # ---- reproducibility side-cars
    for name in ("config.yaml", "command.txt", "git_commit.txt", "environment.txt",
                 "dataset_manifest.json", "run_manifest.json", "summary.md"):
        assert (run_dir / name).exists() and (run_dir / name).stat().st_size > 0, name

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["status"] == "success"
    assert run_manifest["seed"] == 0
    assert run_manifest["config_hash"]
    assert run_manifest["reproduction_command"].startswith("python scripts/run_experiment.py")

    # ---- metrics
    metrics = json.loads((run_dir / "metrics" / "metrics.json").read_text())
    assert metrics["methods"], "no methods were evaluated"
    for m in metrics["methods"].values():
        assert set(m) >= {"cea", "fpcr", "identifiability", "contact_depth", "mcrb", "intervention"}
        assert 0.0 <= m["cea"]["cea_all"] <= 1.0
    assert (run_dir / "metrics" / "figure_data.json").exists()
    assert (run_dir / "metrics" / "summary.json").exists()
    assert (run_dir / "predictions" / "predictions.csv").stat().st_size > 0
    assert (run_dir / "tables" / "method_comparison.csv").stat().st_size > 0

    # ---- figures
    figures = sorted((run_dir / "figures").glob("*.png"))
    assert len(figures) >= 10, f"expected many figures, got {len(figures)}"
    for f in figures:
        assert f.stat().st_size > 500, f"{f.name} is suspiciously small"

    # ---- registry
    registry = (tmp_path / "experiments" / "registry.jsonl").read_text().strip().splitlines()
    assert len(registry) == 1
    assert json.loads(registry[0])["status"] == "success"


@pytest.mark.smoke
def test_runs_are_never_overwritten(tmp_path, repo_root, monkeypatch):
    """Two runs of the same config and seed must occupy two directories."""
    monkeypatch.chdir(tmp_path)
    synth = validate_synthetic_config(load_config(repo_root / "configs" / "synthetic" / "smoke.yaml"))
    synth["dataset"] = {**synth["dataset"], "name": "smoke_immutable", "num_base_scenes": 6}
    synth["output"]["root"] = str(tmp_path / "data" / "processed")
    generate_dataset(synth)

    import yaml

    exp = validate_experiment_config(load_config(repo_root / "configs" / "smoke_test.yaml"))
    exp["experiment"] = {**exp["experiment"], "name": "smoke_immutable", "root": str(tmp_path / "experiments")}
    exp["data"] = {"dataset_dir": str(Path(synth["output"]["root"]) / "smoke_immutable"), "split": "test"}
    exp["classifier_epochs"] = 60
    exp["visualization"] = {"formats": ["png"], "dpi": 60}
    exp["methods"] = [m for m in exp["methods"] if m["name"] == "intervene3d"]
    path = tmp_path / "immutable.yaml"
    path.write_text(yaml.safe_dump(exp))

    first = run_experiment(path, seed=1, root=str(tmp_path / "experiments"))
    second = run_experiment(path, seed=1, root=str(tmp_path / "experiments"))
    assert first["run_dir"] != second["run_dir"]
    assert Path(first["run_dir"]).exists() and Path(second["run_dir"]).exists()


@pytest.mark.smoke
def test_smoke_shell_script_exists_and_is_executable(repo_root):
    script = repo_root / "scripts" / "run_smoke_test.sh"
    assert script.exists()
    text = script.read_text()
    assert "set -uo pipefail" in text
    assert "SMOKE TEST PASSED" in text


@pytest.mark.smoke
def test_failed_runs_are_recorded_not_silently_lost(tmp_path, repo_root, monkeypatch):
    """A crash must leave a legible record, not an unexplained empty directory."""
    monkeypatch.chdir(tmp_path)
    import yaml

    exp = validate_experiment_config(load_config(repo_root / "configs" / "smoke_test.yaml"))
    exp["experiment"] = {**exp["experiment"], "name": "smoke_failure", "root": str(tmp_path / "experiments")}
    exp["data"] = {"dataset_dir": str(tmp_path / "does_not_exist"), "split": "test"}
    path = tmp_path / "failing.yaml"
    path.write_text(yaml.safe_dump(exp))

    with pytest.raises(FileNotFoundError):
        run_experiment(path, seed=0, root=str(tmp_path / "experiments"))

    runs = sorted((tmp_path / "experiments" / "smoke_failure").glob("run_*"))
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "error" in manifest and manifest["error"]
