"""Validation of a generated synthetic dataset.

Checks the properties the benchmark's scientific claims depend on:

1. **checksums** -- every file matches the manifest;
2. **matched counterfactuals** -- all causal variants of a base scene are
   pixel-identical at the reference view;
3. **no split leakage** -- no base scene appears in more than one split;
4. **non-trivial benchmark** -- both resolvable and non-resolvable cases exist
   (a benchmark in which every ambiguity resolves would undermine the research
   question);
5. **determinism metadata** -- seed, config hash and action set are recorded;
6. **array integrity** -- shapes agree with the declared landmark counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.data.dataset import SyntheticDataset
from intervene3d.data.splits import check_no_leakage
from intervene3d.data.synthetic.dataset_writer import sha256_file


@dataclass
class ValidationReport:
    """Result of validating one dataset."""

    dataset: str
    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", *, severity: str = "error") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail, "severity": severity})
        if not ok and severity == "error":
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": sum(1 for c in self.checks if not c["ok"]),
            "checks": self.checks,
        }

    def render(self) -> str:
        lines = [f"Dataset validation: {self.dataset}", "-" * 60]
        for c in self.checks:
            mark = "PASS" if c["ok"] else ("FAIL" if c["severity"] == "error" else "WARN")
            lines.append(f"  [{mark}] {c['check']}{': ' + c['detail'] if c['detail'] else ''}")
        lines.append("-" * 60)
        lines.append(f"  RESULT: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def validate_dataset(root: Path | str, *, verify_checksums: bool = True) -> ValidationReport:
    root = Path(root)
    report = ValidationReport(dataset=str(root))

    try:
        ds = SyntheticDataset(root)
    except Exception as exc:  # noqa: BLE001
        report.add("load", False, f"{type(exc).__name__}: {exc}")
        return report
    report.add("load", True, f"{len(ds)} scene variants")

    # 1. checksums
    if verify_checksums:
        files = ds.manifest.get("files", {})
        bad = [rel for rel, digest in files.items() if not (root / rel).exists() or sha256_file(root / rel) != digest]
        report.add("checksums", not bad, f"{len(files) - len(bad)}/{len(files)} files match" + (f"; mismatched: {bad[:5]}" if bad else ""))

    # 2. matched counterfactuals
    by_base: dict[str, list] = {}
    for scene in ds:
        by_base.setdefault(scene.base_scene_id, []).append(scene)
    max_dev, checked = 0.0, 0
    for scenes in by_base.values():
        if len(scenes) < 2:
            continue
        ref = scenes[0].reference_observation().feature
        for other in scenes[1:]:
            f = other.reference_observation().feature
            if not np.array_equal(ref.visible, f.visible):
                max_dev = float("inf")
                break
            m = ref.visible
            if np.any(m):
                max_dev = max(max_dev, float(np.nanmax(np.abs(ref.uv[m] - f.uv[m]))))
            checked += 1
    report.add(
        "matched_counterfactuals",
        max_dev < 1e-9,
        f"max reference-view pixel deviation across variants = {max_dev:.3e} over {checked} comparisons",
    )

    # 3. leakage
    leaked = check_no_leakage(ds.records)
    report.add("split_leakage", not leaked, "no base scene spans splits" if not leaked else f"leaked: {leaked[:5]}")

    # 4. non-trivial benchmark
    resolvable = np.array([bool(r["resolvable"]) for r in ds.records])
    n_res, n_non = int(np.count_nonzero(resolvable)), int(np.count_nonzero(~resolvable))
    report.add(
        "contains_non_identifiable_cases",
        n_non > 0,
        f"{n_res} resolvable / {n_non} non-resolvable",
        severity="error" if len(ds) >= 8 else "warning",
    )
    report.add("contains_resolvable_cases", n_res > 0, f"{n_res} resolvable")

    # 5. determinism metadata
    have = all(ds.manifest.get(k) is not None for k in ("seed", "config_hash", "action_space"))
    report.add("reproducibility_metadata", have, "seed, config_hash and action_space recorded")

    # 6. array integrity
    problems: list[str] = []
    for scene in ds:
        a = scene.arrays()
        m = int(scene.record["n_landmarks"])
        n_actions = int(scene.record["action_set_size"])
        if a["ref_uv"].shape != (m, 2):
            problems.append(f"{scene.scene_id}: ref_uv {a['ref_uv'].shape} != ({m}, 2)")
        if a["obs_uv"].shape != (n_actions, m, 2):
            problems.append(f"{scene.scene_id}: obs_uv {a['obs_uv'].shape} != ({n_actions}, {m}, 2)")
        if len(problems) > 3:
            break
    report.add("array_shapes", not problems, "; ".join(problems) if problems else "all arrays well-shaped")

    return report
