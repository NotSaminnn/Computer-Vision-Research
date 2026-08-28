"""External dataset registry: status, licences, layouts and validation.

Rules this module enforces (research plan section 20, integrity rules section 29):

* **never** silently download a large dataset;
* refuse automated download unless BOTH the licence and the download permission
  are recorded as ``verified``;
* report expected storage and required authentication before anything happens;
* when a dataset requires manual acquisition, print the exact instructions;
* never fabricate availability -- an unconfirmed dataset is reported as
  ``ACCESS UNVERIFIED``, not as "available".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from intervene3d.config.loader import repo_root
from intervene3d.reproducibility.hashing import sha256_file

DEFAULT_REGISTRY = "configs/datasets/external.yaml"
STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "ACCESS UNVERIFIED"


@dataclass
class ExternalDataset:
    """One registered external dataset."""

    key: str
    payload: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.payload[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def title(self) -> str:
        return str(self.payload.get("title", self.key))

    @property
    def status(self) -> str:
        return str(self.payload.get("status", STATUS_UNVERIFIED))

    @property
    def licence_status(self) -> str:
        return str(self.payload.get("licence_status", STATUS_UNVERIFIED))

    @property
    def automated_download_permitted(self) -> bool:
        return self.payload.get("automated_download_permitted") is True

    @property
    def may_auto_download(self) -> bool:
        """Automated download is allowed only when licence AND permission are verified."""
        return self.licence_status == STATUS_VERIFIED and self.automated_download_permitted

    @property
    def local_root(self) -> Path:
        return repo_root() / str(self.payload.get("expected_layout", f"data/raw/{self.key}/")).split("{")[0].rstrip("/")

    def instructions(self) -> str:
        return str(self.payload.get("instructions", "No instructions recorded.")).rstrip()

    def describe(self) -> str:
        p = self.payload
        size = p.get("approx_size_gb")
        return "\n".join(
            [
                f"{self.key}  --  {self.title}",
                f"  venue                 : {p.get('venue', 'unknown')}",
                f"  arxiv                 : {p.get('arxiv', 'n/a')}",
                f"  official source       : {p.get('official_repo', 'unknown')}",
                f"  registry status       : {self.status}",
                f"  licence               : {p.get('licence', 'unknown')}  [{self.licence_status}]",
                f"  automated download    : {'PERMITTED' if self.may_auto_download else 'NOT PERMITTED (manual only)'}",
                f"  authentication needed : {p.get('authentication_required', 'unknown')}",
                f"  approx. storage       : {f'{size} GB' if size else 'UNKNOWN -- check the official page before downloading'}",
                f"  expected layout       : {p.get('expected_layout', 'n/a')}",
                f"  role in Intervene3D   : {p.get('intervene3d_role', 'n/a')}",
                f"  priority              : {p.get('priority', 'n/a')}/5",
                "  contents              : " + str(p.get("contents", "unknown")),
            ]
        )


class ExternalRegistry:
    """The parsed ``configs/datasets/external.yaml``."""

    def __init__(self, path: Path | str = DEFAULT_REGISTRY) -> None:
        path = Path(path)
        if not path.is_absolute():
            path = repo_root() / path
        if not path.exists():
            raise FileNotFoundError(f"external dataset registry not found at {path}")
        self.path = path
        with path.open("r", encoding="utf-8") as handle:
            self.payload = yaml.safe_load(handle) or {}
        self.datasets = {k: ExternalDataset(k, v) for k, v in (self.payload.get("datasets") or {}).items()}

    def __contains__(self, key: str) -> bool:
        return key in self.datasets

    def __getitem__(self, key: str) -> ExternalDataset:
        if key not in self.datasets:
            raise KeyError(f"unknown dataset {key!r}; registered: {sorted(self.datasets)}")
        return self.datasets[key]

    def keys(self) -> list[str]:
        return sorted(self.datasets)

    def by_priority(self) -> list[ExternalDataset]:
        return sorted(self.datasets.values(), key=lambda d: (-int(d.payload.get("priority", 0)), d.key))

    def summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for d in self.by_priority():
            rows.append(
                {
                    "key": d.key,
                    "title": d.title,
                    "venue": d.payload.get("venue", ""),
                    "priority": d.payload.get("priority", 0),
                    "status": d.status,
                    "licence_status": d.licence_status,
                    "auto_download": d.may_auto_download,
                    "present_locally": d.local_root.exists(),
                }
            )
        return rows


def validate_external_dataset(dataset: ExternalDataset) -> dict[str, Any]:
    """Check what is present on disk for one external dataset.

    Absence is a normal, reported outcome -- external datasets are deliberately
    not fetched by the preliminary pipeline.
    """
    root = dataset.local_root
    if not root.exists():
        return {
            "dataset": dataset.key,
            "present": False,
            "status": "NOT DOWNLOADED",
            "root": str(root),
            "detail": "external data is never fetched automatically; see the instructions",
            "instructions": dataset.instructions(),
        }
    files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    manifest_path = root / "manifest.json"
    checksum_status = "no manifest.json -- run the dataset's own preparation step"
    if manifest_path.exists():
        from intervene3d.utils.io import load_json

        recorded = load_json(manifest_path).get("files", {})
        bad = [rel for rel, dig in recorded.items() if not (root / rel).exists() or sha256_file(root / rel) != dig]
        checksum_status = "all checksums match" if not bad else f"{len(bad)} checksum mismatches: {bad[:5]}"
    return {
        "dataset": dataset.key,
        "present": True,
        "status": "PRESENT",
        "root": str(root),
        "n_files": len(files),
        "size_gb": round(total_bytes / (1 << 30), 4),
        "checksums": checksum_status,
    }
