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
        """Where this dataset lives on disk.

        ``expected_layout`` is written for humans and carries two kinds of
        placeholder: a brace alternation naming the subdirectories
        (``.../{train,test}/...``) and a trailing ``...`` meaning "and whatever
        the dataset puts below here". Both are documentation, not path
        components -- keeping either would create a directory literally named
        ``...``.
        """
        layout = str(self.payload.get("expected_layout") or f"data/raw/{self.key}/")
        parts: list[str] = []
        for part in layout.split("{")[0].split("/"):
            if not part or set(part) == {"."}:  # "", ".", "..."
                continue
            parts.append(part)
        return repo_root().joinpath(*parts) if parts else repo_root() / "data" / "raw" / self.key

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
    # A dataset may be acquired one variant at a time, each into its own
    # subdirectory with its own manifest, so look for all of them.
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        checksum_status = "no manifest.json -- run the dataset's own preparation step"
    else:
        from intervene3d.utils.io import load_json

        checked = 0
        bad: list[str] = []
        for manifest_path in manifests:
            base = manifest_path.parent
            for rel, dig in (load_json(manifest_path).get("files") or {}).items():
                checked += 1
                target = base / rel
                if not target.exists() or sha256_file(target) != dig:
                    bad.append(str(target.relative_to(root)))
        checksum_status = (
            f"all {checked} checksums match"
            if not bad
            else f"{len(bad)}/{checked} checksum mismatches: {bad[:5]}"
        )
    return {
        "dataset": dataset.key,
        "present": True,
        "status": "PRESENT",
        "root": str(root),
        "n_files": len(files),
        "size_gb": round(total_bytes / (1 << 30), 4),
        "variants": [str(m.parent.relative_to(root)) for m in manifests],
        "checksums": checksum_status,
    }
