"""Adapters and the registry for third-party datasets.

No external dataset is required by the preliminary pipeline, the smoke test or
the Phase 1 experiment.  Their status is tracked honestly here and in
``docs/DATASET_MATRIX.md``.
"""

from intervene3d.data.external.registry import (
    DEFAULT_REGISTRY,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    ExternalDataset,
    ExternalRegistry,
    validate_external_dataset,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "STATUS_UNVERIFIED",
    "STATUS_VERIFIED",
    "ExternalDataset",
    "ExternalRegistry",
    "validate_external_dataset",
]
