"""Content hashing for datasets, checkpoints and result files."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path | str, *, pattern: str = "*") -> dict[str, str]:
    """Checksums of every file under ``root``, keyed by relative path."""
    root = Path(root)
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob(pattern))
        if p.is_file()
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
