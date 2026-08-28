"""Centralised logging.

Every experiment writes an identical log stream to stdout and to
``<run_dir>/logs/run.log`` so that the console transcript and the archived log
can never disagree.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_CONFIGURED = False


def setup_logging(log_file: Path | str | None = None, level: int | str = logging.INFO) -> None:
    """Configure the root logger once, optionally teeing to ``log_file``."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        # Only add the file handler for a new run directory.
        if log_file is not None:
            _add_file_handler(root, Path(log_file), level)
        return

    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    stream.setLevel(level)
    root.addHandler(stream)

    if log_file is not None:
        _add_file_handler(root, Path(log_file), level)

    # Matplotlib's font manager and fontTools' PDF font subsetter are extremely
    # chatty at INFO level and would bury the experiment's own log.
    for noisy in ("matplotlib", "PIL", "fontTools", "fontTools.subset", "fontTools.ttLib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def _add_file_handler(root: logging.Logger, log_file: Path, level: int | str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {getattr(h, "baseFilename", None) for h in root.handlers}
    if str(log_file.resolve()) in existing:
        return
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    file_handler.setLevel(level)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger; configures a default stream handler if needed."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
