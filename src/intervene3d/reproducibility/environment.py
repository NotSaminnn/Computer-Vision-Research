"""Capture of environment, hardware and Git metadata."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

#: Environment variables that can change numerical results, so they are recorded.
RELEVANT_ENV_VARS = (
    "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG", "MPLBACKEND", "INTERVENE3D_DATA_ROOT",
)

_TRACKED_PACKAGES = ("numpy", "matplotlib", "PyYAML", "SciencePlots", "torch", "scipy", "pytest")


def git_metadata(repo: Path | str | None = None) -> dict[str, Any]:
    """Commit, branch and dirty state.  Never raises -- a missing Git is recorded."""
    cwd = Path(repo) if repo else Path(__file__).resolve().parents[3]

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit or "UNAVAILABLE (not a git repository, or git is not installed)",
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "UNAVAILABLE",
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:50] if status else [],
        "remote": _run(["git", "config", "--get", "remote.origin.url"]) or "",
        "describe": _run(["git", "describe", "--always", "--dirty"]) or "",
    }


def package_versions() -> dict[str, str]:
    """Versions of the tracked packages, plus the full frozen environment."""
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def frozen_environment() -> list[str]:
    """A ``pip freeze``-equivalent list, produced without invoking pip."""
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])


def hardware_metadata() -> dict[str, Any]:
    """CPU / GPU / CUDA description.  Records absence explicitly, never guesses."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
    }
    try:  # pragma: no cover - torch is optional
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda or "none"
        info["gpu"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
        info["cuda_version"] = "not applicable (torch not installed)"
        info["gpu"] = []
    return info


def environment_metadata(repo: Path | str | None = None) -> dict[str, Any]:
    """The full block archived in every run directory."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "hardware": hardware_metadata(),
        "packages": package_versions(),
        "environment_variables": {k: os.environ.get(k, "") for k in RELEVANT_ENV_VARS},
        "git": git_metadata(repo),
        "cwd": str(Path.cwd()),
    }


def render_environment_text(meta: dict[str, Any]) -> str:
    """Human-readable ``environment.txt``."""
    hw = meta["hardware"]
    lines = [
        "Intervene3D run environment",
        "=" * 60,
        f"python           : {meta['python_version'].splitlines()[0]}",
        f"implementation   : {meta['python_implementation']}",
        f"executable       : {meta['python_executable']}",
        f"os               : {meta['os']} {meta['os_release']}",
        f"platform         : {hw['platform']}",
        f"machine          : {hw['machine']}",
        f"cpu_count        : {hw['cpu_count']}",
        f"hostname         : {hw['hostname']}",
        f"torch            : {hw['torch_version']}",
        f"cuda_available   : {hw['cuda_available']}",
        f"cuda_version     : {hw['cuda_version']}",
        f"gpu              : {', '.join(hw['gpu']) if hw['gpu'] else 'none (CPU-only run)'}",
        "",
        "git",
        "-" * 60,
        f"commit           : {meta['git']['commit']}",
        f"branch           : {meta['git']['branch']}",
        f"dirty            : {meta['git']['dirty']}",
        "",
        "packages",
        "-" * 60,
    ]
    lines += [f"{k:16s} : {v}" for k, v in sorted(meta["packages"].items())]
    lines += ["", "relevant environment variables", "-" * 60]
    lines += [f"{k:28s} = {v!r}" for k, v in sorted(meta["environment_variables"].items())]
    lines += ["", "frozen environment", "-" * 60] + frozen_environment()
    return "\n".join(lines) + "\n"
