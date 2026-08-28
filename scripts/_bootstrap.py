"""Make ``src/`` importable when the package is not pip-installed.

Every script imports this first so `python scripts/<x>.py` works from a clean
checkout without an editable install, while still preferring an installed
package if one is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
