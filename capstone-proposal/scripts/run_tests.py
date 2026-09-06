#!/usr/bin/env python3
"""Run the project test suite using the current Python interpreter."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / ".python-packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))


def main() -> int:
    if importlib.util.find_spec("pytest") is None:
        print("pytest is missing. Install dependencies with: python3 -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    environment = os.environ.copy()
    if LOCAL_PACKAGES.is_dir():
        environment["PYTHONPATH"] = str(LOCAL_PACKAGES) + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
