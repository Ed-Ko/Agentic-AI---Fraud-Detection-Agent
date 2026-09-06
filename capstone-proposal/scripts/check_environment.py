#!/usr/bin/env python3
"""Preflight checks for the Elliptic capstone workflow; makes no changes."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / ".python-packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))


REQUIRED_MODULES = {"pandas": "pandas", "numpy": "numpy", "pytest": "pytest", "scikit-learn": "sklearn", "xgboost": "xgboost"}
REQUIRED_DATA_FILES = ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv")


def check_environment(data_dir: Path) -> list[str]:
    """Return a list of unmet prerequisites, without installing anything."""
    problems: list[str] = []
    for package, module in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            problems.append(f"Missing Python package: {package}")
    for filename in REQUIRED_DATA_FILES:
        if not (data_dir / filename).is_file():
            problems.append(f"Missing dataset file: {data_dir / filename}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dependencies and original Elliptic CSV files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing the three original CSV files.")
    args = parser.parse_args()
    problems = check_environment(args.data_dir)
    if problems:
        print("Preflight failed:")
        for problem in problems:
            print(f"- {problem}")
        print("\nInstall dependencies with: python3 -m pip install -r requirements.txt")
        print("Place the original Elliptic CSV files in data/raw/ (or pass --data-dir).")
        return 1
    print("Preflight passed: dependencies and all three Elliptic CSV files are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
