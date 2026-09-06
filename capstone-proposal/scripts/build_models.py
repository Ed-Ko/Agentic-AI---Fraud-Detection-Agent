#!/usr/bin/env python3
"""Run one reproducible temporal baseline experiment and persist its metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from check_environment import check_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate Elliptic rule and XGBoost baselines once.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/baseline_results.json"))
    args = parser.parse_args()
    problems = check_environment(args.data_dir)
    if problems:
        print("Cannot build models; run `make check` and resolve:", file=sys.stderr)
        print("\n".join(f"- {problem}" for problem in problems), file=sys.stderr)
        return 1

    from src.baselines.run import run_baselines

    results = run_baselines(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=False))
    print(f"\nSaved metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
