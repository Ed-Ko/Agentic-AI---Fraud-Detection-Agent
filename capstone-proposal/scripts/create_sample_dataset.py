#!/usr/bin/env python3
"""Create a small deterministic Elliptic-shaped dataset for development/tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def create_sample_dataset(output_dir: str | Path, rows: int = 100) -> None:
    """Write 100-row features, classes, and edge CSVs without external packages."""
    if rows < 4:
        raise ValueError("rows must be at least 4 so the edge fixture is meaningful")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    transaction_ids = [str(900000000 + index) for index in range(rows)]

    # Features are headerless: txId, time_step, and 165 numeric attributes.
    with (destination / "elliptic_txs_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, tx_id in enumerate(transaction_ids):
            time_step = index % 49 + 1
            label = "unknown" if index % 5 == 0 else ("1" if index % 3 == 0 else "2")
            illicit_offset = 2.0 if label == "1" else 0.0
            features = [round(illicit_offset + (index + column) / 100.0, 6) for column in range(165)]
            writer.writerow([tx_id, time_step, *features])

    with (destination / "elliptic_txs_classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["txId", "class"])
        for index, tx_id in enumerate(transaction_ids):
            label = "unknown" if index % 5 == 0 else ("1" if index % 3 == 0 else "2")
            writer.writerow([tx_id, label])

    with (destination / "elliptic_txs_edgelist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["txId1", "txId2"])
        for index in range(rows):
            writer.writerow([transaction_ids[index], transaction_ids[(index + 1) % rows]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic Elliptic-shaped CSV fixture.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--rows", type=int, default=100)
    args = parser.parse_args()
    create_sample_dataset(args.output_dir, args.rows)
    print(f"Created {args.rows} feature rows, {args.rows} class rows, and {args.rows} edge rows in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
