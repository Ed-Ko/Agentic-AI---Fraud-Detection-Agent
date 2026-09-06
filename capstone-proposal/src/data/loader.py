"""Validated, schema-aware loading for the original Elliptic Bitcoin CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"
VALID_LABELS = frozenset({"1", "2", "unknown"})
EXPECTED_FEATURE_COLUMNS = 167


@dataclass(frozen=True)
class EllipticDataset:
    """Original files loaded into validated, consistently named tables."""

    transactions: pd.DataFrame
    edges: pd.DataFrame


def _require_file(data_dir: Path, name: str) -> Path:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Expected Elliptic file was not found: {path}")
    return path


def load_elliptic_dataset(data_dir: str | Path) -> EllipticDataset:
    """Load the three released Elliptic files without changing row order.

    The headerless feature file is assigned stable names: ``txId``,
    ``time_step``, and ``feature_0`` through ``feature_164``.  Labels are
    merged one-to-one onto transactions; missing labels are rejected because
    they indicate an incomplete or mismatched release.
    """
    directory = Path(data_dir)
    feature_path = _require_file(directory, FEATURES_FILE)
    class_path = _require_file(directory, CLASSES_FILE)
    edge_path = _require_file(directory, EDGES_FILE)

    raw_features = pd.read_csv(feature_path, header=None)
    if raw_features.shape[1] != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COLUMNS} feature columns, found "
            f"{raw_features.shape[1]} in {feature_path}."
        )
    feature_columns = ["txId", "time_step"] + [
        f"feature_{index}" for index in range(EXPECTED_FEATURE_COLUMNS - 2)
    ]
    raw_features.columns = feature_columns

    classes = pd.read_csv(class_path, dtype={"txId": "string", "class": "string"})
    if set(classes.columns) != {"txId", "class"}:
        raise ValueError(f"{class_path} must contain exactly txId and class columns.")
    edges = pd.read_csv(edge_path, dtype={"txId1": "string", "txId2": "string"})
    if set(edges.columns) != {"txId1", "txId2"}:
        raise ValueError(f"{edge_path} must contain exactly txId1 and txId2 columns.")

    transactions = raw_features.copy()
    transactions["txId"] = transactions["txId"].astype("string")
    transactions["time_step"] = pd.to_numeric(transactions["time_step"], errors="raise")
    if transactions["txId"].duplicated().any():
        raise ValueError("Feature file contains duplicate txId values.")
    if classes["txId"].duplicated().any():
        raise ValueError("Class file contains duplicate txId values.")
    invalid_labels = set(classes["class"].dropna().unique()) - VALID_LABELS
    if invalid_labels or classes["class"].isna().any():
        raise ValueError(f"Unexpected class labels: {sorted(invalid_labels)}")

    transactions = transactions.merge(classes, how="left", on="txId", validate="one_to_one", sort=False)
    if transactions["class"].isna().any():
        raise ValueError("Class file does not label every transaction in the feature file.")
    node_ids = set(transactions["txId"])
    missing_endpoints = set(edges["txId1"]) | set(edges["txId2"])
    missing_endpoints -= node_ids
    if missing_endpoints:
        raise ValueError("Edge file contains endpoints absent from the feature file.")
    return EllipticDataset(transactions=transactions, edges=edges)


def dataset_summary(dataset: EllipticDataset) -> dict[str, int]:
    """Return counts computed from the supplied local dataset, not constants."""
    labels = dataset.transactions["class"]
    return {
        "transactions": len(dataset.transactions),
        "illicit_transactions": int((labels == "1").sum()),
        "licit_transactions": int((labels == "2").sum()),
        "unknown_transactions": int((labels == "unknown").sum()),
        "edges": len(dataset.edges),
        "node_feature_dimensionality": EXPECTED_FEATURE_COLUMNS - 1,
        "anonymous_feature_dimensionality": EXPECTED_FEATURE_COLUMNS - 2,
        "time_steps": int(dataset.transactions["time_step"].nunique()),
    }
