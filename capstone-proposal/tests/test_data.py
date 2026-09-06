"""Tests for deterministic, leakage-resistant Elliptic data utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import dataset_summary, load_elliptic_dataset
from src.data.preprocessing import Standardizer, temporal_split
from src.baselines.rules import RuleBasedFraudDetector


def _write_fixture(directory: Path) -> None:
    rows = []
    for tx_id, step in [("a", 1), ("b", 2), ("c", 3), ("d", 4)]:
        rows.append([tx_id, step] + [float(step)] * 165)
    pd.DataFrame(rows).to_csv(directory / "elliptic_txs_features.csv", index=False, header=False)
    pd.DataFrame({"txId": ["a", "b", "c", "d"], "class": ["1", "2", "1", "unknown"]}).to_csv(
        directory / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame({"txId1": ["a", "b"], "txId2": ["b", "c"]}).to_csv(
        directory / "elliptic_txs_edgelist.csv", index=False
    )


def test_loader_joins_original_files_and_reports_local_counts(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset = load_elliptic_dataset(tmp_path)

    assert dataset.transactions.columns[:3].tolist() == ["txId", "time_step", "feature_0"]
    assert dataset_summary(dataset) == {
        "transactions": 4,
        "illicit_transactions": 2,
        "licit_transactions": 1,
        "unknown_transactions": 1,
        "edges": 2,
        "node_feature_dimensionality": 166,
        "anonymous_feature_dimensionality": 165,
        "time_steps": 4,
    }


def test_temporal_split_is_disjoint_forward_and_excludes_unknown() -> None:
    transactions = pd.DataFrame(
        {"txId": ["a", "b", "c", "d"], "time_step": [1, 2, 3, 4], "class": ["1", "2", "1", "unknown"]}
    )
    splits = temporal_split(transactions, [1, 2], [3], [4])

    assert splits.train["txId"].tolist() == ["a", "b"]
    assert splits.validation["target"].tolist() == [1]
    assert splits.test.empty
    with pytest.raises(ValueError, match="overlap"):
        temporal_split(transactions, [1, 2], [2, 3], [4])


def test_standardizer_uses_only_training_statistics() -> None:
    train = pd.DataFrame({"txId": ["a", "b"], "time_step": [1, 2], "class": ["1", "2"], "feature_0": [0.0, 2.0]})
    test = pd.DataFrame({"txId": ["c"], "time_step": [3], "class": ["1"], "feature_0": [100.0]})
    scaler = Standardizer().fit(train)

    assert scaler.mean_.loc["feature_0"] == 1.0
    assert scaler.transform(train)["feature_0"].tolist() == [-1.0, 1.0]
    assert scaler.transform(test)["feature_0"].tolist() == [99.0]
    with pytest.raises(RuntimeError, match="fitted"):
        Standardizer().transform(test)


def test_rule_thresholds_are_fit_on_train_and_calibrated_on_validation() -> None:
    train = pd.DataFrame(
        {"txId": ["a", "b", "c", "d"], "class": ["2", "2", "1", "1"], "target": [0, 0, 1, 1], "feature_0": [0.0, 1.0, 9.0, 10.0]}
    )
    validation = pd.DataFrame(
        {"txId": ["e", "f", "g", "h"], "class": ["2", "2", "1", "1"], "target": [0, 0, 1, 1], "feature_0": [2.0, 3.0, 8.0, 9.0]}
    )
    detector = RuleBasedFraudDetector(max_rules=1).fit(train)
    calibration = detector.calibrate(validation)

    assert detector.rules_[0].threshold == 5.0
    assert calibration["within_accuracy_target"] is False
    assert detector.predict(validation).tolist() == [0, 0, 1, 1]
