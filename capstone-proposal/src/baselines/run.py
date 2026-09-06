"""Run the rule and XGBoost baselines on the fixed Elliptic temporal split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_elliptic_dataset
from src.data.preprocessing import temporal_split
from src.evaluation.evaluate import evaluate_model

from .rules import RuleBasedFraudDetector
from .xgboost import XGBoostFraudDetector


def select_f1_threshold(y_true: pd.Series, scores: pd.Series) -> float:
    """Select a deterministic score threshold using validation labels only."""
    if y_true.empty or len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires both classes in validation data.")
    candidates = np.unique(scores.to_numpy())
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        prediction = scores >= threshold
        tp = int(((prediction == 1) & (y_true == 1)).sum())
        fp = int(((prediction == 1) & (y_true == 0)).sum())
        fn = int(((prediction == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


def run_baselines(data_dir: str | Path) -> dict[str, object]:
    """Fit, validate, and test both baselines without test-set tuning.

    Returns validation-calibration details and the requested test metrics:
    precision, recall, F1, PR-AUC, ROC-AUC, and the licit/illicit confusion
    matrix. The data directory must contain the three original Elliptic CSVs.
    """
    dataset = load_elliptic_dataset(data_dir)
    splits = temporal_split(dataset.transactions)

    rule = RuleBasedFraudDetector().fit(splits.train)
    rule_calibration = rule.calibrate(splits.validation)
    rule_threshold = int(rule_calibration["minimum_illicit_votes"]) / len(rule.rules_ or ())

    xgboost = XGBoostFraudDetector().fit(splits.train)
    xgboost_threshold = select_f1_threshold(splits.validation["target"], xgboost.predict_scores(splits.validation))

    return {
        "rule_based": {
            "validation_calibration": rule_calibration,
            "test_metrics": evaluate_model(rule, splits.test, threshold=rule_threshold),
        },
        "xgboost": {
            "validation_threshold": xgboost_threshold,
            "test_metrics": evaluate_model(xgboost, splits.test, threshold=xgboost_threshold),
        },
    }
