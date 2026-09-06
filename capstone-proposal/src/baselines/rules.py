"""Interpretable, deterministic feature-threshold baseline for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_NON_FEATURE_COLUMNS = frozenset({"txId", "time_step", "class", "target"})


@dataclass(frozen=True)
class FeatureRule:
    """A single feature threshold that votes for the illicit class."""

    feature: str
    threshold: float
    illicit_when_greater: bool
    effect_size: float


class RuleBasedFraudDetector:
    """Classify with the strongest train-derived per-feature fraud rules.

    A rule votes illicit when its feature is on the illicit side of the midpoint
    between train-only class means. The final vote threshold is selected using
    validation labels only; test labels are never inspected.
    """

    def __init__(self, *, max_rules: int = 10) -> None:
        if max_rules < 1:
            raise ValueError("max_rules must be at least one.")
        self.max_rules = max_rules
        self.rules_: tuple[FeatureRule, ...] | None = None
        self.min_illicit_votes_: int | None = None
        self.calibration_: dict[str, float | int | bool] | None = None

    @staticmethod
    def _features(frame: pd.DataFrame) -> list[str]:
        columns = [column for column in frame.columns if column not in _NON_FEATURE_COLUMNS]
        if not columns:
            raise ValueError("No model feature columns are available.")
        return columns

    def fit(self, train: pd.DataFrame) -> "RuleBasedFraudDetector":
        """Derive feature thresholds from labeled training rows only."""
        if "target" not in train or set(train["target"].unique()) != {0, 1}:
            raise ValueError("Training data must include both binary target classes.")
        rules: list[FeatureRule] = []
        for column in self._features(train):
            values = pd.to_numeric(train[column], errors="raise")
            licit, illicit = values[train["target"] == 0], values[train["target"] == 1]
            difference, scale = float(illicit.mean() - licit.mean()), float(values.std(ddof=0))
            if not np.isfinite(difference) or not np.isfinite(scale) or scale == 0:
                continue
            rules.append(FeatureRule(column, float((illicit.mean() + licit.mean()) / 2), difference > 0, abs(difference) / scale))
        if not rules:
            raise ValueError("No non-constant numeric features are available for rules.")
        rules.sort(key=lambda rule: (-rule.effect_size, rule.feature))
        self.rules_ = tuple(rules[: self.max_rules])
        self.min_illicit_votes_ = None
        self.calibration_ = None
        return self

    def predict_scores(self, data: pd.DataFrame) -> pd.Series:
        """Return each row's fraction of fitted rules voting illicit."""
        if self.rules_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        votes = pd.Series(0, index=data.index, dtype="int64")
        for rule in self.rules_:
            if rule.feature not in data:
                raise ValueError(f"Data is missing fitted feature: {rule.feature}")
            values = pd.to_numeric(data[rule.feature], errors="raise")
            votes += (values >= rule.threshold if rule.illicit_when_greater else values <= rule.threshold).astype("int64")
        return votes / len(self.rules_)

    def calibrate(self, validation: pd.DataFrame, *, minimum_accuracy: float = 0.70, maximum_accuracy: float = 0.80) -> dict[str, float | int | bool]:
        """Choose the vote threshold on validation data and report target attainment.

        In-range candidates are ranked by illicit F1. If none are in range, the
        highest-F1 candidate is retained and ``within_accuracy_target`` is false.
        """
        if self.rules_ is None:
            raise RuntimeError("Detector must be fitted before calibration.")
        if not 0 <= minimum_accuracy <= maximum_accuracy <= 1:
            raise ValueError("Accuracy bounds must satisfy 0 <= min <= max <= 1.")
        if "target" not in validation or validation.empty:
            raise ValueError("Validation data with targets is required for calibration.")
        target, scores = validation["target"].astype(int), self.predict_scores(validation)
        candidates: list[dict[str, float | int | bool]] = []
        for minimum_votes in range(1, len(self.rules_) + 1):
            prediction = (scores * len(self.rules_) >= minimum_votes).astype(int)
            tp = int(((prediction == 1) & (target == 1)).sum())
            fp = int(((prediction == 1) & (target == 0)).sum())
            fn = int(((prediction == 0) & (target == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = float((prediction == target).mean())
            candidates.append({"minimum_illicit_votes": minimum_votes, "accuracy": accuracy, "f1": f1, "within_accuracy_target": minimum_accuracy <= accuracy <= maximum_accuracy})
        target_accuracy = (minimum_accuracy + maximum_accuracy) / 2
        pool = [item for item in candidates if item["within_accuracy_target"]] or candidates
        chosen = max(pool, key=lambda item: (item["f1"], -abs(item["accuracy"] - target_accuracy), -item["minimum_illicit_votes"]))
        self.min_illicit_votes_, self.calibration_ = int(chosen["minimum_illicit_votes"]), chosen
        return chosen

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return illicit (1) / licit (0) predictions using calibrated votes."""
        if self.min_illicit_votes_ is None or self.rules_ is None:
            raise RuntimeError("Detector must be fitted and calibrated before prediction.")
        return (self.predict_scores(data) * len(self.rules_) >= self.min_illicit_votes_).astype("int8")
