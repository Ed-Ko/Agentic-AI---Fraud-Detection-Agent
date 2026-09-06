"""Reproducible XGBoost baseline for the temporally split Elliptic data."""

from __future__ import annotations

import pandas as pd

_NON_FEATURE_COLUMNS = frozenset({"txId", "time_step", "class", "target"})


class XGBoostFraudDetector:
    """A fixed-seed, class-weighted XGBoost binary classifier."""

    def __init__(self, *, random_state: int = 42, **overrides: object) -> None:
        self.random_state, self.overrides = random_state, overrides
        self.feature_columns_: list[str] | None = None
        self.model_ = None

    @staticmethod
    def _feature_columns(frame: pd.DataFrame) -> list[str]:
        columns = [column for column in frame.columns if column not in _NON_FEATURE_COLUMNS]
        if not columns:
            raise ValueError("No model feature columns are available.")
        return columns

    def fit(self, train: pd.DataFrame) -> "XGBoostFraudDetector":
        """Fit only on training rows; validation and test rows are not accepted."""
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("Install xgboost (see requirements.txt) to train this baseline.") from error
        if "target" not in train or set(train["target"].unique()) != {0, 1}:
            raise ValueError("Training data must include both binary target classes.")
        self.feature_columns_ = self._feature_columns(train)
        features = train.loc[:, self.feature_columns_].apply(pd.to_numeric, errors="raise")
        if features.isna().any().any():
            raise ValueError("Missing values must be handled before fitting XGBoost.")
        targets = train["target"].astype(int)
        positives, negatives = int(targets.sum()), int(len(targets) - targets.sum())
        parameters: dict[str, object] = {"objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": negatives / positives, "random_state": self.random_state, "n_jobs": 1, "tree_method": "hist"}
        parameters.update(self.overrides)
        self.model_ = XGBClassifier(**parameters)
        self.model_.fit(features, targets)
        return self

    def predict_scores(self, data: pd.DataFrame) -> pd.Series:
        """Return deterministic illicit-class probabilities."""
        if self.model_ is None or self.feature_columns_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        missing = set(self.feature_columns_) - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing fitted feature columns: {sorted(missing)}")
        features = data.loc[:, self.feature_columns_].apply(pd.to_numeric, errors="raise")
        return pd.Series(self.model_.predict_proba(features)[:, 1], index=data.index, name="illicit_score")

    def predict(self, data: pd.DataFrame, *, threshold: float = 0.5) -> pd.Series:
        """Classify with a threshold chosen from validation data."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one.")
        return (self.predict_scores(data) >= threshold).astype("int8")
