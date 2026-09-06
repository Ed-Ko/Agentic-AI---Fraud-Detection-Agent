"""Deterministic, train-only preprocessing for temporal fraud experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_TO_TARGET = {"1": 1, "2": 0}


@dataclass(frozen=True)
class TemporalSplits:
    """Chronologically disjoint labeled transaction partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    transactions: pd.DataFrame,
    train_steps: Iterable[int] = range(1, 35),
    validation_steps: Iterable[int] = range(35, 40),
    test_steps: Iterable[int] = range(40, 50),
) -> TemporalSplits:
    """Create the fixed forward-chaining split, excluding unknown labels.

    No shuffling is performed.  Only labels ``1`` (illicit) and ``2`` (licit)
    are supervised targets; their integer target is added as ``target``.
    """
    required = {"time_step", "class"}
    if missing := required - set(transactions.columns):
        raise ValueError(f"Transactions missing required columns: {sorted(missing)}")
    ranges = [set(train_steps), set(validation_steps), set(test_steps)]
    if any(not step_set for step_set in ranges):
        raise ValueError("Each temporal partition must contain at least one time step.")
    if ranges[0] & ranges[1] or ranges[0] & ranges[2] or ranges[1] & ranges[2]:
        raise ValueError("Temporal partition time steps must not overlap.")
    if max(ranges[0]) >= min(ranges[1]) or max(ranges[1]) >= min(ranges[2]):
        raise ValueError("Temporal partitions must be strictly forward in time.")

    labeled = transactions.loc[transactions["class"].isin(LABEL_TO_TARGET)].copy()
    labeled["target"] = labeled["class"].map(LABEL_TO_TARGET).astype("int8")

    def select(steps: set[int]) -> pd.DataFrame:
        return labeled.loc[labeled["time_step"].isin(steps)].copy()

    return TemporalSplits(select(ranges[0]), select(ranges[1]), select(ranges[2]))


class Standardizer:
    """Column-wise standardization fitted solely on training data."""

    def __init__(self, *, include_time_step: bool = False) -> None:
        self.include_time_step = include_time_step
        self.feature_columns_: list[str] | None = None
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "Standardizer":
        """Learn numeric feature statistics from the training partition only."""
        excluded = {"txId", "class", "target"}
        if not self.include_time_step:
            excluded.add("time_step")
        columns = [column for column in train.columns if column not in excluded]
        if not columns:
            raise ValueError("No feature columns are available for preprocessing.")
        values = train.loc[:, columns].apply(pd.to_numeric, errors="raise")
        if values.isna().any().any():
            raise ValueError("Missing feature values must be handled before standardization.")
        self.feature_columns_ = columns
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        self.scale_ = scale.mask(scale == 0, 1.0)
        return self

    def transform(self, partition: pd.DataFrame) -> pd.DataFrame:
        """Transform a partition using already-fitted training statistics."""
        if self.feature_columns_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted on training data before transform.")
        missing = set(self.feature_columns_) - set(partition.columns)
        if missing:
            raise ValueError(f"Partition missing fitted feature columns: {sorted(missing)}")
        values = partition.loc[:, self.feature_columns_].apply(pd.to_numeric, errors="raise")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data then transform only that same training partition."""
        return self.fit(train).transform(train)
