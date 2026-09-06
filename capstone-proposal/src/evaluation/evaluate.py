"""Leakage-safe evaluation entry points for fitted fraud detectors."""

from __future__ import annotations

import pandas as pd

from .metrics import calculate_metrics


def evaluate_model(model: object, partition: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    """Evaluate a fitted detector without fitting it on this partition."""
    if "target" not in partition or partition.empty:
        raise ValueError("Evaluation requires a non-empty partition with a target column.")
    scorer = getattr(model, "predict_scores", None)
    if not callable(scorer):
        raise TypeError("model must provide a callable predict_scores method.")
    return calculate_metrics(partition["target"], scorer(partition), threshold=threshold)
