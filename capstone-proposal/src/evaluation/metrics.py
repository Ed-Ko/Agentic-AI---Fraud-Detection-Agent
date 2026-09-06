"""Binary fraud-classification metrics used by every baseline."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(y_true: Sequence[int], y_scores: Sequence[float], *, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    """Calculate primary and secondary metrics with illicit as positive class."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one.")
    truth, scores = np.asarray(y_true, dtype=int), np.asarray(y_scores, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or len(truth) != len(scores) or not len(truth):
        raise ValueError("Inputs must be non-empty, one-dimensional, and equally sized.")
    if not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("y_true must use 0=licit and 1=illicit.")
    prediction = (scores >= threshold).astype(int)
    has_both_classes = len(np.unique(truth)) == 2
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, scores)) if has_both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(truth, scores)) if has_both_classes else float("nan"),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
    }
