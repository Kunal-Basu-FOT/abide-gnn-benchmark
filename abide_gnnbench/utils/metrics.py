# abide_gnnbench/utils/metrics.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, matthews_corrcoef, cohen_kappa_score,
)

METRIC_KEYS = [
    "Accuracy", "Sensitivity", "Specificity", "Precision",
    "Recall", "F1", "Balanced_Acc", "AUROC", "AUPRC",
    "Brier", "MCC", "Cohen_Kappa",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    # FIX: force 1-D arrays. A 0-d / scalar-shaped input here causes
    # confusion_matrix(...).ravel() to return fewer than 4 elements,
    # which crashes the `tn, fp, fn, tp = ...` unpack with errors like
    # "only 0-dimensional arrays can be converted to Python scalars" or
    # "only length-1 arrays can be converted to Python scalars"
    # (exact wording depends on sklearn/numpy version — same root cause).
    y_true = np.atleast_1d(np.asarray(y_true)).ravel()
    y_pred = np.atleast_1d(np.asarray(y_pred)).ravel()
    y_prob = np.atleast_1d(np.asarray(y_prob)).ravel()

    # FIX: pin labels=[0, 1] so confusion_matrix always returns a full 2x2
    # matrix (4 elements), even when a fold/bootstrap resample happens to
    # contain only one class. Without this, ravel() can return a 1-element
    # array and the unpack below fails.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm
    specificity = tn / (tn + fp + 1e-9)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen_Kappa": cohen_kappa_score(y_true, y_pred),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float, float]]:
    rng = np.random.RandomState(seed)
    # FIX: same 1-D guard, since bootstrap_ci is the other entry point
    # that raw arrays flow into.
    y_true = np.atleast_1d(np.asarray(y_true)).ravel()
    y_prob = np.atleast_1d(np.asarray(y_prob)).ravel()

    n = len(y_true)
    y_pred = (y_prob >= 0.5).astype(int)
    results: Dict[str, list] = {k: [] for k in METRIC_KEYS}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, ypr)
        for k, v in m.items():
            results[k].append(v)

    ci: Dict[str, Tuple[float, float, float]] = {}
    for k, vals in results.items():
        if not vals:
            ci[k] = (float("nan"), float("nan"), float("nan"))
            continue
        arr = np.array(vals)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        ci[k] = (float(np.mean(arr)), lo, hi)
    return ci