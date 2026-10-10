"""Metric helpers shared by training (CV predictions) and evaluation (test set)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Threshold-free (ROC-AUC, PR-AUC, Brier) and thresholded metrics for one prediction set."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall is still >= ``target_recall``.

    Recall only falls as the threshold rises, so the highest qualifying threshold gives the
    fewest false positives (best precision) while still meeting the recall requirement.
    """
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall  # recall[i] is the recall at thresholds[i]
    return float(thresholds[ok].max()) if ok.any() else float(thresholds.min())


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    keys: tuple[str, ...] = ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"),
) -> dict[str, list[float]]:
    """95% percentile bootstrap intervals; the test set is small, so point estimates are noisy."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()
    }
