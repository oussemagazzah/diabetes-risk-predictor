"""Matplotlib figures written to reports/figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts and CI never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, proba, auc: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title=title)
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr(y_true, proba, ap: float, threshold: float, path: Path, title: str) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, lw=2, label=f"Model (AP = {ap:.3f})")
    ax.axhline(np.mean(y_true), ls="--", color="grey", label="Prevalence (no skill)")
    i = np.searchsorted(thresholds, threshold)  # first point with score >= threshold
    ax.scatter(recall[i], precision[i], color="red", zorder=3, label=f"Threshold {threshold:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1.01), ylim=(0, 1.01))
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_confusion(y_true, pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        pred,
        display_labels=["Non-diabetic", "Diabetic"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    _save(fig, path)


def plot_calibration(y_true, probas: dict[str, np.ndarray], path: Path, title: str) -> None:
    """Reliability diagram: predicted probability vs observed positive rate, per curve."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    for label, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction diabetic", title=title)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_threshold_tradeoff(y_true, proba, threshold: float, target: float, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.axhline(target, ls=":", color="grey", label=f"Recall target {target:.2f}")
    ax.axvline(threshold, ls="--", color="red", label=f"Chosen threshold {threshold:.2f}")
    ax.set(
        xlabel="Decision threshold",
        ylabel="Score",
        title="Threshold trade-off (out-of-fold CV, train set)",
    )
    ax.legend(loc="lower left")
    _save(fig, path)
