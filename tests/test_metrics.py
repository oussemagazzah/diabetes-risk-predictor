import numpy as np

from diabetes_predictor.metrics import bootstrap_ci, compute_metrics, select_threshold


def test_select_threshold_meets_recall_target_with_highest_threshold():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.35, 0.6, 0.3, 0.5, 0.7, 0.8, 0.9])
    t = select_threshold(y, proba, target_recall=0.8)
    assert compute_metrics(y, proba, t)["recall"] >= 0.8
    assert t == 0.5  # 4/5 positives caught; any higher threshold drops recall to 0.6


def test_compute_metrics_confusion_matrix_consistent():
    y = np.array([0, 1, 1, 0])
    m = compute_metrics(y, np.array([0.2, 0.9, 0.4, 0.7]), threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["recall"] == m["precision"] == m["accuracy"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    point = compute_metrics(y, proba, 0.5)["roc_auc"]
    lo, hi = bootstrap_ci(y, proba, 0.5, n_boot=200, seed=0)["roc_auc"]
    assert lo <= point <= hi
