"""Final, one-time evaluation of the saved model on the held-out test split.

Nothing here feeds back into model choice: the threshold, hyperparameters and calibration
were all fixed in train.py. Writes reports/metrics.json and the test-set figures.

Run with:  python -m diabetes_predictor.evaluate
"""

import json

import joblib

from diabetes_predictor import config
from diabetes_predictor.data import load_data, train_test_split_data
from diabetes_predictor.metrics import bootstrap_ci, compute_metrics
from diabetes_predictor.plots import plot_calibration, plot_confusion, plot_pr, plot_roc


def load_bundle() -> dict:
    """Load the model bundle saved by train.py."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `make train` first.")
    return joblib.load(config.MODEL_PATH)


def main() -> None:
    bundle = load_bundle()
    _, X_test, _, y_test = train_test_split_data(load_data())
    # Guard: the test rows must be exactly the ones train.py held out.
    if X_test.index.tolist() != bundle["test_index"]:
        raise RuntimeError("Test split differs from the one used at training time.")

    threshold = bundle["threshold"]
    proba = bundle["model"].predict_proba(X_test)[:, 1]
    at_threshold = compute_metrics(y_test, proba, threshold)
    at_half = compute_metrics(y_test, proba, 0.5)
    ci = bootstrap_ci(y_test, proba, threshold, config.N_BOOTSTRAP, config.SEED)

    figs = config.FIGURES_DIR
    name = bundle["model_name"].replace("_", " ")
    plot_roc(y_test, proba, at_threshold["roc_auc"], figs / "roc_test.png", f"ROC – {name} (test)")
    plot_pr(
        y_test,
        proba,
        at_threshold["pr_auc"],
        threshold,
        figs / "pr_test.png",
        f"Precision-Recall – {name} (test)",
    )
    plot_confusion(
        y_test,
        (proba >= threshold).astype(int),
        figs / "confusion_matrix_test.png",
        f"Test set, threshold {threshold:.2f}",
    )
    plot_calibration(y_test, {name: proba}, figs / "calibration_test.png", "Calibration (test)")

    cv_results = json.loads(config.CV_RESULTS_PATH.read_text())
    metrics = {
        **cv_results,
        "test": {
            "n": len(y_test),
            "positive_rate": float(y_test.mean()),
            "at_threshold": at_threshold,
            "at_0.5": at_half,
            "bootstrap_95ci_at_threshold": ci,
        },
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Model: {bundle['model_name']} (calibrated={bundle['calibrated']})")
    print(f"Test set: {len(y_test)} patients, {y_test.mean():.1%} diabetic\n")
    print(
        f"{'metric':<12}{'@ ' + format(threshold, '.3f'):>10}{'@ 0.5':>10}   95% CI (@ threshold)"
    )
    for key in ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]:
        lo, hi = ci[key]
        print(f"{key:<12}{at_threshold[key]:>10.3f}{at_half[key]:>10.3f}   [{lo:.3f}, {hi:.3f}]")
    print(f"{'specificity':<12}{at_threshold['specificity']:>10.3f}{at_half['specificity']:>10.3f}")
    print(f"{'brier':<12}{at_threshold['brier']:>10.3f}")
    print(f"\nConfusion matrix @ threshold: {at_threshold['confusion_matrix']}")
    print(f"Saved -> {config.METRICS_PATH.relative_to(config.ROOT_DIR)} and {figs.name}/*.png")


if __name__ == "__main__":
    main()
