"""Model selection on the TRAIN split only.

Steps (the test split is never loaded into any estimator here):
  1. Stratified 80/20 split (test indices are stored and re-checked in evaluate.py).
  2. 5-fold stratified CV comparison of four models with default hyperparameters.
  3. RandomizedSearchCV (ROC-AUC) on the top models, also searching median vs KNN imputation.
  4. Out-of-fold predictions -> decide whether sigmoid calibration improves the Brier score.
  5. Out-of-fold predictions -> pick the decision threshold that reaches recall >= 0.80
     (plus a max-accuracy threshold, reported for comparison only).
  6. Refit the chosen pipeline on the full train split and save it with its threshold.

Run with:  python -m diabetes_predictor.train
"""

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from diabetes_predictor import config
from diabetes_predictor.data import load_data, train_test_split_data
from diabetes_predictor.metrics import (
    compute_metrics,
    select_accuracy_threshold,
    select_threshold,
)
from diabetes_predictor.pipeline import MODELS, build_pipeline, make_imputer
from diabetes_predictor.plots import plot_calibration, plot_threshold_tradeoff

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}

# Search spaces use Pipeline step prefixes ("model__", "imputer").
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logistic_regression": {"model__C": loguniform(1e-3, 1e2)},
    "decision_tree": {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": randint(2, 12),
        "model__min_samples_leaf": randint(1, 40),
        "model__ccp_alpha": uniform(0, 0.02),
    },
    "random_forest": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": [None, 4, 6, 8, 10, 12],
        "model__min_samples_leaf": randint(1, 20),
        "model__max_features": ["sqrt", "log2", 0.5, None],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": randint(50, 400),
        "model__max_leaf_nodes": randint(4, 32),
        "model__min_samples_leaf": randint(5, 60),
        "model__l2_regularization": loguniform(1e-3, 10),
    },
}


def make_cv() -> StratifiedKFold:
    """The one CV splitter used everywhere, so every step sees identical folds."""
    return StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)


def compare_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and std of each CV metric for every model at default settings (threshold 0.5)."""
    results = {}
    for name in MODELS:
        scores = cross_validate(
            build_pipeline(name), X, y, cv=make_cv(), scoring=CV_SCORING, n_jobs=-1
        )
        results[name] = {
            m: {
                "mean": float(scores[f"test_{m}"].mean()),
                "std": float(scores[f"test_{m}"].std()),
            }
            for m in CV_SCORING
        }
        print(f"  {name:<24} ROC-AUC {_fmt(results[name]['roc_auc'])}")
    return results


def tune_model(name: str, X: pd.DataFrame, y: pd.Series) -> RandomizedSearchCV:
    """Randomized hyperparameter search (ROC-AUC) on the train split, including the imputer."""
    space = {"imputer": [make_imputer("median"), make_imputer("knn")], **SEARCH_SPACES[name]}
    search = RandomizedSearchCV(
        build_pipeline(name),
        space,
        n_iter=config.N_ITER_SEARCH,
        scoring="roc_auc",
        cv=make_cv(),
        random_state=config.SEED,
        n_jobs=-1,
    )
    return search.fit(X, y)


def summarize_search(search: RandomizedSearchCV) -> dict[str, Any]:
    """JSON-friendly summary: best params, best CV ROC-AUC, and best score per imputer."""
    res = pd.DataFrame(search.cv_results_)
    res["imputer_kind"] = res["param_imputer"].map(
        lambda imp: "median" if type(imp).__name__ == "SimpleImputer" else "knn"
    )
    best = search.best_index_
    return {
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "roc_auc": {
            "mean": float(res.loc[best, "mean_test_score"]),
            "std": float(res.loc[best, "std_test_score"]),
        },
        "best_roc_auc_by_imputer": res.groupby("imputer_kind")["mean_test_score"].max().to_dict(),
    }


def oof_proba(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold P(diabetic): each row is predicted by a model that never saw it."""
    return cross_val_predict(
        clone(estimator), X, y, cv=make_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]


def calibrated(estimator: BaseEstimator) -> CalibratedClassifierCV:
    """Sigmoid (Platt) calibration; isotonic needs more data than ~600 rows to avoid overfitting.

    ensemble=False: one base pipeline is refit on all data it receives, and the sigmoid is fit on
    its internal CV predictions, so there is a single underlying model to explain with SHAP.
    """
    return CalibratedClassifierCV(clone(estimator), method="sigmoid", cv=make_cv(), ensemble=False)


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print(
        f"Train: {len(X_train)} rows ({y_train.mean():.1%} positive) | "
        f"Test: {len(X_test)} rows held out, not used here"
    )

    print("\n[1/4] 5-fold CV comparison (default hyperparameters, threshold 0.5)")
    comparison = compare_models(X_train, y_train)

    ranked = sorted(comparison, key=lambda n: comparison[n]["roc_auc"]["mean"], reverse=True)
    to_tune = ranked[: config.N_MODELS_TO_TUNE]
    print(f"\n[2/4] RandomizedSearchCV ({config.N_ITER_SEARCH} candidates) on: {to_tune}")
    searches = {name: tune_model(name, X_train, y_train) for name in to_tune}
    tuning = {name: summarize_search(s) for name, s in searches.items()}
    for name, t in tuning.items():
        print(
            f"  {name:<24} ROC-AUC {_fmt(t['roc_auc'])}  by imputer {t['best_roc_auc_by_imputer']}"
        )
    selected = max(tuning, key=lambda n: tuning[n]["roc_auc"]["mean"])
    best_pipeline = searches[selected].best_estimator_
    print(f"  -> selected: {selected}")

    print("\n[3/4] Calibration check on out-of-fold predictions")
    proba_raw = oof_proba(best_pipeline, X_train, y_train)
    proba_cal = oof_proba(calibrated(best_pipeline), X_train, y_train)
    brier_raw = compute_metrics(y_train, proba_raw, 0.5)["brier"]
    brier_cal = compute_metrics(y_train, proba_cal, 0.5)["brier"]
    use_calibration = brier_cal < brier_raw
    print(
        f"  Brier uncalibrated {brier_raw:.4f} | sigmoid-calibrated {brier_cal:.4f} "
        f"-> calibrate: {use_calibration}"
    )
    plot_calibration(
        y_train,
        {"Uncalibrated": proba_raw, "Sigmoid-calibrated": proba_cal},
        config.FIGURES_DIR / "calibration_cv.png",
        "Calibration (out-of-fold CV, train set)",
    )

    print(f"\n[4/4] Threshold for recall >= {config.TARGET_RECALL:.2f} (out-of-fold, train set)")
    final_estimator = calibrated(best_pipeline) if use_calibration else clone(best_pipeline)
    proba = proba_cal if use_calibration else proba_raw
    threshold = select_threshold(y_train, proba, config.TARGET_RECALL)
    at_threshold = compute_metrics(y_train, proba, threshold)
    at_half = compute_metrics(y_train, proba, 0.5)
    print(
        f"  threshold {threshold:.3f}: recall {at_threshold['recall']:.3f}, "
        f"precision {at_threshold['precision']:.3f} "
        f"(at 0.5: recall {at_half['recall']:.3f}, precision {at_half['precision']:.3f})"
    )
    # Secondary, reported-only operating point: maximum accuracy (also chosen out-of-fold).
    accuracy_threshold = select_accuracy_threshold(y_train, proba)
    at_accuracy = compute_metrics(y_train, proba, accuracy_threshold)
    print(
        f"  max-accuracy threshold {accuracy_threshold:.3f}: accuracy {at_accuracy['accuracy']:.3f}"
        f", recall {at_accuracy['recall']:.3f} (screening threshold: accuracy "
        f"{at_threshold['accuracy']:.3f})"
    )
    plot_threshold_tradeoff(
        y_train,
        proba,
        threshold,
        config.TARGET_RECALL,
        config.FIGURES_DIR / "threshold_tradeoff_cv.png",
        accuracy_threshold=accuracy_threshold,
    )

    final_estimator.fit(X_train, y_train)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_estimator,
            "threshold": threshold,
            "accuracy_threshold": accuracy_threshold,
            "model_name": selected,
            "calibrated": use_calibration,
            "features": config.FEATURES,
            "test_index": X_test.index.tolist(),
            "sklearn_version": sklearn.__version__,
        },
        config.MODEL_PATH,
    )

    cv_results = {
        "settings": {
            "seed": config.SEED,
            "n_splits": config.N_SPLITS,
            "test_size": config.TEST_SIZE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "n_iter_search": config.N_ITER_SEARCH,
            "target_recall": config.TARGET_RECALL,
        },
        "model_comparison": comparison,
        "tuning": tuning,
        "selected_model": selected,
        "calibration": {
            "method": "sigmoid",
            "brier_uncalibrated": brier_raw,
            "brier_calibrated": brier_cal,
            "calibrated": use_calibration,
        },
        "threshold": {
            "value": threshold,
            "oof_metrics_at_threshold": at_threshold,
            "oof_metrics_at_0.5": at_half,
        },
        "accuracy_threshold": {
            "value": accuracy_threshold,
            "oof_metrics": at_accuracy,
        },
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CV_RESULTS_PATH.write_text(json.dumps(cv_results, indent=2))
    print(f"\nSaved model -> {config.MODEL_PATH.relative_to(config.ROOT_DIR)}")
    print(f"Saved CV results -> {config.CV_RESULTS_PATH.relative_to(config.ROOT_DIR)}")


def _fmt(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BaseEstimator):
        return "median" if type(value).__name__ == "SimpleImputer" else "knn"
    return value


if __name__ == "__main__":
    main()
