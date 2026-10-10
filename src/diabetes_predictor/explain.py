"""SHAP explanations: a global summary plot and one single-patient waterfall.

If the saved model is sigmoid-calibrated, SHAP explains the underlying pipeline (one model,
thanks to ensemble=False). Calibration is a monotone map, so it changes probabilities but not
which features push a patient's risk up or down. SHAP values are in the model's raw output
units (log-odds for logistic regression / boosting).

Run with:  python -m diabetes_predictor.explain
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from diabetes_predictor import config  # noqa: E402
from diabetes_predictor.data import (  # noqa: E402
    load_data,
    replace_invalid_zeros,
    train_test_split_data,
)
from diabetes_predictor.predict import load_bundle  # noqa: E402


def base_pipeline(model) -> Pipeline:
    """Return the preprocessing + classifier Pipeline, unwrapping calibration if present."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def explain(pipeline: Pipeline, X_background: pd.DataFrame, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values for ``X`` (positive class), computed on the model's preprocessed inputs.

    The returned Explanation shows the original-unit values (NaN = not measured) for display.
    """
    preprocess, estimator = pipeline[:-1], pipeline[-1]
    background = preprocess.transform(X_background)
    # Use the whole background set (no random subsampling) so explanations are reproducible.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(estimator, masker)
    exp = explainer(preprocess.transform(X))
    if exp.values.ndim == 3:  # tree models with one output per class
        exp = exp[..., 1]
    return shap.Explanation(
        values=exp.values,
        base_values=exp.base_values,
        data=replace_invalid_zeros(X).to_numpy(),
        feature_names=config.FEATURES,
    )


def save_waterfall(exp: shap.Explanation, title: str, path) -> None:
    """Save a single-patient waterfall plot."""
    shap.plots.waterfall(exp, show=False, max_display=len(config.FEATURES))
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bundle = load_bundle()
    pipeline = base_pipeline(bundle["model"])
    X_train, X_test, _, y_test = train_test_split_data(load_data())
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Global view: what the model learned, computed on the training patients.
    exp_train = explain(pipeline, X_train, X_train)
    np.random.seed(config.SEED)  # beeswarm jitters dots with the global RNG; keep figures stable
    shap.plots.beeswarm(exp_train, show=False, max_display=len(config.FEATURES))
    fig = plt.gcf()
    fig.suptitle("SHAP summary (train set, log-odds contribution)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Local view: the first diabetic patient in the held-out set (a "new" patient).
    idx = y_test[y_test == 1].index[0]
    patient = X_test.loc[[idx]]
    proba = float(bundle["model"].predict_proba(patient)[0, 1])
    exp_patient = explain(pipeline, X_train, patient)[0]
    save_waterfall(
        exp_patient,
        f"Patient #{idx}: P(diabetic) = {proba:.2f} (threshold {bundle['threshold']:.2f})",
        config.FIGURES_DIR / "shap_patient_example.png",
    )

    importance = pd.Series(abs(exp_train.values).mean(axis=0), index=config.FEATURES)
    print("Mean |SHAP| (train set, log-odds):")
    print(importance.sort_values(ascending=False).round(3).to_string())
    print(f"\nExample patient #{idx}: P(diabetic) = {proba:.3f}")
    print(f"Saved -> {config.FIGURES_DIR.relative_to(config.ROOT_DIR)}/shap_*.png")


if __name__ == "__main__":
    main()
