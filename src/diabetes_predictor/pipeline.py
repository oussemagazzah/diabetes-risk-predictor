"""scikit-learn Pipelines: every learned preprocessing step is fit inside CV folds only.

Pipeline layout (identical for every model, so the comparison is fair):

    zeros -> NaN  ->  imputer  ->  scaler  ->  classifier
    (stateless)      (median/KNN)  (only if needed)

Because the imputer and scaler are Pipeline steps, ``cross_validate`` / ``RandomizedSearchCV``
re-fit them on each training fold, so validation folds never influence their statistics.
"""

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from diabetes_predictor import config
from diabetes_predictor.data import replace_invalid_zeros

# class_weight="balanced" re-weights samples by 1 / class frequency (~35% positives), so the
# minority (diabetic) class contributes as much to the loss as the majority class.
MODELS: dict[str, Callable[[], ClassifierMixin]] = {
    "logistic_regression": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
    # max_depth=5: a fully grown tree memorises 600 rows; depth is tuned later anyway.
    "decision_tree": lambda: DecisionTreeClassifier(
        class_weight="balanced", max_depth=5, random_state=config.SEED
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=config.SEED
    ),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=config.SEED
    ),
}

# Distance/gradient-based models need standardized inputs; trees are invariant to scaling.
NEEDS_SCALING = {"logistic_regression"}


def make_imputer(kind: str = "median") -> SimpleImputer | Pipeline:
    """Median imputer (robust to skew/outliers) or KNN imputer.

    KNN distances are scale-dependent (Insulin ranges to ~850, DPF to ~2.4), so the KNN
    variant standardizes first; its output therefore stays standardized.
    """
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn":
        return Pipeline([("scale", StandardScaler()), ("knn", KNNImputer(n_neighbors=5))])
    raise ValueError(f"Unknown imputer {kind!r}; use 'median' or 'knn'")


def build_pipeline(model_name: str, imputer: str = "median") -> Pipeline:
    """Build the full preprocessing + classifier Pipeline for ``model_name``."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {list(MODELS)}")
    scaler = StandardScaler() if model_name in NEEDS_SCALING else "passthrough"
    return Pipeline(
        [
            ("zeros", FunctionTransformer(replace_invalid_zeros, feature_names_out="one-to-one")),
            ("imputer", make_imputer(imputer)),
            ("scaler", scaler),
            ("model", MODELS[model_name]()),
        ]
    )
