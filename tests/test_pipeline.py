import numpy as np
import pandas as pd
import pytest

from diabetes_predictor.data import load_data, split_features_target
from diabetes_predictor.pipeline import MODELS, build_pipeline


@pytest.fixture(scope="module")
def tiny_sample() -> tuple[pd.DataFrame, pd.Series]:
    X, y = split_features_target(load_data())
    idx = y.groupby(y).sample(n=30, random_state=0).index  # 60 rows, both classes
    return X.loc[idx], y.loc[idx]


@pytest.mark.parametrize("model_name", list(MODELS))
@pytest.mark.parametrize("imputer", ["median", "knn"])
def test_pipeline_fits_and_outputs_valid_probabilities(tiny_sample, model_name, imputer):
    X, y = tiny_sample
    pipe = build_pipeline(model_name, imputer=imputer).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(pipe.predict(X)) <= {0, 1}


def test_pipeline_handles_raw_zeros_at_predict_time(tiny_sample):
    X, y = tiny_sample
    pipe = build_pipeline("logistic_regression").fit(X, y)
    patient = X.iloc[[0]].copy()
    patient[["Insulin", "SkinThickness"]] = 0  # "not measured", must be imputed, not used as 0
    assert np.isfinite(pipe.predict_proba(patient)).all()


def test_imputer_statistics_come_from_training_data_only(tiny_sample):
    X, y = tiny_sample
    pipe = build_pipeline("decision_tree").fit(X, y)
    learned_median = pipe.named_steps["imputer"].statistics_
    expected = X.replace(
        {c: {0: np.nan} for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]}
    ).median()
    np.testing.assert_allclose(learned_median, expected.to_numpy())


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_pipeline("svm")
