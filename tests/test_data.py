import numpy as np
import pandas as pd
import pytest

from diabetes_predictor import config
from diabetes_predictor.data import (
    load_data,
    replace_invalid_zeros,
    split_features_target,
    train_test_split_data,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def test_expected_columns_and_shape(df):
    assert list(df.columns) == [*config.FEATURES, config.TARGET]
    assert df.shape == (768, 9)
    assert set(df[config.TARGET].unique()) == {0, 1}


def test_raw_data_has_no_nan_or_duplicates(df):
    # Documents why the old dropna()/drop_duplicates() calls were removed: they were no-ops.
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_zeros_become_nan_only_in_listed_columns():
    X = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=config.FEATURES)
    out = replace_invalid_zeros(X)
    for col in config.ZERO_AS_MISSING:
        assert np.isnan(out.loc[0, col])
    assert out.loc[0, "Pregnancies"] == 0  # zero pregnancies is a real value
    assert X.loc[0, "Glucose"] == 0  # input is not mutated


def test_replace_zeros_accepts_numpy():
    arr = np.array([[1, 0, 70, 20, 0, 30.0, 0.5, 40]])
    out = replace_invalid_zeros(arr)
    assert np.isnan(out.loc[0, "Glucose"]) and np.isnan(out.loc[0, "Insulin"])
    assert out.loc[0, "BloodPressure"] == 70


def test_no_target_leakage_in_features(df):
    X, y = split_features_target(df)
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.FEATURES
    assert y.name == config.TARGET


def test_split_is_stratified_disjoint_and_reproducible(df):
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    assert len(X_test) == round(len(df) * config.TEST_SIZE)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    X_test_again = train_test_split_data(df)[1]
    assert list(X_test.index) == list(X_test_again.index)
