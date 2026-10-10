"""Data loading, cleaning of invalid zeros, and the stratified train/test split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_predictor import config


def load_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and check that it has exactly the expected columns."""
    df = pd.read_csv(path)
    expected = [*config.FEATURES, config.TARGET]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns {list(df.columns)}; expected {expected}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target column."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def replace_invalid_zeros(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return a copy of ``X`` where impossible zeros (e.g. BMI = 0) are replaced by NaN.

    This is stateless (it learns nothing from the data), so it cannot leak information.
    It still lives inside the model Pipeline so the saved model accepts raw inputs.
    """
    X = pd.DataFrame(X, columns=config.FEATURES) if isinstance(X, np.ndarray) else X.copy()
    cols = [c for c in config.ZERO_AS_MISSING if c in X.columns]
    X[cols] = X[cols].replace(0, np.nan)
    return X


def train_test_split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split with the global seed. This is the FIRST thing done to the data."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
