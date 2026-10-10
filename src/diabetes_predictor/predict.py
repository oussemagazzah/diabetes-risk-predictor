"""Predict diabetes risk for a single patient with the saved model.

Example:
    python -m diabetes_predictor.predict --pregnancies 2 --glucose 148 --bmi 33.6 \
        --dpf 0.627 --age 50 --blood-pressure 72
Measurements left out (or given as 0) are treated as "not measured" and imputed.
"""

import argparse
from typing import Any

import joblib
import pandas as pd

from diabetes_predictor import config

# CLI flag -> dataset column
CLI_FLAGS = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "dpf": "DiabetesPedigreeFunction",
    "age": "Age",
}


def load_bundle() -> dict[str, Any]:
    """Load the model bundle (pipeline + threshold) written by train.py."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `make train` first.")
    return joblib.load(config.MODEL_PATH)


def predict_patient(patient: dict[str, float], bundle: dict[str, Any]) -> dict[str, Any]:
    """Return P(diabetic) and the screening decision at the model's chosen threshold."""
    missing = set(config.FEATURES) - set(patient)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    X = pd.DataFrame([patient], columns=config.FEATURES)
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    return {
        "probability": probability,
        "threshold": bundle["threshold"],
        "high_risk": probability >= bundle["threshold"],
        "model": bundle["model_name"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag, column in CLI_FLAGS.items():
        optional = column in config.ZERO_AS_MISSING
        parser.add_argument(
            f"--{flag.replace('_', '-')}",
            dest=flag,
            type=float,
            required=not optional,
            default=0.0 if optional else None,
            help=f"{column}" + (" (omit or 0 = not measured)" if optional else ""),
        )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patient = {column: getattr(args, flag) for flag, column in CLI_FLAGS.items()}
    result = predict_patient(patient, load_bundle())
    decision = "HIGH RISK - refer for testing" if result["high_risk"] else "below threshold"
    print(f"P(diabetic) = {result['probability']:.3f}  (threshold {result['threshold']:.3f})")
    print(f"Screening decision: {decision}")


if __name__ == "__main__":
    main()
