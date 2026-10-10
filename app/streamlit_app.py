"""Streamlit demo: enter a patient's measurements -> risk probability + SHAP explanation.

Run with:  streamlit run app/streamlit_app.py   (after `make train`)
"""

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from diabetes_predictor import config
from diabetes_predictor.data import load_data, train_test_split_data
from diabetes_predictor.explain import base_pipeline, explain
from diabetes_predictor.predict import load_bundle, predict_patient

# column -> (label, min, max, default, step)
FIELDS = {
    "Pregnancies": ("Pregnancies", 0, 20, 2, 1),
    "Glucose": ("Glucose, 2h OGTT (mg/dL)", 0, 250, 120, 1),
    "BloodPressure": ("Diastolic blood pressure (mm Hg)", 0, 140, 72, 1),
    "SkinThickness": ("Triceps skin fold (mm)", 0, 100, 0, 1),
    "Insulin": ("2h serum insulin (µU/mL)", 0, 900, 0, 1),
    "BMI": ("BMI (kg/m²)", 0.0, 70.0, 32.0, 0.1),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.0, 2.5, 0.37, 0.01),
    "Age": ("Age (years)", 21, 90, 33, 1),
}


@st.cache_resource
def load_resources():
    bundle = load_bundle()
    X_train = train_test_split_data(load_data())[0]
    return bundle, base_pipeline(bundle["model"]), X_train


st.set_page_config(page_title="Diabetes risk screening demo", page_icon="🩺")
st.title("Diabetes risk screening — demo")
st.warning(
    "Educational project, **not a medical device**. Trained on 768 Pima Indian women aged 21+; "
    "predictions may not apply to other populations. Do not use for medical decisions."
)

try:
    bundle, pipeline, X_train = load_resources()
except FileNotFoundError:
    st.error("No trained model found. Run `make train` first.")
    st.stop()

with st.form("patient"):
    st.caption("For glucose, blood pressure, skin fold, insulin and BMI, enter 0 if not measured.")
    cols = st.columns(2)
    patient = {}
    for i, (col, (label, lo, hi, default, step)) in enumerate(FIELDS.items()):
        patient[col] = cols[i % 2].number_input(
            label, min_value=lo, max_value=hi, value=default, step=step
        )
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    result = predict_patient({k: float(v) for k, v in patient.items()}, bundle)
    p, t = result["probability"], result["threshold"]
    st.metric("Estimated probability of diabetes", f"{p:.0%}")
    st.progress(min(p, 1.0))
    if result["high_risk"]:
        st.error(f"Above the screening threshold ({t:.2f}): would be referred for a blood test.")
    else:
        st.success(f"Below the screening threshold ({t:.2f}).")
    st.caption(
        f"The threshold {t:.2f} was chosen on cross-validation to catch ≥ 80% of diabetic "
        "patients, accepting more false alarms in exchange."
    )

    st.subheader("Why this prediction?")
    exp = explain(pipeline, X_train, pd.DataFrame([patient], columns=config.FEATURES))[0]
    shap.plots.waterfall(exp, show=False, max_display=len(config.FEATURES))
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption(
        "Red bars push the risk up, blue bars push it down (log-odds units of the underlying "
        "model). 'nan' means not measured; the pipeline imputed it from the training data."
    )
