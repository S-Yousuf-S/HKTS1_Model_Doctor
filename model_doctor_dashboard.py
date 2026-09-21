"""
================================================================================
 MODEL_DOCTOR_DASHBOARD.PY

 Project      : Model Doctor — Automated ML Audit Toolkit
 Author       : Yousuf S. R. Sakkaf
 Description  : Streamlit dashboard for the Model Doctor auditor. Two modes:
                (1) train a model on an uploaded CSV and audit it in one step,
                (2) upload an existing .pkl/.joblib model plus train/test CSVs
                and audit it as-is.
 Run          : streamlit run model_doctor_dashboard.py
================================================================================
"""

import os
import tempfile
import base64
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from model_doctor import audit_from_files, generate_html_report, run_audit

PRIMARY_COLOR = "darkcyan"
SECONDARY_COLOR = "mintcream"
ACCENT_COLOR = "goldenrod"
CRITICAL_COLOR = "#C0392B"
WARNING_COLOR = "#B9770E"
SUCCESS_COLOR = "#2E8B57"
TITLE_COLOR = "darkslategray"

st.set_page_config(page_title="Model Doctor", page_icon="Assets/model_doctor_logo.png", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { font-size: 18px; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
label { font-size: 1.05rem !important; }
h1 { font-size: 2.3rem !important; }
[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------


def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = get_base64_image("Assets/model_doctor_logo.png")

st.markdown(f"""
<div style="background:{PRIMARY_COLOR};color:white;padding:20px;border-radius:8px;text-align:center;margin-bottom:20px;
            display:flex;align-items:center;justify-content:center;gap:16px;">
    <img src="data:image/png;base64,{logo_b64}" width="56" height="56">
    <div>
        <h1 style="margin:0;">Model Doctor</h1>
        <p style="margin:4px 0 0 0;font-style:italic;">Automated ML Audit Toolkit — leakage, contamination, overfitting, imbalance, misleading metrics, and data quality</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="background:{SECONDARY_COLOR};border-left:6px solid {PRIMARY_COLOR};padding:12px;border-radius:6px;">
    <b>About this tool</b><br><br>
    Model Doctor audits a fitted scikit-learn-style pipeline against the
    train/test data it was fit on, and flags common ML pipeline failures —
    without being hardcoded to any one model type or dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F1F3F6;border-left:6px solid {ACCENT_COLOR};padding:12px;border-radius:6px;margin-top:12px;">
    <b>What it checks</b><br><br>
    Data leakage &middot; Train/test contamination<br>
    Misleading metrics &middot; Overfitting signals<br>
    Data quality issues &middot; Class imbalance
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Reusable KPI card component (same pattern as salesfc_dashboard.py)
# ------------------------------------------------------------------------------
def kpi_card(label, value, color=SUCCESS_COLOR):
    st.markdown(f"""
    <div style="background:#EAF7EA;border-left:6px solid {color};padding:14px 18px;border-radius:8px;text-align:center;">
        <div style="color:#555;font-size:14px;font-weight:600;">{label}</div>
        <div style="color:{color};font-size:28px;font-weight:bold;margin-top:4px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def show_kpi_row(findings):
    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_warn = sum(1 for f in findings if f.severity == "warning")
    n_info = sum(1 for f in findings if f.severity == "info")
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Critical", n_crit, CRITICAL_COLOR if n_crit else SUCCESS_COLOR)
    with k2:
        kpi_card("Warning", n_warn, WARNING_COLOR if n_warn else SUCCESS_COLOR)
    with k3:
        kpi_card("Info", n_info, "#1F77B4" if n_info else SUCCESS_COLOR)


# ------------------------------------------------------------------------------
# Two modes, as tabs (matches the tab pattern in salesfc_dashboard.py)
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["\U0001F9EA Train & Audit a New Model", "\U0001F4C2 Audit an Existing Model"])

with tab1:
    st.markdown("Upload a CSV, pick a target column and model type, and Model Doctor will train it and audit the resulting pipeline in one step.")
    uploaded = st.file_uploader("Upload a CSV", type="csv", key="train_csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Target column", df.columns)
        with c2:
            task = st.radio("Task type", ["classification", "regression"], horizontal=True)
        with c3:
            model_choice = st.selectbox(
                "Model type",
                ["LogisticRegression", "RandomForest", "XGBoost"]
                if task == "classification"
                else ["RandomForest", "XGBoost"],
            )

        if st.button("\U0001F50D Run Audit", type="primary"):
            X = pd.get_dummies(df.drop(columns=[target]))  # simple categorical handling for this demo
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if task == "classification":
                model = {
                    "LogisticRegression": LogisticRegression(max_iter=1000),
                    "RandomForest": RandomForestClassifier(random_state=42),
                    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
                }[model_choice]
            else:
                model = {
                    "RandomForest": RandomForestRegressor(random_state=42),
                    "XGBoost": XGBRegressor(random_state=42),
                }[model_choice]

            with st.spinner("Training model and running audit..."):
                model.fit(X_train, y_train)
                findings = run_audit(model, X_train, X_test, y_train, y_test)
                report_html = generate_html_report(findings, model_name=f"{model_choice} on {uploaded.name}")

            show_kpi_row(findings)
            st.components.v1.html(report_html, height=800, scrolling=True)
            st.download_button("\U0001F4E5 Download report (HTML)", report_html, "audit_report.html")

with tab2:
    st.markdown("Upload an already-trained `.pkl`/`.joblib` model plus the train and test CSVs (each including the target column) it was evaluated on.")

    model_file = st.file_uploader("Model file (.pkl or .joblib)", type=["pkl", "joblib"], key="model_file")
    col_a, col_b = st.columns(2)
    with col_a:
        train_file = st.file_uploader("Training CSV", type="csv", key="train_file")
    with col_b:
        test_file = st.file_uploader("Test CSV", type="csv", key="test_file")
    target = st.text_input("Target column name")

    if st.button("\U0001F50D Run Audit", type="primary", key="existing_model_btn"):
        if not (model_file and train_file and test_file and target):
            st.warning("Please provide the model, both CSVs, and the target column name.")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                model_path = os.path.join(tmp, model_file.name)
                train_path = os.path.join(tmp, "train.csv")
                test_path = os.path.join(tmp, "test.csv")
                with open(model_path, "wb") as f:
                    f.write(model_file.read())
                with open(train_path, "wb") as f:
                    f.write(train_file.read())
                with open(test_path, "wb") as f:
                    f.write(test_file.read())

                with st.spinner("Running audit..."):
                    findings, X_train, X_test, y_train, y_test = audit_from_files(
                        model_path, train_path, test_path, target
                    )
                    report_html = generate_html_report(findings, model_name=model_file.name)

            show_kpi_row(findings)
            st.components.v1.html(report_html, height=800, scrolling=True)
            st.download_button("\U0001F4E5 Download report (HTML)", report_html, "audit_report.html")
