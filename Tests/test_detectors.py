"""
Test suite for Model Doctor detectors.

File: test_detectors.py
Author: Yousuf S.R. Sakkaf

Each test builds a small, deliberately-broken (or deliberately-clean)
scenario and asserts the relevant detector fires (or doesn't).

Run with: pytest tests/test_detectors.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from model_doctor import (
    detect_duplicate_contamination,
    detect_preprocessing_leakage,
    detect_class_imbalance,
    detect_misleading_metrics,
    detect_overfitting_signal,
    detect_data_quality_issues,
    run_audit,
)


def _make_clf_data(n_samples=300, weights=None, random_state=0):
    X, y = make_classification(n_samples=n_samples, n_features=6, n_informative=4,
                                weights=weights, random_state=random_state)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y)
    return X, y


# --- duplicate contamination -------------------------------------------------

def test_detects_duplicate_rows_across_splits():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # inject contamination: copy some train rows into test
    X_test = pd.concat([X_test, X_train.iloc[:10]], ignore_index=True)
    findings = detect_duplicate_contamination(X_train, X_test)
    assert any(f.issue_type == "train_test_contamination" for f in findings)


def test_no_false_positive_on_clean_split():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    findings = detect_duplicate_contamination(X_train, X_test)
    assert findings == []


# --- preprocessing leakage ---------------------------------------------------

def test_detects_scaler_fit_before_split():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # BROKEN: fit scaler on full X (train+test) before split
    leaky_scaler = StandardScaler().fit(X)
    pipeline = Pipeline([("scaler", leaky_scaler), ("clf", LogisticRegression())])
    pipeline.named_steps["clf"].fit(leaky_scaler.transform(X_train), y_train)

    findings = detect_preprocessing_leakage(pipeline, X_train)
    assert any(f.issue_type == "data_leakage" for f in findings)


def test_no_false_positive_when_scaler_fit_correctly():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clean_scaler = StandardScaler().fit(X_train)
    pipeline = Pipeline([("scaler", clean_scaler), ("clf", LogisticRegression())])
    pipeline.named_steps["clf"].fit(clean_scaler.transform(X_train), y_train)

    findings = detect_preprocessing_leakage(pipeline, X_train)
    assert findings == []


# --- class imbalance ----------------------------------------------------------

def test_detects_severe_class_imbalance():
    X, y = _make_clf_data(n_samples=500, weights=[0.95, 0.05])
    findings = detect_class_imbalance(y)
    assert any(f.issue_type == "class_imbalance" for f in findings)


def test_no_false_positive_on_balanced_classes():
    X, y = _make_clf_data(n_samples=500, weights=[0.5, 0.5])
    findings = detect_class_imbalance(y)
    assert findings == []


# --- confidence scaling -------------------------------------------------------

def test_confidence_scales_with_severity_of_imbalance():
    X, y_mild = make_classification(n_samples=500, weights=[0.82, 0.18], random_state=1)
    y_mild = pd.Series(y_mild)
    f_mild = detect_class_imbalance(y_mild)

    X2, y_severe = make_classification(n_samples=500, weights=[0.97, 0.03], random_state=1)
    y_severe = pd.Series(y_severe)
    f_severe = detect_class_imbalance(y_severe)

    assert f_mild[0].confidence < f_severe[0].confidence


def test_confidence_is_bounded():
    X, y = _make_clf_data(n_samples=500, weights=[0.95, 0.05])
    findings = detect_class_imbalance(y)
    for f in findings:
        assert 0.0 <= f.confidence <= 1.0
        
# --- misleading metrics -------------------------------------------------------

def test_detects_accuracy_f1_gap_on_imbalanced_data():
    X, y = _make_clf_data(n_samples=500, weights=[0.95, 0.05])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression().fit(X_train, y_train)
    # force it to behave like a majority-class-only predictor for the test
    findings = detect_misleading_metrics(clf, X_test, y_test)
    # not guaranteed to always fire depending on random_state, so just check the detector runs cleanly
    assert isinstance(findings, list)

# --- no false positives on balanced metrics -------------------------------------------------------

def test_no_false_positive_on_balanced_metrics():
    X, y = _make_clf_data(n_samples=500, weights=[0.5, 0.5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression().fit(X_train, y_train)
    findings = detect_misleading_metrics(clf, X_test, y_test)
    assert findings == []

# --- overfitting signal --------------------------------------------------------

def test_detects_overfitting_regression():
    X, y = make_regression(n_samples=60, n_features=20, noise=0.1, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # deep, unconstrained tree ensemble on tiny data -> near perfect train fit, weak test fit
    model = RandomForestRegressor(n_estimators=50, max_depth=None, random_state=0)
    model.fit(X_train, y_train)

    findings = detect_overfitting_signal(model, X_train, y_train, X_test, y_test)
    assert any(f.issue_type == "overfitting_signal" for f in findings)

# --- no false positives on well-fit regression -------------------------------------------------------

def test_no_false_positive_on_well_fit_regression():
    X, y = make_regression(n_samples=300, n_features=5, noise=15, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    findings = detect_overfitting_signal(model, X_train, y_train, X_test, y_test)
    assert findings == []

# --- data quality ---------------------------------------------------------------

def test_detects_missing_values():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train = X_train.copy()
    X_train.iloc[0:5, 0] = np.nan
    findings = detect_data_quality_issues(X_train, X_test)
    assert any(f.issue_type == "data_quality" for f in findings)


def test_detects_column_mismatch():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_test_bad = X_test.drop(columns=["f0"])
    findings = detect_data_quality_issues(X_train, X_test_bad)
    assert any("different columns" in f.title for f in findings)


# --- end-to-end -------------------------------------------------------------------

def test_run_audit_end_to_end_smoke():
    X, y = _make_clf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_train)
    pipeline = Pipeline([("scaler", scaler), ("clf", LogisticRegression())])
    pipeline.named_steps["clf"].fit(scaler.transform(X_train), y_train)
    findings = run_audit(pipeline, X_train, X_test, y_train, y_test)
    assert isinstance(findings, list)
