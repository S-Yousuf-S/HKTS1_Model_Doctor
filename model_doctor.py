"""
Model Doctor — Automated ML Audit Toolkit
Core module: detectors + report generator.

File: model_doctor.py
Author: Yousuf S.R. Sakkaf

Design principle: every detector is generic — it inspects a *fitted*
sklearn-style estimator/Pipeline and the train/test data it was fit
on, using duck-typing (predict / predict_proba / classes_ / steps)
rather than checking for a specific model class. This is what lets
the toolkit work across LogisticRegression, RandomForest, XGBoost,
etc. without hardcoding.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import copy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score, mean_absolute_error,
)


# ---------------------------------------------------------------------------
# Finding schema
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"critical": 3, "warning": 2, "info": 1}


@dataclass
class AuditFinding:
    issue_type: str            # e.g. "data_leakage", "class_imbalance"
    severity: str               # "critical" | "warning" | "info"
    title: str                  # short human-readable headline
    message: str                 # plain-language explanation, non-technical friendly
    evidence: dict = field(default_factory=dict)   # supporting numbers
    suggested_fix: Optional[str] = None

    def to_dict(self):
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
        }


def _is_classifier(estimator) -> bool:
    return hasattr(estimator, "predict_proba") or hasattr(estimator, "classes_")


def _get_final_estimator(pipeline):
    """Return the last step of a Pipeline, or the estimator itself."""
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    return pipeline


def _preprocessing_steps(pipeline):
    """Return all-but-last steps of a Pipeline (the preprocessing chain)."""
    if hasattr(pipeline, "steps"):
        return pipeline.steps[:-1]
    return []


# ---------------------------------------------------------------------------
# Detector 1: Train/test duplicate contamination
# ---------------------------------------------------------------------------

def detect_duplicate_contamination(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[AuditFinding]:
    findings = []
    try:
        train_hashes = pd.util.hash_pandas_object(X_train, index=False)
        test_hashes = pd.util.hash_pandas_object(X_test, index=False)
        overlap = set(train_hashes) & set(test_hashes)
        n_overlap = len(overlap)
        if n_overlap > 0:
            pct = 100 * n_overlap / max(len(X_test), 1)
            severity = "critical" if pct > 1 else "warning"
            findings.append(AuditFinding(
                issue_type="train_test_contamination",
                severity=severity,
                title="Duplicate rows found across train and test sets",
                message=(
                    f"{n_overlap} rows appear in both the training and test data "
                    f"({pct:.2f}% of the test set). This means the model may have "
                    "effectively 'seen' some of the data it's being evaluated on, "
                    "which inflates test performance and hides how the model would "
                    "really perform on new data."
                ),
                evidence={"duplicate_row_count": n_overlap, "pct_of_test": round(pct, 3)},
                suggested_fix=(
                    "De-duplicate the full dataset before splitting, or re-split "
                    "using a method that guarantees no row (or group, e.g. same "
                    "customer/session) appears in both sets."
                ),
            ))
    except Exception as e:
        findings.append(_detector_error("train_test_contamination", e))
    return findings


# ---------------------------------------------------------------------------
# Detector 2: Preprocessing fit-before-split leakage
# ---------------------------------------------------------------------------

def detect_preprocessing_leakage(pipeline, X_train: pd.DataFrame) -> list[AuditFinding]:
    """
    Heuristic: refit a clone of each preprocessing step on X_train ONLY,
    then compare its learned parameters (mean_, scale_, categories_, etc.)
    to the parameters already stored in the fitted pipeline. A meaningful
    mismatch suggests the original step was fit on more data than X_train
    alone (e.g. fit on the full dataset before the split).
    """
    findings = []
    steps = _preprocessing_steps(pipeline)
    if not steps:
        return findings

    X_prefix = X_train
    for name, step in steps:
        try:
            for attr in ("mean_", "scale_", "data_min_", "data_max_"):
                if hasattr(step, attr):
                    fitted_val = getattr(step, attr)
                    refit_step = clone(step)
                    refit_step.fit(X_prefix)
                    refit_val = getattr(refit_step, attr)
                    if not np.allclose(fitted_val, refit_val, rtol=1e-3, atol=1e-6):
                        findings.append(AuditFinding(
                            issue_type="data_leakage",
                            severity="critical",
                            title=f"Preprocessing step '{name}' may have been fit before the train/test split",
                            message=(
                                f"The fitted '{name}' step's learned statistics ({attr}) "
                                "don't match what you'd get fitting it on the training "
                                "data alone. This usually means the scaler/encoder saw "
                                "the test (or full) data during fitting — a classic "
                                "leakage bug that makes test performance look better "
                                "than it really is."
                            ),
                            evidence={"step": name, "attribute": attr},
                            suggested_fix=(
                                f"Fit '{name}' only on X_train inside the pipeline, "
                                "then transform X_test with the already-fitted step. "
                                "Never call .fit() or .fit_transform() on data that "
                                "includes the test set."
                            ),
                        ))
                    break  # one matching attribute is enough evidence either way
        except Exception as e:
            findings.append(_detector_error("data_leakage", e))
    return findings


# ---------------------------------------------------------------------------
# Detector 3: Class imbalance blindness
# ---------------------------------------------------------------------------

def detect_class_imbalance(y_train, imbalance_ratio_threshold: float = 4.0) -> list[AuditFinding]:
    findings = []
    y_series = pd.Series(y_train)
    if y_series.nunique() > 20:  # looks like regression, not classification
        return findings
    counts = y_series.value_counts()
    if len(counts) < 2:
        return findings
    ratio = counts.iloc[0] / counts.iloc[-1]
    if ratio >= imbalance_ratio_threshold:
        severity = "critical" if ratio >= 10 else "warning"
        findings.append(AuditFinding(
            issue_type="class_imbalance",
            severity=severity,
            title="Significant class imbalance detected",
            message=(
                f"The majority class outnumbers the minority class by {ratio:.1f}:1 "
                f"in the training data (classes: {dict(counts)}). A model can score "
                "deceptively well here just by mostly predicting the majority class, "
                "without actually learning to distinguish the minority class."
            ),
            evidence={"class_counts": counts.to_dict(), "ratio": round(float(ratio), 2)},
            suggested_fix=(
                "Use class_weight='balanced', resampling (SMOTE/undersampling), "
                "or at minimum report precision/recall/F1 per class instead of "
                "plain accuracy."
            ),
        ))
    return findings


# ---------------------------------------------------------------------------
# Detector 4: Misleading metrics (accuracy on imbalanced data)
# ---------------------------------------------------------------------------

def detect_misleading_metrics(estimator, X_test, y_test) -> list[AuditFinding]:
    findings = []
    if not _is_classifier(_get_final_estimator(estimator)):
        return findings
    try:
        y_pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        gap = acc - f1_macro
        if gap > 0.15:
            findings.append(AuditFinding(
                issue_type="misleading_metrics",
                severity="warning" if gap < 0.3 else "critical",
                title="Accuracy looks good but macro-F1 tells a different story",
                message=(
                    f"Accuracy is {acc:.1%}, but macro-averaged F1 is only "
                    f"{f1_macro:.1%}. This gap usually means the model is doing "
                    "well on the majority class and poorly on minority class(es) "
                    "— a sign that accuracy alone is a misleading headline metric "
                    "for this dataset."
                ),
                evidence={"accuracy": round(float(acc), 4), "f1_macro": round(float(f1_macro), 4)},
                suggested_fix=(
                    "Report precision/recall/F1 per class (and macro-average) "
                    "alongside accuracy, and use those—not accuracy—to judge "
                    "the model on imbalanced data."
                ),
            ))
    except Exception as e:
        findings.append(_detector_error("misleading_metrics", e))
    return findings


# ---------------------------------------------------------------------------
# Detector 5: Overfitting signal (train vs test gap)
# ---------------------------------------------------------------------------

def detect_overfitting_signal(estimator, X_train, y_train, X_test, y_test) -> list[AuditFinding]:
    findings = []
    is_clf = _is_classifier(_get_final_estimator(estimator))
    try:
        if is_clf:
            train_score = accuracy_score(y_train, estimator.predict(X_train))
            test_score = accuracy_score(y_test, estimator.predict(X_test))
            metric_name = "accuracy"
        else:
            train_score = r2_score(y_train, estimator.predict(X_train))
            test_score = r2_score(y_test, estimator.predict(X_test))
            metric_name = "R\u00b2"
        gap = train_score - test_score
        if gap > 0.15 or train_score > 0.995:
            severity = "critical" if gap > 0.3 or train_score > 0.999 else "warning"
            findings.append(AuditFinding(
                issue_type="overfitting_signal",
                severity=severity,
                title="Large train/test performance gap (possible overfitting)",
                message=(
                    f"Train {metric_name} is {train_score:.3f} vs test {metric_name} "
                    f"{test_score:.3f} (gap of {gap:.3f}). Suspiciously high train "
                    "scores or a large train-test gap usually mean the model has "
                    "memorized the training data rather than learned generalizable "
                    "patterns."
                ),
                evidence={"train_score": round(float(train_score), 4),
                          "test_score": round(float(test_score), 4),
                          "metric": metric_name},
                suggested_fix=(
                    "Add regularization, reduce model complexity/depth, use "
                    "cross-validation to tune hyperparameters, or gather more "
                    "training data."
                ),
            ))
    except Exception as e:
        findings.append(_detector_error("overfitting_signal", e))
    return findings


# ---------------------------------------------------------------------------
# Detector 6: Data quality issues
# ---------------------------------------------------------------------------

def detect_data_quality_issues(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[AuditFinding]:
    findings = []
    try:
        train_na = X_train.isna().mean()
        bad_cols = train_na[train_na > 0]
        if len(bad_cols) > 0:
            findings.append(AuditFinding(
                issue_type="data_quality",
                severity="warning",
                title="Missing values present in training data",
                message=(
                    f"{len(bad_cols)} column(s) have missing values "
                    f"(worst: '{bad_cols.idxmax()}' at {bad_cols.max():.1%} missing). "
                    "Depending on how these are handled, missing values can silently "
                    "bias the model or cause train/inference mismatches."
                ),
                evidence={"columns_with_na": bad_cols.round(4).to_dict()},
                suggested_fix="Impute explicitly (median/mode/model-based) inside the pipeline, and verify the same imputer is applied at inference time.",
            ))

        train_cols, test_cols = set(X_train.columns), set(X_test.columns)
        if train_cols != test_cols:
            findings.append(AuditFinding(
                issue_type="data_quality",
                severity="critical",
                title="Train and test sets have different columns",
                message=(
                    f"Columns only in train: {sorted(train_cols - test_cols)}. "
                    f"Columns only in test: {sorted(test_cols - train_cols)}. "
                    "This will break or silently misalign predictions at inference time."
                ),
                evidence={"train_only": sorted(train_cols - test_cols),
                          "test_only": sorted(test_cols - train_cols)},
                suggested_fix="Ensure identical feature engineering/encoding is applied to train and test (and future production data).",
            ))

        for col in X_train.select_dtypes(include=["object", "category"]).columns:
            if col in X_test.columns:
                unseen = set(X_test[col].dropna().unique()) - set(X_train[col].dropna().unique())
                if unseen:
                    findings.append(AuditFinding(
                        issue_type="data_quality",
                        severity="warning",
                        title=f"Unseen categories in test column '{col}'",
                        message=(
                            f"Test data contains categories in '{col}' not seen during "
                            f"training: {sorted(list(unseen))[:5]}{'...' if len(unseen) > 5 else ''}. "
                            "Depending on the encoder, this can cause errors or silently "
                            "incorrect encodings at inference time."
                        ),
                        evidence={"column": col, "unseen_categories": sorted(list(unseen))[:10]},
                        suggested_fix="Use an encoder with explicit unknown-category handling (e.g. OneHotEncoder(handle_unknown='ignore')).",
                    ))
    except Exception as e:
        findings.append(_detector_error("data_quality", e))
    return findings


def _detector_error(issue_type: str, e: Exception) -> AuditFinding:
    return AuditFinding(
        issue_type=issue_type,
        severity="info",
        title=f"Detector for '{issue_type}' could not run",
        message=f"Skipped due to an internal error: {e}",
        evidence={},
        suggested_fix=None,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_audit(pipeline, X_train, X_test, y_train, y_test) -> list[AuditFinding]:
    """Run every detector and return a flat, severity-sorted list of findings."""
    findings: list[AuditFinding] = []
    findings += detect_duplicate_contamination(X_train, X_test)
    findings += detect_preprocessing_leakage(pipeline, X_train)
    findings += detect_class_imbalance(y_train)
    findings += detect_misleading_metrics(pipeline, X_test, y_test)
    findings += detect_overfitting_signal(pipeline, X_train, y_train, X_test, y_test)
    findings += detect_data_quality_issues(X_train, X_test)

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 0), reverse=True)
    return findings


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Report generation — styled to match the project's established HTML theme
# (section_header / subsection_header / colored severity boxes, same visual
# language as prerequisites.py's success_box/warning_box/info_box pattern)
# ---------------------------------------------------------------------------

PRIMARY_COLOR = "darkcyan"
SECONDARY_COLOR = "mintcream"

SEVERITY_STYLE = {
    "critical": {"color": "#C0392B", "bg": "#FDEDEC", "icon": "\U0001F534", "label": "Critical"},
    "warning":  {"color": "#B9770E", "bg": "#FEF5E7", "icon": "\U0001F7E1", "label": "Warning"},
    "info":     {"color": "#1F77B4", "bg": "#F1F3F6", "icon": "\U0001F535", "label": "Info"},
}


def _section_header_html(title: str) -> str:
    return (f'<div style="background:{PRIMARY_COLOR};color:white;padding:16px;'
            f'border-radius:8px;font-size:26px;font-weight:bold;margin-top:10px;'
            f'margin-bottom:12px;text-align:center;">{title}</div>')


def _subsection_header_html(title: str) -> str:
    return (f'<div style="background:{SECONDARY_COLOR};border-left:6px solid {PRIMARY_COLOR};'
            f'color:{PRIMARY_COLOR};padding:10px;border-radius:6px;font-size:18px;'
            f'font-weight:bold;margin-top:12px;margin-bottom:10px;text-align:center;">'
            f'<i>{title}</i></div>')


def _finding_box_html(f: AuditFinding) -> str:
    style = SEVERITY_STYLE.get(f.severity, SEVERITY_STYLE["info"])
    fix_html = (f'<div style="margin-top:6px;color:black;"><b>Suggested fix:</b> {f.suggested_fix}</div>'
                if f.suggested_fix else "")
    return (
        f'<div style="background:{style["bg"]};border-left:6px solid {style["color"]};'
        f'padding:12px 16px;border-radius:6px;margin:10px 0;line-height:1.5;">'
        f'<div style="color:{style["color"]};font-weight:bold;">{style["icon"]} {style["label"]} — {f.title}:</div>'
        f'<div style="color:black;margin-top:4px;">{f.message}</div>'
        f'{fix_html}</div>'
    )


def _metrics_row_html(metrics: dict | None) -> str:
    if not metrics:
        return ""
    task = metrics.get("task", "").capitalize()
    parts = " &nbsp;|&nbsp; ".join(
        f"<b>{k}:</b> {v}" for k, v in metrics.items() if k != "task"
    )
    return (
        f'<div style="background:#f3f4f6;border-radius:6px;padding:10px 16px;'
        f'margin-bottom:16px;text-align:center;color:#333;">'
        f'<b>Task: {task}</b> &nbsp;|&nbsp; {parts}</div>'
    )


def generate_html_report(findings: list[AuditFinding], model_name: str = "Model",
                          metrics: dict | None = None) -> str:
    """
    Full standalone HTML document styled to match this project's established
    theme (indigo section headers, thistle subsection headers, colored
    severity boxes) rather than a generic card layout. All styling lives on
    inline-styled <div>s, not <body> — Jupyter's display(HTML(...)) discards
    <body> styling, so this keeps the report looking identical whether it's
    opened standalone or shown inline in a notebook cell.
    """
    n_critical = sum(1 for f in findings if f.severity == "critical")
    n_warning = sum(1 for f in findings if f.severity == "warning")
    n_info = sum(1 for f in findings if f.severity == "info")

    if findings:
        boxes = "".join(_finding_box_html(f) for f in findings)
    else:
        boxes = ('<div style="text-align:center;color:#2E8B57;font-weight:bold;padding:16px;">'
                  '\u2705 No issues detected by the current checks.</div>')

    body_content = f"""
    <div style="font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 850px;
                margin: 0 auto; background: #ffffff; padding: 24px 32px; border-radius: 10px;">
      {_section_header_html("\U0001FA7A Model Doctor — Audit Report")}
      {_subsection_header_html(model_name)}
      {_metrics_row_html(metrics)}
      <p style="text-align:center; color:#333;">
        <b>{n_critical}</b> critical &nbsp;|&nbsp; <b>{n_warning}</b> warning &nbsp;|&nbsp; <b>{n_info}</b> info
      </p>
      {boxes}
    </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Doctor Audit Report — {model_name}</title></head>
<body style="margin:0; padding: 24px; background:#f3f4f6;">{body_content}</body></html>"""


def generate_markdown_report(findings: list[AuditFinding], model_name: str = "Model") -> str:
    lines = [f"# \U0001FA7A Model Doctor — Audit Report\n## {model_name}\n"]
    for f in findings:
        style = SEVERITY_STYLE.get(f.severity, SEVERITY_STYLE["info"])
        lines.append(f"### {style['icon']} {style['label']} — {f.title}\n{f.message}\n")
        if f.suggested_fix:
            lines.append(f"**Suggested fix:** {f.suggested_fix}\n")
    if not findings:
        lines.append("\u2705 No issues detected by the current checks.\n")
    return "\n".join(lines)
