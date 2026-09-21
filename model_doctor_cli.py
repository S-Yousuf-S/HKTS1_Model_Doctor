#!/usr/bin/env python3
"""
Model Doctor CLI — audit any fitted scikit-learn-style model saved as
.pkl/.joblib against a train/test CSV pair.

Author: Yousuf S.R. Sakkaf
File: model_doctor_cli.py

Usage:
    python cli.py --model model.pkl --train train.csv --test test.csv --target Survived --out audit_report.html
"""
import argparse
from datetime import datetime
from model_doctor import audit_from_files, generate_html_report, generate_markdown_report


def main():
    parser = argparse.ArgumentParser(
        description="Model Doctor — audit a saved model against train/test CSVs.")
    parser.add_argument("--model", required=True, help="Path to a .pkl or .joblib fitted model")
    parser.add_argument("--train", required=True, help="Path to training CSV (includes target column)")
    parser.add_argument("--test", required=True, help="Path to test CSV (includes target column)")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--out", default=f"model_doctor_audit_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.html", help="Output report path (.html or .md)")
    args = parser.parse_args()

    findings, X_train, X_test, y_train, y_test = audit_from_files(
        args.model, args.train, args.test, args.target)

    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_warn = sum(1 for f in findings if f.severity == "warning")
    print(f"Model Doctor audit complete: {len(findings)} finding(s) \u2014 {n_crit} critical, {n_warn} warning")
    for f in findings:
        print(f"  [{f.severity.upper()}] {f.issue_type}: {f.title}")

    report = (generate_markdown_report(findings, model_name=args.model)
              if args.out.endswith(".md")
              else generate_html_report(findings, model_name=args.model))

    with open(args.out, "w") as fh:
        fh.write(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
