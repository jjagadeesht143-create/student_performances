"""
train_model.py

Trains and compares models for BOTH tasks:

  1. Regression   -> predict final_score (continuous)
  2. Classification -> predict performance_category (Excellent/Good/Average/
     Needs Improvement), derived from final_score AFTER the split (see below)

Leakage-safety measures used throughout:
  - train_test_split happens FIRST, before any preprocessing is fit.
  - All imputation/scaling/encoding lives inside sklearn Pipelines, which are
    fit() only on the training split. fit_transform is never called on the
    full dataset.
  - The classification target (performance_category) is derived from
    y (final_score) - never from a feature - and is derived independently
    for the train and test targets after the split, so no information about
    the test set's labels influences training.
  - Cross-validation is done only on the training split.

Run:
    python src/train_model.py
Produces:
    models/student_performance_model.joblib
    outputs/model_comparison_report.txt
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from data_preprocessing import load_raw_data, clean_data, get_features_and_target
from feature_engineering import build_preprocessor, categorize_performance, PERFORMANCE_CATEGORIES

RANDOM_STATE = 42
MODEL_PATH = "models/student_performance_model.joblib"
REPORT_PATH = "outputs/model_comparison_report.txt"


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    # ---------------------------------------------------------------
    # 1. Load + clean data
    # ---------------------------------------------------------------
    raw_df = load_raw_data()
    df = clean_data(raw_df)
    X, y_reg = get_features_and_target(df)

    # ---------------------------------------------------------------
    # 2. Split BEFORE any preprocessing is fit (prevents leakage)
    # ---------------------------------------------------------------
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_STATE
    )

    # Classification target is derived independently, AFTER the split,
    # from each split's own final_score values.
    y_clf_train = y_reg_train.apply(categorize_performance)
    y_clf_test = y_reg_test.apply(categorize_performance)

    log("=" * 70)
    log("TRAIN/TEST SPLIT")
    log("=" * 70)
    log(f"Training rows: {X_train.shape[0]} | Test rows: {X_test.shape[0]}")

    # =================================================================
    # REGRESSION: predict final_score
    # =================================================================
    log("\n" + "=" * 70)
    log("REGRESSION MODELS - predicting final_score")
    log("=" * 70)

    reg_candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    reg_results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in reg_candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        # Cross-validation on the TRAINING split only
        cv_scores = cross_val_score(
            pipe, X_train, y_reg_train, cv=kf, scoring="r2"
        )

        # Fit on full training split, evaluate once on the held-out test split
        pipe.fit(X_train, y_reg_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_reg_test, preds)
        mse = mean_squared_error(y_reg_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_reg_test, preds)

        reg_results[name] = {
            "pipeline": pipe,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "test_r2": r2,
        }

        log(f"\n--- {name} ---")
        log(f"CV R^2 (5-fold, train split): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        log(f"Test MAE : {mae:.4f}")
        log(f"Test MSE : {mse:.4f}")
        log(f"Test RMSE: {rmse:.4f}")
        log(f"Test R^2 : {r2:.4f}")

    best_reg_name = max(reg_results, key=lambda n: reg_results[n]["test_r2"])
    best_reg_pipeline = reg_results[best_reg_name]["pipeline"]
    log(f"\n>>> Selected regression model: {best_reg_name} "
        f"(highest test R^2 = {reg_results[best_reg_name]['test_r2']:.4f})")

    # =================================================================
    # CLASSIFICATION: predict performance_category
    # =================================================================
    log("\n" + "=" * 70)
    log("CLASSIFICATION MODELS - predicting performance_category")
    log("=" * 70)

    clf_candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    clf_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in clf_candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        cv_scores = cross_val_score(
            pipe, X_train, y_clf_train, cv=skf, scoring="accuracy"
        )

        pipe.fit(X_train, y_clf_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_clf_test, preds)
        prec = precision_score(y_clf_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_clf_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_clf_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_clf_test, preds, labels=PERFORMANCE_CATEGORIES)

        clf_results[name] = {
            "pipeline": pipe,
            "cv_acc_mean": cv_scores.mean(),
            "cv_acc_std": cv_scores.std(),
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "confusion_matrix": cm,
        }

        log(f"\n--- {name} ---")
        log(f"CV Accuracy (5-fold, train split): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        log(f"Test Accuracy : {acc:.4f}")
        log(f"Test Precision: {prec:.4f}")
        log(f"Test Recall   : {rec:.4f}")
        log(f"Test F1 Score : {f1:.4f}")
        log(f"Confusion Matrix (labels={PERFORMANCE_CATEGORIES}):")
        log(cm)

    best_clf_name = max(clf_results, key=lambda n: clf_results[n]["test_f1"])
    best_clf_pipeline = clf_results[best_clf_name]["pipeline"]
    log(f"\n>>> Selected classification model: {best_clf_name} "
        f"(highest test weighted F1 = {clf_results[best_clf_name]['test_f1']:.4f})")

    # ---------------------------------------------------------------
    # 3. Save both winning pipelines together (single joblib artifact)
    # ---------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    artifact = {
        "regressor": best_reg_pipeline,
        "regressor_name": best_reg_name,
        "classifier": best_clf_pipeline,
        "classifier_name": best_clf_name,
        "performance_categories": PERFORMANCE_CATEGORIES,
        "feature_columns": list(X.columns),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, MODEL_PATH)
    log(f"\nSaved combined model artifact to {MODEL_PATH}")

    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Saved full comparison report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
