"""
evaluate_model.py

Loads the saved model artifact and re-generates a full evaluation report,
including a confusion matrix plot, using the SAME train/test split logic
(same random_state) as train_model.py, so results are reproducible.

Run:
    python src/evaluate_model.py
Produces:
    outputs/figures/confusion_matrix.png
    outputs/evaluation_report.txt
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report,
)
import numpy as np

from data_preprocessing import load_raw_data, clean_data, get_features_and_target
from feature_engineering import categorize_performance, PERFORMANCE_CATEGORIES

MODEL_PATH = "models/student_performance_model.joblib"


def main():
    artifact = joblib.load(MODEL_PATH)
    reg_pipeline = artifact["regressor"]
    clf_pipeline = artifact["classifier"]
    random_state = artifact["random_state"]

    raw_df = load_raw_data()
    df = clean_data(raw_df)
    X, y_reg = get_features_and_target(df)

    # Use the SAME split parameters as training, so this is a genuine
    # re-evaluation of the held-out test set, not a re-fit.
    _, X_test, _, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=random_state
    )
    y_clf_test = y_reg_test.apply(categorize_performance)

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    # ---- Regression evaluation ----
    reg_preds = reg_pipeline.predict(X_test)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    mse = mean_squared_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, reg_preds)

    log("=" * 60)
    log(f"REGRESSION EVALUATION ({artifact['regressor_name']})")
    log("=" * 60)
    log(f"MAE : {mae:.4f}")
    log(f"MSE : {mse:.4f}")
    log(f"RMSE: {rmse:.4f}")
    log(f"R^2 : {r2:.4f}")

    # ---- Classification evaluation ----
    clf_preds = clf_pipeline.predict(X_test)
    acc = accuracy_score(y_clf_test, clf_preds)
    prec = precision_score(y_clf_test, clf_preds, average="weighted", zero_division=0)
    rec = recall_score(y_clf_test, clf_preds, average="weighted", zero_division=0)
    f1 = f1_score(y_clf_test, clf_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_clf_test, clf_preds, labels=PERFORMANCE_CATEGORIES)

    log("\n" + "=" * 60)
    log(f"CLASSIFICATION EVALUATION ({artifact['classifier_name']})")
    log("=" * 60)
    log(f"Accuracy : {acc:.4f}")
    log(f"Precision: {prec:.4f}")
    log(f"Recall   : {rec:.4f}")
    log(f"F1 Score : {f1:.4f}")
    log("\nFull classification report:")
    log(classification_report(y_clf_test, clf_preds, labels=PERFORMANCE_CATEGORIES, zero_division=0))

    # ---- Confusion matrix plot ----
    os.makedirs("outputs/figures", exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=PERFORMANCE_CATEGORIES, yticklabels=PERFORMANCE_CATEGORIES)
    plt.title(f"Confusion Matrix - {artifact['classifier_name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("outputs/figures/confusion_matrix.png", dpi=120)
    plt.close()
    log("\nSaved confusion matrix plot to outputs/figures/confusion_matrix.png")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.txt", "w") as f:
        f.write("\n".join(lines))
    log("Saved evaluation_report.txt")


if __name__ == "__main__":
    main()
