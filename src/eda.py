"""
eda.py

Generates the exploratory data analysis charts required for the project and
saves them to outputs/figures/. Run this AFTER data has been cleaned.

Charts produced:
    1. score_distribution.png
    2. attendance_distribution.png
    3. study_time_vs_score.png
    4. attendance_vs_score.png
    5. previous_score_vs_score.png
    6. gender_performance.png
    7. performance_category_distribution.png
    8. correlation_heatmap.png
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_preprocessing import load_raw_data, clean_data

FIG_DIR = "outputs/figures"
sns.set_theme(style="whitegrid")


def categorize_performance(score: float) -> str:
    """Business-defined performance buckets, based ONLY on final_score.
    This is used purely for descriptive/EDA purposes and for labeling the
    classification target - it is not a feature."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"


def run_eda(df: pd.DataFrame) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = df.copy()
    df["performance_category"] = df["final_score"].apply(categorize_performance)

    # 1. Score distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["final_score"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/score_distribution.png", dpi=120)
    plt.close()

    # 2. Attendance distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["attendance"].dropna(), bins=20, kde=True, color="seagreen")
    plt.title("Distribution of Attendance (%)")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/attendance_distribution.png", dpi=120)
    plt.close()

    # 3. Study time vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="study_time", y="final_score", hue="gender", alpha=0.7)
    plt.title("Study Time vs Final Score")
    plt.xlabel("Study Time (hours/day)")
    plt.ylabel("Final Score")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/study_time_vs_score.png", dpi=120)
    plt.close()

    # 4. Attendance vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="attendance", y="final_score", alpha=0.7, color="darkorange")
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/attendance_vs_score.png", dpi=120)
    plt.close()

    # 5. Previous score vs final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="previous_score", y="final_score", alpha=0.7, color="purple")
    plt.title("Previous Score vs Final Score")
    plt.xlabel("Previous Score")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/previous_score_vs_score.png", dpi=120)
    plt.close()

    # 6. Gender-wise performance
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="gender", y="final_score", hue="gender", palette="Set2", legend=False)
    plt.title("Final Score by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/gender_performance.png", dpi=120)
    plt.close()

    # 7. Performance category distribution
    plt.figure(figsize=(7, 5))
    order = ["Needs Improvement", "Average", "Good", "Excellent"]
    sns.countplot(data=df, x="performance_category", order=order, hue="performance_category",
                  palette="viridis", legend=False)
    plt.title("Performance Category Distribution")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/performance_category_distribution.png", dpi=120)
    plt.close()

    # 8. Correlation heatmap (numeric features only)
    numeric_cols = ["age", "study_time", "attendance", "previous_score",
                     "assignment_score", "final_score"]
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()

    print(f"Saved 8 EDA figures to {FIG_DIR}/")


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    run_eda(cleaned_df)
