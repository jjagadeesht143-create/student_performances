"""
data_preprocessing.py

Responsible for:
    1. Loading the raw CSV.
    2. Producing a "before cleaning" report (shape, dtypes, missing values,
       duplicates, descriptive stats) so decisions are documented, not hidden.
    3. Cleaning the data (drop duplicates, fix dtypes) WITHOUT doing anything
       that would leak test-set information (no scaling/imputation happens
       here at the whole-dataset level for the features used by the model —
       that is handled inside the sklearn Pipeline after the train/test split).

This module is imported by both the notebooks and src/train_model.py so the
exact same cleaning logic is used everywhere.
"""

import pandas as pd


RAW_DATA_PATH = "data/raw/students.csv"

# Columns that are realistically available BEFORE the final exam takes place.
# These (and only these) become model input features later on.
FEATURE_COLUMNS = [
    "study_time",
    "attendance",
    "previous_score",
    "assignment_score",
    "parental_education",
    "extracurricular_activity",
]
TARGET_COLUMN = "final_score"

NUMERIC_FEATURES = ["study_time", "attendance", "previous_score", "assignment_score"]
CATEGORICAL_FEATURES = ["parental_education", "extracurricular_activity"]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file as-is."""
    return pd.read_csv(path)


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Return a dictionary summarizing data quality BEFORE any cleaning.
    This is printed/shown explicitly (per project requirements) instead of
    silently cleaning the data.
    """
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "descriptive_stats": df.describe(include="all").to_dict(),
    }
    return report


def print_data_quality_report(df: pd.DataFrame) -> None:
    report = data_quality_report(df)
    print("=" * 60)
    print("DATA QUALITY REPORT (before cleaning)")
    print("=" * 60)
    print(f"Shape: {report['shape']}")
    print("\nData types:")
    for col, dtype in report["dtypes"].items():
        print(f"  {col}: {dtype}")
    print("\nMissing values per column:")
    for col, n in report["missing_values"].items():
        if n > 0:
            print(f"  {col}: {n} missing")
    if sum(report["missing_values"].values()) == 0:
        print("  (none)")
    print(f"\nDuplicate rows: {report['duplicate_rows']}")
    print("=" * 60)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset at the WHOLE-DATASET level only in ways that do not
    leak information across the future train/test split:

      - Drop exact duplicate rows (a duplicate row is not "new information",
        it is a data entry artifact - safe to drop before splitting).
      - Drop rows with a missing target (final_score). We cannot train or
        evaluate on a row where the label itself is unknown, and imputing
        a label would be fabricating ground truth.
      - Enforce sane dtypes.

    What we deliberately do NOT do here:
      - Impute missing FEATURE values (study_time, attendance, etc.) using
        statistics (mean/median) computed on the full dataset. Doing so
        would leak information from the future test split into training.
        That imputation is instead done inside the sklearn Pipeline, fitted
        only on the training split.
      - Scale/encode features here, for the same reason.
    """
    before_shape = df.shape

    # 1. Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Drop rows with missing target - can't train/evaluate without a label
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # 3. Enforce dtypes
    for col in NUMERIC_FEATURES + [TARGET_COLUMN, "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES + ["gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    after_shape = df.shape
    print(
        f"clean_data: {before_shape} -> {after_shape} "
        f"(removed {before_shape[0] - after_shape[0]} rows: duplicates / missing target)"
    )
    return df


def get_features_and_target(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (target).
    Only FEATURE_COLUMNS are used as X - final_score and any leakage-prone
    columns (student_id, name) are excluded on purpose.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


if __name__ == "__main__":
    raw_df = load_raw_data()
    print_data_quality_report(raw_df)
    cleaned_df = clean_data(raw_df)
    print("\nCleaned data preview:")
    print(cleaned_df.head())
