"""
feature_engineering.py

Defines:
  - build_preprocessor(): a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features. This is meant to be placed
    INSIDE a sklearn Pipeline together with the model, so that
    fit() is only ever called on the training split (never on the full
    dataset) - this is what prevents preprocessing leakage.
  - categorize_performance(): converts a numeric final_score into a
    performance-category label (Excellent / Good / Average / Needs
    Improvement). Used to build the CLASSIFICATION target from the
    regression target - this is a legitimate label derivation for a
    separate task, not a feature leak, because it is only ever used as
    the y (target) for the classification model, never as an X (feature).
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from data_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric pipeline: median imputation (robust to outliers) + standard scaling.
    Categorical pipeline: most-frequent imputation + one-hot encoding.

    Wrapped in a ColumnTransformer so it can be fit ONLY on X_train and then
    applied unchanged to X_test / new prediction requests - guaranteeing
    training/inference consistency.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def categorize_performance(score: float) -> str:
    """Map a final_score (0-100) to a performance category.
    NOTE: this function is applied to y (final_score), never to X. It is used
    to create the classification target, not a feature, so it does not
    constitute leakage."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Improvement"


PERFORMANCE_CATEGORIES = ["Needs Improvement", "Average", "Good", "Excellent"]
