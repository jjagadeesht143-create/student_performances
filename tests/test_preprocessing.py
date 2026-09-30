"""
test_preprocessing.py

Run from the project root:
    pytest tests/ -v
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    clean_data, get_features_and_target, FEATURE_COLUMNS, TARGET_COLUMN
)


def make_dirty_df():
    return pd.DataFrame({
        "student_id": [1, 2, 3, 3],           # duplicate row (id 3 twice)
        "name": ["A", "B", "C", "C"],
        "age": [16, 17, 18, 18],
        "gender": ["Male", "Female", "Male", "Male"],
        "study_time": [3.0, 4.0, None, None],
        "attendance": [80.0, 90.0, 70.0, 70.0],
        "previous_score": [60.0, 70.0, 80.0, 80.0],
        "assignment_score": [65.0, 75.0, 85.0, 85.0],
        "parental_education": ["Bachelors", "Graduate", "High School", "High School"],
        "extracurricular_activity": ["Yes", "No", "Yes", "Yes"],
        "final_score": [70.0, 80.0, None, None],  # row 3/4 missing target
    })


def test_clean_data_removes_duplicates():
    df = make_dirty_df()
    cleaned = clean_data(df)
    assert cleaned.duplicated().sum() == 0


def test_clean_data_removes_missing_target_rows():
    df = make_dirty_df()
    cleaned = clean_data(df)
    assert cleaned[TARGET_COLUMN].isnull().sum() == 0
    # Both duplicate rows with missing final_score should be gone
    assert len(cleaned) == 2


def test_get_features_and_target_excludes_leakage_columns():
    df = make_dirty_df()
    cleaned = clean_data(df)
    X, y = get_features_and_target(cleaned)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "final_score" not in X.columns
    assert "student_id" not in X.columns
    assert "name" not in X.columns
    assert len(y) == len(X)
