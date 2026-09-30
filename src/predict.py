"""
predict.py

Loads the saved model artifact ONCE and exposes a simple predict_single()
function that the Flask app (and tests) call. Using the exact same joblib
pipeline for inference as was used for training/testing guarantees that
preprocessing (imputation, scaling, encoding) is 100% consistent.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "student_performance_model.joblib")

VALID_PARENTAL_EDUCATION = {"High School", "Bachelors", "Graduate"}
VALID_EXTRACURRICULAR = {"Yes", "No"}

_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python src/train_model.py` first."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def validate_input(data: dict) -> list:
    """Return a list of human-readable error messages (empty list = valid)."""
    errors = []

    required = [
        "study_time", "attendance", "previous_score",
        "assignment_score", "parental_education", "extracurricular_activity",
    ]
    for field in required:
        if field not in data or data[field] in (None, ""):
            errors.append(f"'{field}' is required.")
    if errors:
        return errors

    try:
        study_time = float(data["study_time"])
        if not (0 <= study_time <= 24):
            errors.append("'study_time' must be between 0 and 24 hours.")
    except (TypeError, ValueError):
        errors.append("'study_time' must be a number.")

    try:
        attendance = float(data["attendance"])
        if not (0 <= attendance <= 100):
            errors.append("'attendance' must be between 0 and 100.")
    except (TypeError, ValueError):
        errors.append("'attendance' must be a number.")

    try:
        previous_score = float(data["previous_score"])
        if not (0 <= previous_score <= 100):
            errors.append("'previous_score' must be between 0 and 100.")
    except (TypeError, ValueError):
        errors.append("'previous_score' must be a number.")

    try:
        assignment_score = float(data["assignment_score"])
        if not (0 <= assignment_score <= 100):
            errors.append("'assignment_score' must be between 0 and 100.")
    except (TypeError, ValueError):
        errors.append("'assignment_score' must be a number.")

    if data.get("parental_education") not in VALID_PARENTAL_EDUCATION:
        errors.append(
            f"'parental_education' must be one of {sorted(VALID_PARENTAL_EDUCATION)}."
        )

    if data.get("extracurricular_activity") not in VALID_EXTRACURRICULAR:
        errors.append(
            f"'extracurricular_activity' must be one of {sorted(VALID_EXTRACURRICULAR)}."
        )

    return errors


def predict_single(data: dict) -> dict:
    """
    data: dict with keys study_time, attendance, previous_score,
          assignment_score, parental_education, extracurricular_activity

    Returns: {"predicted_score": float, "performance_category": str}

    Raises ValueError if input is invalid (caller / Flask route should catch
    this and return a 400 response).
    """
    errors = validate_input(data)
    if errors:
        raise ValueError("; ".join(errors))

    artifact = _load_artifact()
    feature_columns = artifact["feature_columns"]

    row = pd.DataFrame([{
        "study_time": float(data["study_time"]),
        "attendance": float(data["attendance"]),
        "previous_score": float(data["previous_score"]),
        "assignment_score": float(data["assignment_score"]),
        "parental_education": data["parental_education"],
        "extracurricular_activity": data["extracurricular_activity"],
    }])[feature_columns]

    predicted_score = float(artifact["regressor"].predict(row)[0])
    predicted_score = round(max(0.0, min(100.0, predicted_score)), 1)

    performance_category = artifact["classifier"].predict(row)[0]

    return {
        "predicted_score": predicted_score,
        "performance_category": performance_category,
        "regressor_used": artifact["regressor_name"],
        "classifier_used": artifact["classifier_name"],
        "disclaimer": "This is a model-based estimate, not a guaranteed result.",
    }


def get_model_info() -> dict:
    """Return display metadata derived from the fitted artifact."""
    artifact = _load_artifact()
    regressor = artifact["regressor"]
    estimator = regressor.named_steps["model"]
    factors = []
    coefficients = getattr(estimator, "coef_", None)
    preprocessor = regressor.named_steps.get("preprocessor")
    if coefficients is not None and preprocessor is not None:
        names = preprocessor.get_feature_names_out()
        values = coefficients[0] if getattr(coefficients, "ndim", 1) > 1 else coefficients
        grouped = {feature: 0.0 for feature in artifact["feature_columns"]}
        for transformed_name, coefficient in zip(names, values):
            raw_name = transformed_name.split("__", 1)[-1]
            feature = next((item for item in grouped if raw_name.startswith(item)), raw_name)
            grouped[feature] += abs(float(coefficient))
        maximum = max(grouped.values(), default=0)
        factors = [
            {"name": name.replace("_", " ").title(), "value": round(value / maximum * 100) if maximum else 0}
            for name, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        ]
    return {
        "regressor": artifact["regressor_name"],
        "classifier": artifact["classifier_name"],
        "target": "final_score",
        "features": artifact["feature_columns"],
        "feature_count": len(artifact["feature_columns"]),
        "factors": factors,
    }


if __name__ == "__main__":
    # Small manual smoke test
    sample = {
        "study_time": 4,
        "attendance": 85,
        "previous_score": 72,
        "assignment_score": 80,
        "parental_education": "Graduate",
        "extracurricular_activity": "Yes",
    }
    print(predict_single(sample))
