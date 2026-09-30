"""
test_api.py

Tests the Flask API and the underlying prediction function, including
invalid-input handling. Requires the model artifact to already exist
(run `python src/train_model.py` first) - this mirrors a real CI setup
where the model is trained/cached before tests run.

Run from the project root:
    pytest tests/ -v
"""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict_single, validate_input

VALID_PAYLOAD = {
    "study_time": 4,
    "attendance": 85,
    "previous_score": 72,
    "assignment_score": 80,
    "parental_education": "Graduate",
    "extracurricular_activity": "Yes",
}


# ------------------------------------------------------------------
# predict.py unit tests
# ------------------------------------------------------------------

def test_predict_single_valid_input_returns_expected_keys():
    result = predict_single(VALID_PAYLOAD)
    assert "predicted_score" in result
    assert "performance_category" in result
    assert isinstance(result["predicted_score"], float)
    assert 0 <= result["predicted_score"] <= 100
    assert result["performance_category"] in {
        "Excellent", "Good", "Average", "Needs Improvement"
    }


def test_validate_input_missing_field():
    bad = dict(VALID_PAYLOAD)
    del bad["study_time"]
    errors = validate_input(bad)
    assert any("study_time" in e for e in errors)


def test_validate_input_out_of_range_attendance():
    bad = dict(VALID_PAYLOAD)
    bad["attendance"] = 150
    errors = validate_input(bad)
    assert any("attendance" in e for e in errors)


def test_validate_input_invalid_categorical():
    bad = dict(VALID_PAYLOAD)
    bad["parental_education"] = "PhD"  # not one of the allowed categories
    errors = validate_input(bad)
    assert any("parental_education" in e for e in errors)


def test_predict_single_raises_on_invalid_input():
    bad = dict(VALID_PAYLOAD)
    bad["study_time"] = "not-a-number"
    with pytest.raises(ValueError):
        predict_single(bad)


# ------------------------------------------------------------------
# Flask API tests (using Flask's test client - no live server needed)
# ------------------------------------------------------------------

@pytest.fixture
def client():
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_USER", "root")
    os.environ.setdefault("DB_PASSWORD", "")
    os.environ.setdefault("DB_NAME", "student_performance_db")
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        with c.session_transaction() as session:
            session["user_id"] = "test-user"
            session["user_name"] = "Test User"
            session["user_email"] = "test@example.com"
        yield c


def test_api_predict_valid(client):
    resp = client.post("/api/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted_score" in data
    assert "performance_category" in data


def test_api_predict_invalid_range(client):
    bad = dict(VALID_PAYLOAD)
    bad["attendance"] = 500
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_predict_missing_field(client):
    bad = dict(VALID_PAYLOAD)
    del bad["previous_score"]
    resp = client.post("/api/predict", json=bad)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_predict_non_json_body(client):
    resp = client.post("/api/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_index_page_loads(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_prediction_page_loads(client):
    resp = client.get("/prediction")
    assert resp.status_code == 200
