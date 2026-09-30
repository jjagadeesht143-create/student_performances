import os
from unittest.mock import patch

import app as app_module
import pytest


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        yield client


def test_login_and_register_show_google_oauth_button(client):
    assert b"Continue with Google" in client.get("/login").data
    assert b"Continue with Google" in client.get("/register").data


def test_google_callback_creates_new_user(client):
    profile = {
        "sub": "google-new-123",
        "name": "New Google User",
        "email": "new-google@example.com",
        "email_verified": True,
        "picture": "https://example.com/avatar.png",
    }
    with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "client", "GOOGLE_CLIENT_SECRET": "secret"}), \
            patch.object(app_module.google, "authorize_access_token", return_value={"userinfo": profile}), \
            patch.object(app_module.db, "get_user_by_google_id", return_value=None), \
            patch.object(app_module.db, "get_user_by_email", return_value=None), \
            patch.object(app_module.db, "create_user", return_value=14) as create_user:
        response = client.get("/auth/google/callback")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")
    create_user.assert_called_once_with(
        "New Google User", "new-google@example.com",
        google_id="google-new-123", profile_picture="https://example.com/avatar.png",
        auth_provider="google",
    )


def test_google_callback_links_existing_email_without_overwriting_password(client):
    profile = {
        "sub": "google-existing-123",
        "name": "Existing Local User",
        "email": "local@example.com",
        "email_verified": True,
    }
    existing_user = {
        "id": 8,
        "name": "Existing Local User",
        "email": "local@example.com",
        "password_hash": "existing-hash",
        "google_id": None,
        "profile_picture": None,
    }
    with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "client", "GOOGLE_CLIENT_SECRET": "secret"}), \
            patch.object(app_module.google, "authorize_access_token", return_value={"userinfo": profile}), \
            patch.object(app_module.db, "get_user_by_google_id", return_value=None), \
            patch.object(app_module.db, "get_user_by_email", return_value=existing_user), \
            patch.object(app_module.db, "update_user_auth") as update_user:
        response = client.get("/auth/google/callback")

    assert response.status_code == 302
    update_user.assert_called_once_with(8, "google-existing-123", None, "both")
    assert existing_user["password_hash"] == "existing-hash"


def test_google_callback_rejects_identity_collision(client):
    profile = {
        "sub": "google-collision-123",
        "name": "Collision User",
        "email": "collision@example.com",
        "email_verified": True,
    }
    google_user = {"id": 1, "email": "other@example.com", "password_hash": None}
    email_user = {"id": 2, "email": "collision@example.com", "password_hash": "hash"}
    with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "client", "GOOGLE_CLIENT_SECRET": "secret"}), \
            patch.object(app_module.google, "authorize_access_token", return_value={"userinfo": profile}), \
            patch.object(app_module.db, "get_user_by_google_id", return_value=google_user), \
            patch.object(app_module.db, "get_user_by_email", return_value=email_user), \
            patch.object(app_module.db, "update_user_auth") as update_user:
        response = client.get("/auth/google/callback")

    assert response.status_code == 302
    update_user.assert_not_called()
