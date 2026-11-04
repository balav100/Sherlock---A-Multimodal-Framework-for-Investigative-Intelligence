import pytest
from fastapi import HTTPException

from backend import security


def test_missing_authorization_returns_401():
    with pytest.raises(HTTPException) as exc_info:
        security.get_current_user(None)

    assert exc_info.value.status_code == 401


def test_wrong_authorization_scheme_returns_401():
    with pytest.raises(HTTPException) as exc_info:
        security.get_current_user("Basic abc123")

    assert exc_info.value.status_code == 401


def test_empty_bearer_token_returns_401():
    with pytest.raises(HTTPException) as exc_info:
        security.get_current_user("Bearer   ")

    assert exc_info.value.status_code == 401


def test_invalid_firebase_token_returns_401(monkeypatch):
    monkeypatch.setattr(security, "initialize_firebase", lambda: None)

    def reject_token(*args, **kwargs):
        raise ValueError("Invalid token")

    monkeypatch.setattr(security.auth, "verify_id_token", reject_token)

    with pytest.raises(HTTPException) as exc_info:
        security.get_current_user("Bearer fake-token")

    assert exc_info.value.status_code == 401


def test_valid_firebase_token_returns_verified_identity(monkeypatch):
    monkeypatch.setattr(security, "initialize_firebase", lambda: None)

    def verify_token(token, check_revoked):
        assert token == "valid-test-token"
        assert check_revoked is True
        return {"uid": "test-user-1", "email": "test@example.com"}

    monkeypatch.setattr(security.auth, "verify_id_token", verify_token)

    result = security.get_current_user("Bearer valid-test-token")

    assert result == {
        "uid": "test-user-1",
        "email": "test@example.com",
    }


def test_firebase_infrastructure_error_returns_503(monkeypatch):
    monkeypatch.setattr(security, "initialize_firebase", lambda: None)
    def unavailable(*args, **kwargs):
        raise RuntimeError("Firebase network unavailable")
    monkeypatch.setattr(security.auth, "verify_id_token", unavailable)
    with pytest.raises(HTTPException) as exc_info:
        security.get_current_user("Bearer valid-format-token")
    assert exc_info.value.status_code == 503

