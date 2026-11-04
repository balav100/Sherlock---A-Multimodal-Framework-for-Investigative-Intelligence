from fastapi.testclient import TestClient

from backend.main import app
from backend.security import get_current_user
from backend import security


def test_protected_route_rejects_missing_bearer_token():
    client = TestClient(app)
    response = client.get("/api/cases")

    assert response.status_code == 401


def test_protected_route_rejects_invalid_firebase_token(monkeypatch):
    monkeypatch.setattr(
        security,
        "initialize_firebase",
        lambda: None,
    )

    def reject_token(*args, **kwargs):
        raise ValueError("Invalid test token")

    monkeypatch.setattr(
        security.auth,
        "verify_id_token",
        reject_token,
    )

    client = TestClient(app)
    response = client.get(
        "/api/cases",
        headers={"Authorization": "Bearer fake-test-token"},
    )

    assert response.status_code == 401


def test_protected_route_accepts_mock_verified_identity(monkeypatch):
    monkeypatch.setattr(
        security,
        "initialize_firebase",
        lambda: None,
    )
    monkeypatch.setattr(
        security.auth,
        "verify_id_token",
        lambda token, check_revoked: {
            "uid": "test-user",
            "email": "test@example.com",
        },
    )

    import backend.routes.cases as cases
    class EmptyReference:
        def get(self):
            return {}
    monkeypatch.setattr(cases, "database_ref", lambda path: EmptyReference())
    app.dependency_overrides[get_current_user] = lambda: (
        get_current_user("Bearer valid-test-token")
    )

    try:
        client = TestClient(app)
        response = client.get("/api/cases")
        assert response.status_code == 200
        assert response.json() == {"cases": []}
    finally:
        app.dependency_overrides.pop(get_current_user, None)
