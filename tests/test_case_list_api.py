import backend.routes.cases as cases
from fastapi.testclient import TestClient

from backend.main import app
from backend.security import get_current_user


def test_list_cases_returns_only_authenticated_users_cases(monkeypatch):
    user_id = "test-user"

    database = {
        "users/test-user/cases": {
            "CASE-OWN": True,
            "CASE-OTHER": True,
        },
        "cases/CASE-OWN": {
            "case_id": "CASE-OWN",
            "owner_id": "test-user",
            "case_name": "My case",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "cases/CASE-OTHER": {
            "case_id": "CASE-OTHER",
            "owner_id": "different-user",
            "case_name": "Other user's case",
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    }

    class FakeReference:
        def __init__(self, path):
            self.path = path

        def get(self):
            if self.path == "cases":
                return {
                    key.split("/", 1)[1]: value
                    for key, value in database.items()
                    if key.startswith("cases/")
                }
            return database.get(self.path)

    monkeypatch.setattr(
        cases, "database_ref", lambda path: FakeReference(path)
    )
    app.dependency_overrides[get_current_user] = (
        lambda: {"uid": user_id, "email": "test@example.com"}
    )

    try:
        client = TestClient(app)
        response = client.get("/api/cases")

        assert response.status_code == 200
        assert response.json() == {
            "cases": [database["cases/CASE-OWN"]]
        }
    finally:
        app.dependency_overrides.pop(get_current_user, None)
