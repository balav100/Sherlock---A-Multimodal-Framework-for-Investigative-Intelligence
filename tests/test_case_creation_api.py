from fastapi.testclient import TestClient

import backend.routes.cases as cases
from backend.main import app
from backend.security import get_current_user


def test_create_case_saves_case_for_authenticated_user(monkeypatch):
    writes = {}

    class FakeReference:
        def __init__(self, path):
            self.path = path

        def set(self, value):
            writes[self.path] = value

        def update(self, value):
            writes.update(value)

    monkeypatch.setattr(
        cases, "database_ref", lambda path: FakeReference(path)
    )
    app.dependency_overrides[get_current_user] = (
        lambda: {"uid": "test-user", "email": "test@example.com"}
    )

    try:
        client = TestClient(app)
        response = client.post(
            "/api/cases",
            json={
                "case_name": "Test Investigation",
                "case_description": "A test case for API verification.",
            },
        )

        assert response.status_code == 200
        result = response.json()
        case_id = result["case_id"]

        assert case_id.startswith("CASE-")
        assert result["owner_id"] == "test-user"
        assert result["case_name"] == "Test Investigation"
        assert result["evidence_count"] == 0
        assert writes[f"cases/{case_id}"] == result
        assert writes[f"users/test-user/cases/{case_id}"] is True
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_case_create_rejects_whitespace_only_fields():
    from pydantic import ValidationError
    import pytest
    with pytest.raises(ValidationError):
        cases.CaseCreate(case_name="   ", case_description="A valid length description")
    with pytest.raises(ValidationError):
        cases.CaseCreate(case_name="Valid name", case_description="          ")

