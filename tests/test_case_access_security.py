
import pytest
from fastapi import HTTPException

from backend.routes import cases


def test_case_owner_can_access_case(monkeypatch):
    case_data = {
        "case_id": "CASE-TEST123",
        "owner_id": "user-1",
    }

    monkeypatch.setattr(
        cases,
        "database_ref",
        lambda path: type(
            "FakeReference",
            (),
            {"get": lambda self: case_data},
        )(),
    )

    result = cases.check_case_access("CASE-TEST123", "user-1")

    assert result == case_data


def test_different_user_cannot_access_case(monkeypatch):
    case_data = {
        "case_id": "CASE-TEST123",
        "owner_id": "user-1",
    }

    monkeypatch.setattr(
        cases,
        "database_ref",
        lambda path: type(
            "FakeReference",
            (),
            {"get": lambda self: case_data},
        )(),
    )

    with pytest.raises(HTTPException) as exc_info:
        cases.check_case_access("CASE-TEST123", "user-2")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Case not found."


def test_nonexistent_case_returns_404(monkeypatch):
    monkeypatch.setattr(
        cases,
        "database_ref",
        lambda path: type(
            "FakeReference",
            (),
            {"get": lambda self: None},
        )(),
    )

    with pytest.raises(HTTPException) as exc_info:
        cases.check_case_access("CASE-MISSING", "user-1")

    assert exc_info.value.status_code == 404
