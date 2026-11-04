
import pytest
from fastapi import HTTPException

import backend.routes.analysis as analysis


def test_ask_route_returns_400_for_invalid_question(monkeypatch):
    monkeypatch.setattr(
        analysis,
        "check_case_access",
        lambda case_id, uid: None,
    )

    def fake_answer_case(**kwargs):
        raise ValueError("Question must contain at least 3 characters.")

    monkeypatch.setattr(
        analysis,
        "answer_case",
        fake_answer_case,
    )
    monkeypatch.setattr(
        analysis,
        "database_ref",
        lambda path: type("FakeRef", (), {"get": lambda self: {"index_status": "ready"}})(),
    )

    payload = analysis.AskRequest(
        question="Valid question?",
        top_k=6,
    )

    with pytest.raises(HTTPException) as exc_info:
        analysis.ask_case(
            "CASE-TEST123",
            payload,
            user={"uid": "test-user"},
        )

    assert exc_info.value.status_code == 400
