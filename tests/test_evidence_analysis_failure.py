
import pytest
from fastapi import HTTPException

from backend.routes import evidence


def test_analyze_case_marks_failure_when_investigation_crashes(
    monkeypatch,
):
    case_id = "CASE-TEST123"
    records = {
        "EVD-001": {
            "evidence_id": "EVD-001",
            "analysis_status": "pending",
        }
    }
    case_updates = []

    class FakeReference:
        def __init__(self, path):
            self.path = path

        def get(self):
            if self.path == f"cases/{case_id}/evidence":
                return records
            if self.path == f"cases/{case_id}":
                return {"status": "created"}
            return None

        def update(self, value):
            if self.path == f"cases/{case_id}":
                case_updates.append(value)

    monkeypatch.setattr(
        evidence,
        "database_ref",
        lambda path: FakeReference(path),
    )
    monkeypatch.setattr(
        evidence,
        "check_case_access",
        lambda case_id, uid: None,
    )

    def fake_ingestion(_case_id, _records):
        raise RuntimeError("Simulated investigation failure")

    monkeypatch.setattr(evidence, "ingest_case", fake_ingestion)

    with pytest.raises(HTTPException) as exc_info:
        evidence.analyze_case(
            case_id,
            user={"uid": "test-user"},
        )

    assert exc_info.value.status_code == 500
    assert {"status": "indexing"} in case_updates
    assert {"status": "analysis_failed"} in case_updates


def test_analyze_case_marks_failure_when_result_persistence_crashes(
    monkeypatch,
):
    case_id = "CASE-TEST123"
    records = {
        "EVD-001": {
            "evidence_id": "EVD-001",
            "analysis_status": "pending",
        }
    }
    case_updates = []

    class FakeReference:
        def __init__(self, path):
            self.path = path

        def get(self):
            if path_is_evidence(self.path):
                return records
            if self.path == f"cases/{case_id}":
                return {"status": "indexing"}
            return None

        def update(self, value):
            if self.path == f"cases/{case_id}":
                case_updates.append(value)
            elif self.path.startswith(f"cases/{case_id}/evidence/"):
                raise RuntimeError("Simulated persistence failure")

    def path_is_evidence(path):
        return path == f"cases/{case_id}/evidence"

    monkeypatch.setattr(
        evidence,
        "database_ref",
        lambda path: FakeReference(path),
    )
    monkeypatch.setattr(
        evidence,
        "check_case_access",
        lambda case_id, uid: None,
    )
    monkeypatch.setattr(
        evidence,
        "ingest_case",
        lambda _case_id, _records: {
            "indexed_chunks": 1,
            "indexed_evidence": 1,
            "failed_evidence": 0,
            "evidence": [],
            "vision_model": "test-vision",
            "reasoning_model": "deterministic_evidence_rules",
            "camera_clock_offset_seconds": 0,
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        evidence.analyze_case(
            case_id,
            user={"uid": "test-user"},
        )

    assert exc_info.value.status_code == 500
    assert {"status": "indexing"} in case_updates
    assert {"status": "analysis_failed"} in case_updates
