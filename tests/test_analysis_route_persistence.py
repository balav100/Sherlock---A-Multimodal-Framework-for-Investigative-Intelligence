
import pytest
from fastapi import HTTPException

import backend.routes.analysis as analysis


def test_failed_ingestion_persists_evidence_status(monkeypatch):
    case_id = "CASE-TEST123"
    evidence_id = "EVIDENCE-001"

    records = {
        evidence_id: {
            "case_id": case_id,
            "analysis_status": "pending",
        }
    }
    writes = {}

    class FakeReference:
        def __init__(self, path):
            self.path = path

        def get(self):
            return records

        def set(self, value):
            writes[self.path] = value.copy()

        def update(self, value):
            writes[self.path] = value.copy()

    monkeypatch.setattr(
        analysis, "database_ref", lambda path: FakeReference(path)
    )
    monkeypatch.setattr(
        analysis, "check_case_access", lambda case_id, uid: None
    )

    def fake_ingest(case_id, evidence_records):
        evidence_records[evidence_id]["analysis_status"] = "failed"
        evidence_records[evidence_id]["analysis_error"] = (
            "SHA-256 mismatch. Evidence was not indexed."
        )
        raise RuntimeError(
            "No evidence was indexed successfully."
        )

    monkeypatch.setattr(analysis, "ingest_case", fake_ingest)

    with pytest.raises(HTTPException) as exc_info:
        analysis.ingest_and_index(
            case_id, user={"uid": "test-user"}
        )

    assert exc_info.value.status_code == 500
    assert (
        f"cases/{case_id}/evidence/{evidence_id}" in writes
    )
    assert (
        writes[f"cases/{case_id}/evidence/{evidence_id}"]
        ["analysis_status"]
        == "failed"
    )
