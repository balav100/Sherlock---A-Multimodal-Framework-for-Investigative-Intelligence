import io

from fastapi import UploadFile

from backend.routes import evidence


def test_metadata_failure_rolls_back_uploaded_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "check_case_access", lambda case_id, uid: None)
    monkeypatch.setattr(evidence, "uuid4", lambda: type("FakeUUID", (), {"hex": "TEST1234567890"})())

    class FailingReference:
        def get(self):
            return {}
        def set(self, value):
            raise RuntimeError("simulated database failure")

    monkeypatch.setattr(evidence, "database_ref", lambda path: FailingReference())
    upload = UploadFile(filename="evidence.txt", file=io.BytesIO(b"valid text evidence"))
    from fastapi import HTTPException
    import pytest
    with pytest.raises(HTTPException) as exc:
        evidence.upload_evidence("CASE-TEST", upload, {"uid": "user-1"})
    assert exc.value.status_code == 503
    assert not list(tmp_path.rglob("original"))


def test_case_count_failure_does_not_delete_committed_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "check_case_access", lambda case_id, uid: None)
    monkeypatch.setattr(evidence, "uuid4", lambda: type("FakeUUID", (), {"hex": "ABCDEF123456"})())
    saved = {}

    class Ref:
        def __init__(self, path): self.path = path
        def set(self, value): saved.update(value)
        def get(self): return {}
        def update(self, value): raise RuntimeError("simulated count failure")

    monkeypatch.setattr(evidence, "database_ref", lambda path: Ref(path))
    upload = UploadFile(filename="evidence.txt", file=io.BytesIO(b"valid text evidence"))
    result = evidence.upload_evidence("CASE-TEST", upload, {"uid": "user-1"})
    assert result["evidence_id"] == "EVD-ABCDEF123456"
    assert (tmp_path / result["storage_path"]).read_bytes() == b"valid text evidence"
    assert result["count_reconciliation_required"] is True
