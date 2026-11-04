import hashlib
import pytest
from backend.services import hybrid_rag


def test_sha256_matches_known_file_content(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"forensic evidence")

    expected = hashlib.sha256(b"forensic evidence").hexdigest()

    assert hybrid_rag._sha256_file(evidence) == expected


def test_sha256_detects_modified_file_content(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes(b"original evidence")
    stored_hash = hybrid_rag._sha256_file(evidence)

    evidence.write_bytes(b"modified evidence")
    actual_hash = hybrid_rag._sha256_file(evidence)

    assert actual_hash != stored_hash


def test_sha256_is_consistent_for_empty_file(tmp_path):
    evidence = tmp_path / "empty.txt"
    evidence.write_bytes(b"")

    assert hybrid_rag._sha256_file(evidence) == hashlib.sha256(b"").hexdigest()


def test_ingestion_rejects_tampered_evidence_before_extraction(
    monkeypatch, tmp_path
):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Modified evidence", encoding="utf-8")

    record = {
        "case_id": "CASE-TEST123456",
        "storage_path": "evidence.txt",
        "sha256": hashlib.sha256(
            b"Original evidence"
        ).hexdigest(),
        "original_filename": "evidence.txt",
    }

    monkeypatch.setattr(hybrid_rag, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(hybrid_rag, "_embedder", lambda: object())

    def extraction_must_not_run(*args, **kwargs):
        pytest.fail("Extraction ran despite a SHA-256 mismatch")

    monkeypatch.setattr(hybrid_rag, "_extract", extraction_must_not_run)

    res = hybrid_rag.ingest_case("CASE-TEST123456", {"E001": record})
    assert res["published"] is False

    assert record["analysis_status"] == "failed"
    assert "SHA-256 mismatch" in record["analysis_error"]


def test_ingestion_rejects_evidence_from_another_case(
    monkeypatch, tmp_path
):
    record = {
        "case_id": "CASE-OTHER123",
        "storage_path": "evidence.txt",
        "sha256": "unused",
    }

    monkeypatch.setattr(hybrid_rag, "_embedder", lambda: object())

    res = hybrid_rag.ingest_case("CASE-TEST123456", {"E001": record})
    assert res["published"] is False

    assert record["analysis_status"] == "failed"
    assert "different case" in record["analysis_error"]
