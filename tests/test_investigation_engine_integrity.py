import hashlib

import pytest

from backend.services import investigation_engine as engine


def test_investigation_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "EVIDENCE_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Invalid evidence storage path"):
        engine.resolve_evidence_path("../outside.txt")


def test_investigation_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "EVIDENCE_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="Evidence file not found"):
        engine.resolve_evidence_path("missing.txt")


def test_investigation_rejects_tampered_evidence_before_extraction(
    tmp_path, monkeypatch
):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("modified content", encoding="utf-8")

    monkeypatch.setattr(engine, "EVIDENCE_ROOT", tmp_path)

    def extraction_must_not_run(*args, **kwargs):
        pytest.fail("Extraction ran despite a SHA-256 mismatch")

    monkeypatch.setattr(engine, "extract_text", extraction_must_not_run)

    record = {
        "evidence_id": "EVD-TEST123",
        "storage_path": "evidence.txt",
        "sha256": hashlib.sha256(b"original content").hexdigest(),
        "original_filename": "evidence.txt",
        "modality": "document",
    }

    with pytest.raises(ValueError, match="SHA-256 integrity check failed"):
        engine.analyze_evidence(record)
