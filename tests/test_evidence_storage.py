import hashlib
import io

import pytest

from backend.services.evidence_storage import (
    delete_object, safe_object_path, store_upload, validate_content,
)


def test_store_upload_streams_and_records_integrity(tmp_path):
    content = b"Forensic evidence text\n"
    result = store_upload(io.BytesIO(content), "cases/C1/evidence/E1/original", ".txt", 1024, root=tmp_path)
    saved = tmp_path / "cases/C1/evidence/E1/original"
    assert saved.read_bytes() == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["size_bytes"] == len(content)
    assert result["detected_content_type"] == "text/plain"


def test_store_upload_rejects_mismatched_signature(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        store_upload(io.BytesIO(b"not a pdf"), "bad.pdf", ".pdf", 1024, root=tmp_path)
    assert not (tmp_path / "bad.pdf").exists()


def test_store_upload_rejects_invalid_json(tmp_path):
    with pytest.raises(ValueError, match="valid JSON"):
        store_upload(io.BytesIO(b"{broken"), "bad.json", ".json", 1024, root=tmp_path)


def test_store_upload_enforces_size_limit_and_cleans_temp_file(tmp_path):
    with pytest.raises(OverflowError):
        store_upload(io.BytesIO(b"12345"), "large.txt", ".txt", 4, root=tmp_path)
    assert not (tmp_path / "large.txt").exists()
    assert list(tmp_path.iterdir()) == []


def test_safe_path_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_object_path("../outside.txt", root=tmp_path)


def test_delete_object_removes_only_target(tmp_path):
    target = tmp_path / "cases/C1/E1"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"evidence")
    assert delete_object("cases/C1/E1", root=tmp_path) is True
    assert not target.exists()
    assert delete_object("cases/C1/E1", root=tmp_path) is False


def test_validate_pdf_signature():
    assert validate_content(b"%PDF-1.7", ".pdf") == "application/pdf"
