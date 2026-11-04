from pathlib import Path

import pytest

from backend.services import hybrid_rag


def test_case_dir_creates_directory_under_index_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    result = hybrid_rag._case_dir("CASE-ABC123")

    assert result == (tmp_path / "CASE-ABC123").resolve()
    assert result.is_dir()
    assert result.is_relative_to(tmp_path.resolve())


def test_case_dir_rejects_empty_case_id(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Case ID cannot be empty"):
        hybrid_rag._case_dir("")


def test_case_dir_rejects_whitespace_case_id(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    with pytest.raises(ValueError, match="Case ID cannot be empty"):
        hybrid_rag._case_dir("   ")


def test_case_dir_sanitization_collision_is_documented(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "INDEX_ROOT", tmp_path)

    first = hybrid_rag._case_dir("CASE/001")
    second = hybrid_rag._case_dir("CASE?001")
    third = hybrid_rag._case_dir("CASE_001")

    # This test records the current behavior; it does not endorse it.
    assert first == second == third
