import hashlib
from pathlib import Path

import pytest

from backend.services import hybrid_rag


def test_sha256_file_matches_known_hash(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(b"Mr. Sherlock test evidence")

    expected = hashlib.sha256(
        b"Mr. Sherlock test evidence"
    ).hexdigest()

    assert hybrid_rag._sha256_file(test_file) == expected


def test_sha256_file_handles_empty_file(tmp_path):
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")

    assert hybrid_rag._sha256_file(test_file) == hashlib.sha256(b"").hexdigest()


def test_safe_path_rejects_parent_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "EVIDENCE_ROOT", tmp_path)

    with pytest.raises(ValueError):
        hybrid_rag._safe_path("../outside.txt")


def test_safe_path_rejects_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "EVIDENCE_ROOT", tmp_path)

    with pytest.raises(ValueError):
        hybrid_rag._safe_path(str(tmp_path / "outside.txt"))


def test_safe_path_accepts_path_inside_evidence_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_rag, "EVIDENCE_ROOT", tmp_path)

    result = hybrid_rag._safe_path("case_test/evidence.txt")

    assert result == (tmp_path / "case_test" / "evidence.txt").resolve()
    assert result.is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("hello", 5, 1, ["hello"]),
        ("abcdefgh", 4, 1, ["abcd", "defg", "gh"]),
    ],
)
def test_chunks_basic_cases(text, size, overlap, expected):
    assert hybrid_rag._chunks(
        text,
        size=size,
        overlap=overlap,
    ) == expected


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_chunks_rejects_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        hybrid_rag._chunks(
            "sample evidence",
            size=size,
            overlap=overlap,
        )